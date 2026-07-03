/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "utils/AdaptiveParallelZstdCodec.h"

#include <arrow/buffer.h>
#include <arrow/memory_pool.h>
#include <arrow/util/compression.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <zstd.h>

#include <cstring>
#include <vector>

#include "utils/TestUtils.h"

namespace gluten {
namespace {

constexpr int64_t kParallelThreshold = 2 * 1024 * 1024;

// Compressible data: low-entropy repeating pattern.
std::vector<uint8_t> buildCompressibleData(int64_t size, uint8_t seed) {
  std::vector<uint8_t> data(static_cast<size_t>(size));
  for (int64_t i = 0; i < size; ++i) {
    data[static_cast<size_t>(i)] = static_cast<uint8_t>((seed + i) % 7);
  }
  return data;
}

// Incompressible data: LCG pseudo-random bytes.
std::vector<uint8_t> buildIncompressibleData(int64_t size, uint32_t seed) {
  std::vector<uint8_t> data(static_cast<size_t>(size));
  uint32_t state = seed;
  for (int64_t i = 0; i < size; ++i) {
    state = state * 1664525u + 1013904223u;
    data[static_cast<size_t>(i)] = static_cast<uint8_t>(state >> 24);
  }
  return data;
}

// Compress with `compressor`, decompress with `decompressor`, assert the
// round-trip is byte-identical. Returns actual compressed length via out-param.
//
// arrow::util::Codec API:
//   Result<int64_t> Compress(int64_t inputLen, const uint8_t* input,
//                             int64_t outputLen, uint8_t* output)   -> bytes written
//   Result<int64_t> Decompress(int64_t inputLen, const uint8_t* input,
//                              int64_t outputLen, uint8_t* output)  -> bytes written
void roundTrip(
    const std::vector<uint8_t>& input,
    arrow::util::Codec* compressor,
    arrow::util::Codec* decompressor,
    int64_t& compressedLenOut) {
  int64_t inputLen = static_cast<int64_t>(input.size());
  int64_t maxLen = compressor->MaxCompressedLen(inputLen, input.data());
  std::vector<uint8_t> compressed(static_cast<size_t>(maxLen));

  // Compress returns the ACTUAL number of compressed bytes written.
  ARROW_ASSIGN_OR_THROW(
      int64_t compressedLen,
      compressor->Compress(inputLen, input.data(), maxLen, compressed.data()));
  compressedLenOut = compressedLen;

  // Decompress: pass the true compressed length as inputLen, original size as outputLen.
  std::vector<uint8_t> decompressed(input.empty() ? 1 : input.size());
  ARROW_ASSIGN_OR_THROW(
      int64_t decompressedLen,
      decompressor->Decompress(
          compressedLen, compressed.data(), inputLen, decompressed.data()));
  EXPECT_EQ(decompressedLen, inputLen);
  if (inputLen > 0) {
    ASSERT_EQ(0, std::memcmp(decompressed.data(), input.data(), input.size()));
  }
}

} // namespace

// Test 1: small payload (< 2MB) uses the single-threaded path; round-trip OK.
TEST(AdaptiveParallelZstdCodecTest, SmallPayloadSingleThread) {
  auto codec = makeDefaultAdaptiveParallelZstdCodec();
  auto data = buildCompressibleData(64 * 1024, 7); // 64KB, well below 2MB
  int64_t compressedLen = 0;
  roundTrip(data, codec.get(), codec.get(), compressedLen);
  // Small compressible data must actually compress smaller.
  EXPECT_LT(compressedLen, static_cast<int64_t>(data.size()));
}

// Test 2: large payload (>= 2MB) uses the parallel path; round-trip OK.
TEST(AdaptiveParallelZstdCodecTest, LargePayloadParallel) {
  auto codec = makeDefaultAdaptiveParallelZstdCodec();
  // 2.3MB: crosses the 2MB threshold (matches bolt kLargePayloadSize).
  auto data = buildCompressibleData(2 * 1024 * 1024 + 300 * 1024, 7);
  int64_t compressedLen = 0;
  roundTrip(data, codec.get(), codec.get(), compressedLen);
  EXPECT_LT(compressedLen, static_cast<int64_t>(data.size()));
}

// Test 3: threshold boundary — exactly 2MB (parallel) and 2MB-1 (single).
TEST(AdaptiveParallelZstdCodecTest, ThresholdBoundary) {
  auto codec = makeDefaultAdaptiveParallelZstdCodec();
  // Exactly 2MB -> parallel path.
  {
    auto data = buildCompressibleData(kParallelThreshold, 1);
    int64_t compressedLen = 0;
    roundTrip(data, codec.get(), codec.get(), compressedLen);
  }
  // 2MB - 1 -> single-threaded path.
  {
    auto data = buildCompressibleData(kParallelThreshold - 1, 2);
    int64_t compressedLen = 0;
    roundTrip(data, codec.get(), codec.get(), compressedLen);
  }
}

// Test 4: compress with adaptive codec, decompress with Arrow's default ZSTD.
// Proves output is a standard zstd frame (read side needs no changes).
TEST(AdaptiveParallelZstdCodecTest, CrossCompatAdaptiveCompressArrowDecompress) {
  auto adaptive = makeDefaultAdaptiveParallelZstdCodec();
  ARROW_ASSIGN_OR_THROW(auto arrowCodec, arrow::util::Codec::Create(arrow::Compression::ZSTD));

  // Small buffer (single-thread path).
  {
    auto data = buildCompressibleData(100 * 1024, 3);
    int64_t compressedLen = 0;
    roundTrip(data, adaptive.get(), arrowCodec.get(), compressedLen);
  }
  // Large buffer (parallel path) — the critical cross-compat case.
  {
    auto data = buildCompressibleData(3 * 1024 * 1024, 9);
    int64_t compressedLen = 0;
    roundTrip(data, adaptive.get(), arrowCodec.get(), compressedLen);
  }
}

// Test 5: compress with Arrow's default ZSTD, decompress with adaptive codec.
TEST(AdaptiveParallelZstdCodecTest, CrossCompatArrowCompressAdaptiveDecompress) {
  ARROW_ASSIGN_OR_THROW(auto arrowCodec, arrow::util::Codec::Create(arrow::Compression::ZSTD));
  auto adaptive = makeDefaultAdaptiveParallelZstdCodec();

  auto data = buildCompressibleData(2 * 1024 * 1024 + 100 * 1024, 5);
  int64_t compressedLen = 0;
  roundTrip(data, arrowCodec.get(), adaptive.get(), compressedLen);
}

// Test 6: incompressible data round-trips correctly (codec only compresses;
// the Payload.cc kUncompressedBuffer negative-optimization is separate).
TEST(AdaptiveParallelZstdCodecTest, IncompressibleData) {
  auto codec = makeDefaultAdaptiveParallelZstdCodec();
  auto data = buildIncompressibleData(2 * 1024 * 1024 + 50 * 1024, 0xDEADBEEFu);
  int64_t compressedLen = 0;
  roundTrip(data, codec.get(), codec.get(), compressedLen);
  // Incompressible data may not shrink; we only require correct round-trip.
}

// Test 7: compression level is passed through and reported correctly.
TEST(AdaptiveParallelZstdCodecTest, CompressionLevelPassthrough) {
  { // Default level -> ZSTD_CLEVEL_DEFAULT (3).
    auto codec = makeDefaultAdaptiveParallelZstdCodec();
    EXPECT_EQ(codec->compression_level(), ZSTD_CLEVEL_DEFAULT);
  }
  { // Explicit level 1.
    auto codec = makeAdaptiveParallelZstdCodec(1);
    EXPECT_EQ(codec->compression_level(), 1);
    auto data = buildCompressibleData(2 * 1024 * 1024 + 10, 4);
    int64_t compressedLen = 0;
    roundTrip(data, codec.get(), codec.get(), compressedLen);
  }
  { // Explicit level 9.
    auto codec = makeAdaptiveParallelZstdCodec(9);
    EXPECT_EQ(codec->compression_level(), 9);
    auto data = buildCompressibleData(2 * 1024 * 1024 + 10, 4);
    int64_t compressedLen = 0;
    roundTrip(data, codec.get(), codec.get(), compressedLen);
  }
}

// Test 8: MaxCompressedLen is sane (== ZSTD_compressBound) and >= actual.
TEST(AdaptiveParallelZstdCodecTest, MaxCompressedLenSane) {
  auto codec = makeDefaultAdaptiveParallelZstdCodec();
  for (int64_t n : std::vector<int64_t>{1, 1024, kParallelThreshold, kParallelThreshold + 1}) {
    auto data = buildCompressibleData(n, 8);
    int64_t maxLen = codec->MaxCompressedLen(n, data.data());
    EXPECT_EQ(maxLen, static_cast<int64_t>(ZSTD_compressBound(static_cast<size_t>(n))));
    int64_t compressedLen = 0;
    roundTrip(data, codec.get(), codec.get(), compressedLen);
    EXPECT_LE(compressedLen, maxLen);
  }
}

// Test 9: empty input edge case (0-byte buffer).
TEST(AdaptiveParallelZstdCodecTest, EmptyInput) {
  auto codec = makeDefaultAdaptiveParallelZstdCodec();
  std::vector<uint8_t> empty;
  int64_t maxLen = codec->MaxCompressedLen(0, nullptr);
  EXPECT_GT(maxLen, 0); // zstd frame header overhead
  std::vector<uint8_t> compressed(static_cast<size_t>(maxLen));
  ASSERT_NOT_OK(codec->Compress(0, nullptr, maxLen, compressed.data()));
  // Decompress: output buffer is 0-size; Decompress handles null output.
  ASSERT_NOT_OK(codec->Decompress(maxLen, compressed.data(), 0, nullptr));
}

} // namespace gluten
