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

#include "utils/StreamCodec.h"

#include <arrow/buffer.h>
#include <arrow/memory_pool.h>
#include <arrow/util/compression.h>
#include <gtest/gtest.h>
#include <zstd.h>

#include <cstring>
#include <vector>

#include "utils/TestUtils.h"

namespace gluten {
namespace {

std::vector<uint8_t> buildData(int64_t size, uint8_t seed, bool compressible) {
  std::vector<uint8_t> data(static_cast<size_t>(size));
  if (compressible) {
    for (int64_t i = 0; i < size; ++i) {
      data[static_cast<size_t>(i)] = static_cast<uint8_t>((seed + i) % 7);
    }
  } else {
    uint32_t state = seed;
    for (int64_t i = 0; i < size; ++i) {
      state = state * 1664525u + 1013904223u;
      data[static_cast<size_t>(i)] = static_cast<uint8_t>(state >> 24);
    }
  }
  return data;
}

// Compress via StreamCompressor (ZSTD), decompress via Arrow one-shot Codec.
// Proves streaming output is a standard frame (read side compatible).
void roundTripViaArrowDecompress(const std::vector<uint8_t>& input, int level) {
  auto compressor = StreamCompressor::create(arrow::Compression::ZSTD, level);
  ASSERT_NE(compressor, nullptr);

  // Compress incrementally into a growing buffer.
  int64_t inputLen = static_cast<int64_t>(input.size());
  int64_t outCap = compressor->recommendedOutputSize(inputLen);
  std::vector<uint8_t> compressed(static_cast<size_t>(outCap));

  // Feed all input, draining output as needed.
  const uint8_t* inPtr = input.data();
  int64_t remaining = inputLen;
  int64_t totalWritten = 0;
  while (remaining > 0) {
    int64_t avail = outCap - totalWritten;
    if (avail < 64) { // grow if nearly full
      compressed.resize(compressed.size() + 65536);
      outCap = static_cast<int64_t>(compressed.size());
      avail = outCap - totalWritten;
    }
    auto r = compressor->compress(inPtr, remaining, compressed.data() + totalWritten, avail);
    ASSERT_GT(r.bytesRead, 0) << "compress made no progress";
    inPtr += r.bytesRead;
    remaining -= r.bytesRead;
    totalWritten += r.bytesWritten;
  }
  // End stream (may need multiple calls).
  while (true) {
    int64_t avail = outCap - totalWritten;
    if (avail < 64) {
      compressed.resize(compressed.size() + 65536);
      outCap = static_cast<int64_t>(compressed.size());
      avail = outCap - totalWritten;
    }
    auto e = compressor->end(compressed.data() + totalWritten, avail);
    totalWritten += e.bytesWritten;
    if (e.noMoreOutput) {
      break;
    }
  }
  compressed.resize(static_cast<size_t>(totalWritten));

  // Decompress with Arrow one-shot ZSTD (read-side compatibility).
  ARROW_ASSIGN_OR_THROW(auto arrowCodec, arrow::util::Codec::Create(arrow::Compression::ZSTD, level));
  std::vector<uint8_t> decompressed(input.size());
  ARROW_ASSIGN_OR_THROW(
      int64_t decompressedLen,
      arrowCodec->Decompress(
          static_cast<int64_t>(compressed.size()), compressed.data(),
          inputLen, decompressed.data()));
  EXPECT_EQ(decompressedLen, inputLen);
  ASSERT_EQ(0, std::memcmp(decompressed.data(), input.data(), input.size()));
}

} // namespace

TEST(StreamCodecTest, ZstdRoundTripSmall) {
  auto data = buildData(64 * 1024, 7, /*compressible=*/true);
  roundTripViaArrowDecompress(data, 3);
}

TEST(StreamCodecTest, ZstdRoundTripLarge) {
  auto data = buildData(3 * 1024 * 1024, 9, /*compressible=*/true);
  roundTripViaArrowDecompress(data, 3);
}

TEST(StreamCodecTest, ZstdRoundTripIncompressible) {
  auto data = buildData(1024 * 1024, 0xCAFEBABEu, /*compressible=*/false);
  roundTripViaArrowDecompress(data, 3);
}

TEST(StreamCodecTest, ZstdRoundTripLevel1) {
  auto data = buildData(512 * 1024, 3, /*compressible=*/true);
  roundTripViaArrowDecompress(data, 1);
}

TEST(StreamCodecTest, UnsupportedCodecReturnsNull) {
  // GZIP not supported by StreamCodec (only ZSTD/LZ4); expect nullptr fallback.
  auto compressor = StreamCompressor::create(arrow::Compression::GZIP, 3);
  EXPECT_EQ(compressor, nullptr);
}

} // namespace gluten
