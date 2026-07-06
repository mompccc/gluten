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

#include "shuffle/Payload.h"

#include <arrow/buffer.h>
#include <arrow/io/memory.h>
#include <arrow/memory_pool.h>
#include <arrow/util/compression.h>
#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "utils/TestUtils.h"

namespace gluten {
namespace {

// Build a vector of column buffers simulating a wide row (many columns).
// Each buffer is filled with a repeating pattern (compressible).
std::vector<std::shared_ptr<arrow::Buffer>> buildColumnBuffers(
    int numColumns,
    int64_t bytesPerColumn,
    uint8_t seed) {
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.reserve(static_cast<size_t>(numColumns));
  for (int c = 0; c < numColumns; ++c) {
    auto buf = arrow::AllocateResizableBuffer(bytesPerColumn, arrow::default_memory_pool()).ValueOrDie();
    for (int64_t i = 0; i < bytesPerColumn; ++i) {
      buf->mutable_data()[i] = static_cast<uint8_t>((seed + c + i) % 13);
    }
    buffers.push_back(std::move(buf));
  }
  return buffers;
}

// Round-trip test: write buffers via BlockPayload (RowVector mode), read back,
// verify the decompressed buffers match the originals byte-for-byte.
void roundTripRowVector(
    const std::vector<std::shared_ptr<arrow::Buffer>>& originalBuffers,
    arrow::Compression::type codecType,
    int compressionLevel) {
  ARROW_ASSIGN_OR_THROW(auto codec, arrow::util::Codec::Create(codecType, compressionLevel));
  auto pool = arrow::default_memory_pool();

  uint32_t numRows = 1000;
  std::vector<bool> isValidityBuffer(originalBuffers.size(), false);

  // Write: fromBuffers with RowVector mode -> serialize to a BufferOutputStream.
  ARROW_ASSIGN_OR_THROW(
      auto payload,
      BlockPayload::fromBuffers(
          Payload::Type::kCompressed,
          numRows,
          std::vector<std::shared_ptr<arrow::Buffer>>(originalBuffers), // copy
          &isValidityBuffer,
          pool,
          codec.get(),
          nullptr,
          PayloadMode::kRowVector));

  ARROW_ASSIGN_OR_THROW(auto outStream, arrow::io::BufferOutputStream::Create(1024, pool));
  ASSERT_NOT_OK(payload->serialize(outStream.get()));
  ARROW_ASSIGN_OR_THROW(auto writtenBuffer, outStream->Finish());

  // Read: deserialize from the written buffer.
  auto inStream = std::make_shared<arrow::io::BufferReader>(writtenBuffer);
  uint32_t readNumRows = 0;
  int64_t deserializeTime = 0;
  int64_t decompressTime = 0;
  ARROW_ASSIGN_OR_THROW(
      auto readBuffers,
      BlockPayload::deserialize(inStream, codec, pool, readNumRows, deserializeTime, decompressTime));

  // Verify.
  EXPECT_EQ(readNumRows, numRows);
  ASSERT_EQ(readBuffers.size(), originalBuffers.size());
  for (size_t i = 0; i < originalBuffers.size(); ++i) {
    if (originalBuffers[i] == nullptr) {
      EXPECT_EQ(readBuffers[i], nullptr);
    } else if (originalBuffers[i]->size() == 0) {
      // empty buffer
    } else {
      ASSERT_NE(readBuffers[i], nullptr) << "buffer " << i << " is null";
      ASSERT_EQ(readBuffers[i]->size(), originalBuffers[i]->size())
          << "buffer " << i << " size mismatch";
      ASSERT_EQ(0, std::memcmp(readBuffers[i]->data(), originalBuffers[i]->data(), originalBuffers[i]->size()))
          << "buffer " << i << " content mismatch";
    }
  }
}

} // namespace

// Test 1: RowVector round-trip with ZSTD, 25 columns (above threshold 20).
TEST(RowVectorModeTest, ZstdRoundTrip25Columns) {
  auto buffers = buildColumnBuffers(25, 4096, 7);
  roundTripRowVector(buffers, arrow::Compression::ZSTD, 3);
}

// Test 2: RowVector round-trip with LZ4, 25 columns.
TEST(RowVectorModeTest, Lz4RoundTrip25Columns) {
  auto buffers = buildColumnBuffers(25, 4096, 9);
  roundTripRowVector(buffers, arrow::Compression::LZ4_FRAME, 0);
}

// Test 3: RowVector with null and empty buffers interspersed.
TEST(RowVectorModeTest, RowVectorWithNullAndEmptyBuffers) {
  auto buffers = buildColumnBuffers(22, 2048, 3);
  buffers[5] = nullptr;               // null buffer
  buffers[10] = arrow::AllocateResizableBuffer(0, arrow::default_memory_pool()).ValueOrDie(); // empty
  roundTripRowVector(buffers, arrow::Compression::ZSTD, 3);
}

// Test 4: BUFFER mode round-trip (mode=kBuffer) still works after header change.
TEST(RowVectorModeTest, BufferModeRoundTripStillWorks) {
  auto buffers = buildColumnBuffers(5, 4096, 1); // few columns -> BUFFER mode
  ARROW_ASSIGN_OR_THROW(auto codec, arrow::util::Codec::Create(arrow::Compression::ZSTD, 3));
  auto pool = arrow::default_memory_pool();
  uint32_t numRows = 1000;
  std::vector<bool> isValidityBuffer(buffers.size(), false);

  ARROW_ASSIGN_OR_THROW(
      auto payload,
      BlockPayload::fromBuffers(
          Payload::Type::kCompressed, numRows, std::vector<std::shared_ptr<arrow::Buffer>>(buffers),
          &isValidityBuffer, pool, codec.get(), nullptr, PayloadMode::kBuffer));

  ARROW_ASSIGN_OR_THROW(auto outStream, arrow::io::BufferOutputStream::Create(1024, pool));
  ASSERT_NOT_OK(payload->serialize(outStream.get()));
  ARROW_ASSIGN_OR_THROW(auto writtenBuffer, outStream->Finish());

  auto inStream = std::make_shared<arrow::io::BufferReader>(writtenBuffer);
  uint32_t readNumRows = 0;
  int64_t dt = 0, ddt = 0;
  ARROW_ASSIGN_OR_THROW(
      auto readBuffers, BlockPayload::deserialize(inStream, codec, pool, readNumRows, dt, ddt));

  EXPECT_EQ(readNumRows, numRows);
  ASSERT_EQ(readBuffers.size(), buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    ASSERT_EQ(readBuffers[i]->size(), buffers[i]->size());
    ASSERT_EQ(0, std::memcmp(readBuffers[i]->data(), buffers[i]->data(), buffers[i]->size()));
  }
}

} // namespace gluten
