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

#include <gtest/gtest.h>

#include <arrow/memory_pool.h>

#include "shuffle/BufferPool.h"
#include "shuffle/ShuffleMemoryPool.h"

namespace gluten {
namespace {

class BufferPoolTest : public ::testing::Test {
 protected:
  arrow::MemoryPool* arrowPool_ = arrow::default_memory_pool();
  ShuffleMemoryPool shufflePool_{arrowPool_};
  BufferPool pool_{&shufflePool_};
};

TEST_F(BufferPoolTest, allocateFixedWithinChunk) {
  uint8_t* p1 = nullptr;
  uint8_t* p2 = nullptr;
  ASSERT_TRUE(pool_.allocateFixed(1024, &p1).ok());
  ASSERT_TRUE(pool_.allocateFixed(2048, &p2).ok());
  EXPECT_NE(p1, nullptr);
  EXPECT_EQ(p2, p1 + 1024);
  EXPECT_EQ(pool_.reservedBytes(), BufferPool::kDefaultBufferSize);
}

TEST_F(BufferPoolTest, allocateFixedAligned64) {
  uint8_t* p = nullptr;
  ASSERT_TRUE(pool_.allocateFixedAligned(128, &p, 64).ok());
  EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 64, 0u);
}

TEST_F(BufferPoolTest, allocateLargeBuffer) {
  const uint64_t largeSize = BufferPool::kDefaultBufferSize + 1;
  uint8_t* p = nullptr;
  ASSERT_TRUE(pool_.allocateFixed(largeSize, &p).ok());
  EXPECT_NE(p, nullptr);
  EXPECT_EQ(pool_.reservedBytes(), largeSize);
}

TEST_F(BufferPoolTest, resetRewindsWithoutFreeing) {
  uint8_t* p1 = nullptr;
  ASSERT_TRUE(pool_.allocateFixed(4096, &p1).ok());
  const auto reservedAfterAlloc = pool_.reservedBytes();
  pool_.reset();
  uint8_t* p2 = nullptr;
  ASSERT_TRUE(pool_.allocateFixed(4096, &p2).ok());
  EXPECT_EQ(p2, p1);
  EXPECT_EQ(pool_.reservedBytes(), reservedAfterAlloc);
}

TEST_F(BufferPoolTest, shrinkReleasesTrailingChunks) {
  uint8_t* p = nullptr;
  ASSERT_TRUE(pool_.allocateFixed(BufferPool::kDefaultBufferSize, &p).ok());
  ASSERT_TRUE(pool_.allocateFixed(1024, &p).ok());
  EXPECT_EQ(pool_.reservedBytes(), 2 * BufferPool::kDefaultBufferSize);
  const auto shrunk = pool_.shrink();
  EXPECT_EQ(shrunk, BufferPool::kDefaultBufferSize);
  EXPECT_EQ(pool_.reservedBytes(), BufferPool::kDefaultBufferSize);
}

TEST_F(BufferPoolTest, clearReleasesAll) {
  uint8_t* p = nullptr;
  ASSERT_TRUE(pool_.allocateFixed(1024, &p).ok());
  ASSERT_TRUE(pool_.allocateFixed(BufferPool::kDefaultBufferSize, &p).ok());
  pool_.clear();
  EXPECT_EQ(pool_.reservedBytes(), 0u);
}

TEST_F(BufferPoolTest, bumpResizableBufferGrow) {
  BufferPool pool(&shufflePool_);
  auto bufResult = BumpResizableBuffer::Allocate(&pool, 256);
  ASSERT_TRUE(bufResult.ok());
  auto buf = std::move(bufResult).ValueOrDie();
  EXPECT_EQ(buf->size(), 256);
  ASSERT_TRUE(buf->Resize(512).ok());
  EXPECT_EQ(buf->size(), 512);
  pool.clear();
}

} // namespace
} // namespace gluten
