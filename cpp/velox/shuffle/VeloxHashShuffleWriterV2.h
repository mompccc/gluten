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

#pragma once

#include "VeloxHashShuffleWriter.h"
#include "shuffle/BufferPool.h"

namespace gluten {

/// Low-memory-footprint hash shuffle writer (bolt BoltShuffleWriterV2 counterpart).
/// Phase-1 skeleton: BufferPool bump allocation + kCacheNoMerge evict.
/// Multi-batch assemble / precompute-fill (3.14) and RowVector mode are follow-ups.
class VeloxHashShuffleWriterV2 final : public VeloxHashShuffleWriter {
 public:
  static arrow::Result<std::shared_ptr<VeloxShuffleWriter>> create(
      uint32_t numPartitions,
      std::unique_ptr<PartitionWriter> partitionWriter,
      ShuffleWriterOptions options,
      std::shared_ptr<facebook::velox::memory::MemoryPool> veloxPool,
      arrow::MemoryPool* arrowPool);

  ~VeloxHashShuffleWriterV2();

 protected:
  VeloxHashShuffleWriterV2(
      uint32_t numPartitions,
      std::unique_ptr<PartitionWriter> partitionWriter,
      ShuffleWriterOptions options,
      std::shared_ptr<facebook::velox::memory::MemoryPool> veloxPool,
      arrow::MemoryPool* pool);

  arrow::Result<std::shared_ptr<arrow::ResizableBuffer>> allocatePartitionResizableBuffer(int64_t size) override;

  Evict::type hashEvictType() const override;

  void onBeforeStopEvict() override;

  void onAfterStop() override;

  void onBeforeEvictPartitionBuffersMinSize() override;

  void onAfterEvictPartitionBuffersMinSize() override;

 private:
  BufferPool bufferPool_;
  BumpMemoryPool bumpMemoryPool_;
};

} // namespace gluten
