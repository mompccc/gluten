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

#include "VeloxShuffleWriter.h"

namespace gluten {

/// Delays V1/V2 hash writer creation until the first write(), using first-batch
/// metrics (bolt SparkShuffleWriter::init + decideBoltShuffleWriterType).
class VeloxAdaptiveHashShuffleWriter final : public VeloxShuffleWriter {
 public:
  static arrow::Result<std::shared_ptr<VeloxShuffleWriter>> create(
      uint32_t numPartitions,
      std::unique_ptr<PartitionWriter> partitionWriter,
      ShuffleWriterOptions options,
      std::shared_ptr<facebook::velox::memory::MemoryPool> veloxPool,
      arrow::MemoryPool* arrowPool);

  arrow::Status write(std::shared_ptr<ColumnarBatch> cb, int64_t memLimit) override;

  arrow::Status stop() override;

  arrow::Status reclaimFixedSize(int64_t size, int64_t* actual) override;

  int64_t peakBytesAllocated() const override;

  const uint64_t cachedPayloadSize() const override;

 private:
  VeloxAdaptiveHashShuffleWriter(
      uint32_t numPartitions,
      std::unique_ptr<PartitionWriter> partitionWriter,
      ShuffleWriterOptions options,
      std::shared_ptr<facebook::velox::memory::MemoryPool> veloxPool,
      arrow::MemoryPool* arrowPool);

  arrow::Status ensureDelegate(const std::shared_ptr<ColumnarBatch>& cb, int64_t memLimit);

  /// Held until first write creates the real V1/V2 writer.
  std::unique_ptr<PartitionWriter> pendingPartitionWriter_;
  arrow::MemoryPool* arrowPool_;
  std::shared_ptr<VeloxShuffleWriter> delegate_;
};

} // namespace gluten
