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

#include <limits>
#include <unordered_map>
#include <vector>

#include "VeloxHashShuffleWriter.h"
#include "shuffle/BufferPool.h"

namespace gluten {

/// Low-memory-footprint hash shuffle writer (bolt BoltShuffleWriterV2 counterpart).
/// Multi-batch accumulate + BufferPool bump allocation + kCacheNoMerge / sequential spill.
/// RowVector-mode assemble is intentionally not ported.
class VeloxHashShuffleWriterV2 final : public VeloxHashShuffleWriter {
 public:
  static arrow::Result<std::shared_ptr<VeloxShuffleWriter>> create(
      uint32_t numPartitions,
      std::unique_ptr<PartitionWriter> partitionWriter,
      ShuffleWriterOptions options,
      std::shared_ptr<facebook::velox::memory::MemoryPool> veloxPool,
      arrow::MemoryPool* arrowPool);

  ~VeloxHashShuffleWriterV2() override;

  arrow::Status write(std::shared_ptr<ColumnarBatch> cb, int64_t memLimit) override;

  arrow::Status stop() override;

  arrow::Status reclaimFixedSize(int64_t size, int64_t* actual) override;

  arrow::Status evictPartitionBuffers(uint32_t partitionId, bool reuseBuffers) override;

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

  arrow::Status initPartitions() override;

  arrow::Status initFromRowVector(const facebook::velox::RowVector& rv) override;

  arrow::Status splitRowVector(const facebook::velox::RowVector& rv) override;

  arrow::Status splitBinaryArray(const facebook::velox::RowVector& rv) override;

  arrow::Status resetValidityBuffer(uint32_t partitionId) override;

  arrow::Result<int64_t> evictPartitionBuffersMinSize(int64_t size) override;

 private:
  static constexpr int64_t kMinMemLimit = 128LL << 20; // 128MB floor, matches bolt.
  static constexpr int32_t kMaxComplexTypePageSize = 1 << 30;

  // PrestoVectorSerializer has INT32_MAX limit; keep combined complex batches below that.
  static constexpr int32_t kMaxCombinedBytesWithComplexType = static_cast<int32_t>(INT32_MAX * 0.7);

  arrow::Status splitSingleBatch(int64_t memLimit);

  arrow::Status splitBatches(int64_t memLimit);

  arrow::Status doSplit(const facebook::velox::RowVector& rv, bool doAlloc, bool doEvict, int64_t memLimit);

  arrow::Status tryEvict(int64_t memLimit = std::numeric_limits<int64_t>::max());

  arrow::Result<bool> sequentialEvictAllPartitions();

  arrow::Status evictFullPartitions();

  arrow::Status evictPartitionBuffers(uint32_t partitionId, Evict::type evictType);

  arrow::Status allocateValidityBufferV2(uint32_t col, uint32_t partitionId, int32_t bytesNeeded);

  arrow::Status allocatePartitionBufferV2(uint32_t partitionId, uint32_t newSize);

  arrow::Status preAllocPartitionBuffers(
      std::vector<uint32_t>& partition2RowCount,
      std::vector<uint32_t>& partitionUsed);

  arrow::Status splitBinaryTypeV2(
      uint32_t binaryIdx,
      const facebook::velox::FlatVector<facebook::velox::StringView>& src,
      std::vector<std::vector<BinaryBuf>>& dst);

  arrow::Status splitFixedWidthValueBufferV2(const facebook::velox::RowVector& rv);

  arrow::Status splitBoolTypeV2(const uint8_t* srcAddr, const std::vector<std::vector<uint8_t*>>& dstAddrs);

  arrow::Status splitValidityBufferV2(const facebook::velox::RowVector& rv);

  template <typename T>
  arrow::Status splitFixedTypeV2(const uint8_t* srcAddr, const std::vector<std::vector<uint8_t*>>& dstAddrs) {
    for (auto& pid : partitionUsed_) {
      auto* dstPidBase = reinterpret_cast<T*>(dstAddrs[pid].back()) + partitionBufferBaseInBatches_[pid];
      auto pos = partition2RowOffsetBase_[pid];
      auto end = partition2RowOffsetBase_[pid + 1];
      for (; pos < end; ++pos) {
        auto rowId = rowOffset2RowId_[pos];
        *dstPidBase++ = reinterpret_cast<const T*>(srcAddr)[rowId];
      }
    }
    return arrow::Status::OK();
  }

  arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>> assembleBuffersGeneral(uint32_t partitionId);

  arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>> assembleBuffersOneBatch(uint32_t partitionId);

  arrow::Status allocateAssembledBuffer(uint64_t size, uint8_t** out) {
    return assembleBufferPool_.allocateFixed(size, out);
  }

  arrow::Status allocateBuffer(uint64_t size, uint8_t** out) {
    return bufferPool_.allocateFixed(size, out);
  }

  arrow::Status initFixedColumnSize();

  uint64_t fixedWidthValueBytes(uint32_t fixedWidthIndex) const;

  BufferPool bufferPool_;
  BumpMemoryPool bumpMemoryPool_;

  std::shared_ptr<ShuffleMemoryPool> assemblePool_;
  BufferPool assembleBufferPool_;

  std::vector<std::vector<std::shared_ptr<arrow::ResizableBuffer>>> partitionValidityBuffers_;
  std::vector<std::vector<std::vector<uint8_t*>>> partitionFixedWidthValueAddrsVector_;
  std::vector<std::vector<std::vector<BinaryBuf>>> partitionBinaryAddrsVector_;
  std::vector<std::vector<std::shared_ptr<arrow::ResizableBuffer>>> partitionBooleanValueBuffers_;

  std::vector<uint16_t> fixedColValueSize_;
  std::vector<uint32_t> partitionBufferBaseInBatches_;
  std::vector<std::vector<uint32_t>> batchNumRows_;
  std::vector<uint32_t> combinedPartition2RowCount_;
  std::vector<uint32_t> combinedPartitionUsed_;

  std::vector<facebook::velox::RowVectorPtr> batches_;
  int32_t numRowsInBatches_{0};
  uint64_t numBytesInBatches_{0};

  std::unordered_map<uint32_t, bool> fullBatch_;
  bool requestSpill_{false};

  std::vector<uint64_t> variableMemoryUsage_;
  uint64_t maxVariableMemoryUsage_{0};
};

} // namespace gluten
