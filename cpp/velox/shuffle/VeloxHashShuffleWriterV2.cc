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

#include "VeloxHashShuffleWriterV2.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <unistd.h>

#include <arrow/buffer.h>
#include <arrow/io/memory.h>
#include <arrow/util/bit_util.h>
#include <glog/logging.h>

#include "memory/ArrowMemory.h"
#include "memory/VeloxColumnarBatch.h"
#include "shuffle/Utils.h"
#include "utils/Common.h"
#include "utils/Macros.h"
#include "velox/common/base/Nulls.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/type/Timestamp.h"

#if defined(__x86_64__)
#include <immintrin.h>
#include <x86intrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace gluten {

namespace {

constexpr const char* kV2MarkerFile = "/tmp/gluten_hash_shuffle_v2.log";

#if !defined(__x86_64__)
#define rotateLeft(x, s) (x << (s - ((s >> 3) << 3)) | x >> (8 - (s - ((s >> 3) << 3))))
#endif

void logV2WriterActivated(int64_t taskAttemptId, uint32_t numPartitions, int32_t bufferSize) {
  LOG(INFO) << "[VeloxHashShuffleWriterV2] activated"
            << " taskAttemptId=" << taskAttemptId << " pid=" << getpid()
            << " numPartitions=" << numPartitions << " bufferSize=" << bufferSize
            << " evict=kCacheNoMerge multiBatch=1";
  std::ofstream marker(kV2MarkerFile, std::ios::app);
  if (marker) {
    marker << "taskAttemptId=" << taskAttemptId << " pid=" << getpid()
           << " partitions=" << numPartitions << " bufferSize=" << bufferSize << '\n';
  }
}

bool vectorHasNull(const facebook::velox::VectorPtr& vp) {
  if (!vp->mayHaveNulls()) {
    return false;
  }
  return vp->countNulls(vp->nulls(), vp->size()) != 0;
}

} // namespace

VeloxHashShuffleWriterV2::~VeloxHashShuffleWriterV2() {
  bufferPool_.clear();
  assembleBufferPool_.clear();
}

arrow::Result<std::shared_ptr<VeloxShuffleWriter>> VeloxHashShuffleWriterV2::create(
    uint32_t numPartitions,
    std::unique_ptr<PartitionWriter> partitionWriter,
    ShuffleWriterOptions options,
    std::shared_ptr<facebook::velox::memory::MemoryPool> veloxPool,
    arrow::MemoryPool* arrowPool) {
  const auto taskAttemptId = options.taskAttemptId;
  const auto bufferSize = options.bufferSize;
  std::shared_ptr<VeloxHashShuffleWriterV2> res(new VeloxHashShuffleWriterV2(
      numPartitions, std::move(partitionWriter), std::move(options), std::move(veloxPool), arrowPool));
  RETURN_NOT_OK(res->init());
  res->partitionBufferBaseInBatches_.resize(numPartitions);
  res->variableMemoryUsage_.resize(numPartitions, 0);
  logV2WriterActivated(taskAttemptId, numPartitions, bufferSize);
  return res;
}

VeloxHashShuffleWriterV2::VeloxHashShuffleWriterV2(
    uint32_t numPartitions,
    std::unique_ptr<PartitionWriter> partitionWriter,
    ShuffleWriterOptions options,
    std::shared_ptr<facebook::velox::memory::MemoryPool> veloxPool,
    arrow::MemoryPool* pool)
    : VeloxHashShuffleWriter(
          numPartitions,
          std::move(partitionWriter),
          std::move(options),
          std::move(veloxPool),
          pool),
      bufferPool_(partitionBufferPool_.get()),
      bumpMemoryPool_(&bufferPool_),
      assemblePool_(std::make_shared<ShuffleMemoryPool>(pool)),
      assembleBufferPool_(assemblePool_.get()) {}

arrow::Result<std::shared_ptr<arrow::ResizableBuffer>> VeloxHashShuffleWriterV2::allocatePartitionResizableBuffer(
    int64_t size) {
  // Validity stays on the Arrow pool; value/length use BufferPool bump via allocateBuffer.
  return arrow::AllocateResizableBuffer(size, partitionBufferPool_.get());
}

Evict::type VeloxHashShuffleWriterV2::hashEvictType() const {
  return Evict::kCacheNoMerge;
}

void VeloxHashShuffleWriterV2::onBeforeStopEvict() {
  bufferPool_.shrink();
}

void VeloxHashShuffleWriterV2::onAfterStop() {
  bufferPool_.clear();
  assembleBufferPool_.clear();
}

void VeloxHashShuffleWriterV2::onBeforeEvictPartitionBuffersMinSize() {
  bufferPool_.shrink();
}

void VeloxHashShuffleWriterV2::onAfterEvictPartitionBuffersMinSize() {
  bufferPool_.clear();
}

uint64_t VeloxHashShuffleWriterV2::fixedWidthValueBytes(uint32_t fixedWidthIndex) const {
  auto columnIdx = simpleColumnIndices_[fixedWidthIndex];
  if (arrowColumnTypes_[columnIdx]->id() == arrow::BooleanType::type_id) {
    return 0;
  }
  if (veloxColumnTypes_[columnIdx]->isShortDecimal()) {
    return arrow::bit_width(arrow::Int64Type::type_id) >> 3;
  }
  if (veloxColumnTypes_[columnIdx]->kind() == facebook::velox::TypeKind::TIMESTAMP) {
    return facebook::velox::BaseVector::byteSize<facebook::velox::Timestamp>(1);
  }
  return arrow::bit_width(arrowColumnTypes_[columnIdx]->id()) >> 3;
}

arrow::Status VeloxHashShuffleWriterV2::initFixedColumnSize() {
  uint32_t numberBooleanColumns = 0;
  fixedColValueSize_.resize(fixedWidthColumnCount_);
  for (size_t i = 0; i < fixedWidthColumnCount_; ++i) {
    auto columnType = arrowColumnTypes_[simpleColumnIndices_[i]]->id();
    if (columnType == arrow::BooleanType::type_id) {
      ++numberBooleanColumns;
      fixedColValueSize_[i] = 0;
    } else {
      fixedColValueSize_[i] = static_cast<uint16_t>(fixedWidthValueBytes(i));
    }
  }
  if (numberBooleanColumns) {
    partitionBooleanValueBuffers_.resize(numberBooleanColumns);
    std::for_each(
        partitionBooleanValueBuffers_.begin(),
        partitionBooleanValueBuffers_.end(),
        [this](std::vector<std::shared_ptr<arrow::ResizableBuffer>>& v) { v.resize(numPartitions_); });
  }
  return arrow::Status::OK();
}

arrow::Status VeloxHashShuffleWriterV2::initPartitions() {
  auto simpleColumnCount = simpleColumnIndices_.size();

  partitionValidityAddrs_.resize(simpleColumnCount);
  std::for_each(partitionValidityAddrs_.begin(), partitionValidityAddrs_.end(), [this](std::vector<uint8_t*>& v) {
    v.resize(numPartitions_, nullptr);
  });

  partitionFixedWidthValueAddrsVector_.resize(fixedWidthColumnCount_);
  std::for_each(
      partitionFixedWidthValueAddrsVector_.begin(),
      partitionFixedWidthValueAddrsVector_.end(),
      [this](std::vector<std::vector<uint8_t*>>& v) { v.resize(numPartitions_); });

  partitionBinaryAddrsVector_.resize(binaryColumnIndices_.size());
  std::for_each(
      partitionBinaryAddrsVector_.begin(),
      partitionBinaryAddrsVector_.end(),
      [this](std::vector<std::vector<BinaryBuf>>& v) { v.resize(numPartitions_); });

  partitionValidityBuffers_.resize(simpleColumnCount);
  std::for_each(
      partitionValidityBuffers_.begin(),
      partitionValidityBuffers_.end(),
      [this](std::vector<std::shared_ptr<arrow::ResizableBuffer>>& v) { v.resize(numPartitions_); });

  batchNumRows_.resize(numPartitions_);
  combinedPartition2RowCount_.resize(numPartitions_);
  return arrow::Status::OK();
}

arrow::Status VeloxHashShuffleWriterV2::initFromRowVector(const facebook::velox::RowVector& rv) {
  if (veloxColumnTypes_.empty()) {
    RETURN_NOT_OK(initColumnTypes(rv));
    RETURN_NOT_OK(initPartitions());
    RETURN_NOT_OK(initFixedColumnSize());
    calculateSimpleColumnBytes();
  }
  return arrow::Status::OK();
}

arrow::Status VeloxHashShuffleWriterV2::write(std::shared_ptr<ColumnarBatch> cb, int64_t memLimit) {
  VELOX_CHECK(options_.partitioning != Partitioning::kSingle, "V2 writer does not support single partitioning");

  memLimit = std::max(memLimit, kMinMemLimit);

  auto veloxColumnBatch = VeloxColumnarBatch::from(veloxPool_.get(), cb);
  VELOX_CHECK_NOT_NULL(veloxColumnBatch);

  facebook::velox::RowVectorPtr rv;
  START_TIMING(cpuWallTimingList_[CpuWallTimingFlattenRV]);
  rv = veloxColumnBatch->getFlattenedRowVector();
  END_TIMING();

  auto flatSize = rv->estimateFlatSize();
  if (flatSize > static_cast<uint64_t>(memLimit)) {
    requestSpill_ = true;
  }
  RETURN_NOT_OK(tryEvict(memLimit));

  auto maxBatchBytes = static_cast<uint64_t>(std::max(memLimit, kMinMemLimit) / 2);
  if (hasComplexType_) {
    maxBatchBytes = std::min(maxBatchBytes, static_cast<uint64_t>(kMaxCombinedBytesWithComplexType));
  }

  if (isExtremelyLargeBatch(rv) || flatSize > maxBatchBytes) {
    // Flush any accumulated batches first, then slice and split via V2 path.
    if (!batches_.empty()) {
      if (batches_.size() == 1) {
        RETURN_NOT_OK(splitSingleBatch(memLimit));
      } else {
        RETURN_NOT_OK(splitBatches(memLimit));
      }
    }
    auto numRows = rv->size();
    int32_t offset = 0;
    // Prefer row-based slice size; fall back to ~half memLimit worth of rows.
    int32_t sliceRows = maxBatchSize_ > 0 ? maxBatchSize_ : std::max<int32_t>(1, options_.bufferSize);
    do {
      auto length = std::min(sliceRows, numRows);
      auto slicedBatch = std::dynamic_pointer_cast<facebook::velox::RowVector>(rv->slice(offset, length));
      batches_.emplace_back(std::move(slicedBatch));
      numRowsInBatches_ = length;
      numBytesInBatches_ = 0;
      RETURN_NOT_OK(splitSingleBatch(memLimit));
      offset += length;
      numRows -= length;
    } while (numRows > 0);
    return arrow::Status::OK();
  }

  numRowsInBatches_ += rv->size();
  numBytesInBatches_ += flatSize;
  batches_.emplace_back(std::move(rv));

  if (options_.enableVectorCombination &&
      (!hasComplexType_ || numBytesInBatches_ < static_cast<uint64_t>(kMaxCombinedBytesWithComplexType)) &&
      numRowsInBatches_ < options_.bufferSize && numBytesInBatches_ < static_cast<uint64_t>(memLimit * 0.25) &&
      !veloxColumnTypes_.empty()) {
    return arrow::Status::OK();
  }

  if (batches_.size() == 1) {
    RETURN_NOT_OK(splitSingleBatch(memLimit));
  } else {
    RETURN_NOT_OK(splitBatches(memLimit));
  }
  return arrow::Status::OK();
}

arrow::Status VeloxHashShuffleWriterV2::splitSingleBatch(int64_t memLimit) {
  auto rv = batches_[0];
  std::fill(partitionBufferBaseInBatches_.begin(), partitionBufferBaseInBatches_.end(), 0);
  std::fill(partition2RowCount_.begin(), partition2RowCount_.end(), 0);

  facebook::velox::RowVectorPtr splitRv;
  START_TIMING(cpuWallTimingList_[CpuWallTimingCompute]);
  if (partitioner_->hasPid()) {
    auto pidArr = getFirstColumn(*rv);
    RETURN_NOT_OK(partitioner_->compute(pidArr, rv->size(), row2Partition_));
    for (auto& pid : row2Partition_) {
      partition2RowCount_[pid]++;
    }
    splitRv = getStrippedRowVector(*rv);
  } else {
    RETURN_NOT_OK(partitioner_->compute(nullptr, rv->size(), row2Partition_));
    for (auto& pid : row2Partition_) {
      partition2RowCount_[pid]++;
    }
    splitRv = rv;
  }
  END_TIMING();

  RETURN_NOT_OK(initFromRowVector(*splitRv));
  RETURN_NOT_OK(doSplit(*splitRv, true, true, memLimit));
  setSplitState(SplitState::kInit);
  batches_.clear();
  numRowsInBatches_ = 0;
  numBytesInBatches_ = 0;
  return arrow::Status::OK();
}

arrow::Status VeloxHashShuffleWriterV2::splitBatches(int64_t memLimit) {
  // Multi-batch precompute/fill requires HashPartitioner.
  // RoundRobin / Range / others: flush one-by-one via splitSingleBatch.
  if (!partitioner_->hasPid() || options_.partitioning != Partitioning::kHash) {
    auto pending = std::move(batches_);
    batches_.clear();
    numRowsInBatches_ = 0;
    numBytesInBatches_ = 0;
    for (auto& batch : pending) {
      batches_.emplace_back(std::move(batch));
      numRowsInBatches_ = batches_.back()->size();
      RETURN_NOT_OK(splitSingleBatch(memLimit));
    }
    return arrow::Status::OK();
  }

  START_TIMING(cpuWallTimingList_[CpuWallTimingCompute]);
  for (int32_t i = 0; i < static_cast<int32_t>(batches_.size()); ++i) {
    auto batch = batches_[i];
    auto* pidAddr = const_cast<int32_t*>(getFirstColumn(*batch));
    RETURN_NOT_OK(partitioner_->precompute(pidAddr, batch->size(), combinedPartition2RowCount_, (i == 0)));
  }
  END_TIMING();

  START_TIMING(cpuWallTimingList_[CpuWallTimingIteratePartitions]);
  setSplitState(SplitState::kPreAlloc);
  combinedPartitionUsed_.clear();
  for (auto pid = 0; pid < numPartitions_; ++pid) {
    if (combinedPartition2RowCount_[pid] > 0) {
      combinedPartitionUsed_.push_back(pid);
    }
  }
  for (int32_t i = 0; i < static_cast<int32_t>(batches_.size()); ++i) {
    // Batches still contain the pid column at index 0.
    RETURN_NOT_OK(updateInputHasNull(*(batches_[i]), 1));
  }
  RETURN_NOT_OK(preAllocPartitionBuffers(combinedPartition2RowCount_, combinedPartitionUsed_));
  END_TIMING();

  std::fill(partitionBufferBaseInBatches_.begin(), partitionBufferBaseInBatches_.end(), 0);
  for (int32_t i = 0; i < static_cast<int32_t>(batches_.size()); ++i) {
    auto batch = batches_[i];
    auto pidAddr = getFirstColumn(*batch);
    RETURN_NOT_OK(partitioner_->fill(pidAddr, batch->size(), row2Partition_, partition2RowCount_));
    auto strippedRv = getStrippedRowVector(*batch);
    RETURN_NOT_OK(initFromRowVector(*strippedRv));
    RETURN_NOT_OK(doSplit(*strippedRv, false, (i == static_cast<int32_t>(batches_.size()) - 1), memLimit));
    batches_[i] = nullptr;
  }
  setSplitState(SplitState::kInit);
  batches_.clear();
  numRowsInBatches_ = 0;
  numBytesInBatches_ = 0;
  return arrow::Status::OK();
}

arrow::Status VeloxHashShuffleWriterV2::doSplit(
    const facebook::velox::RowVector& rv,
    bool doAlloc,
    bool doEvict,
    int64_t memLimit) {
  auto rowNum = rv.size();
  RETURN_NOT_OK(buildPartition2Row(rowNum));

  if (doAlloc) {
    RETURN_NOT_OK(updateInputHasNull(rv));
    START_TIMING(cpuWallTimingList_[CpuWallTimingIteratePartitions]);
    setSplitState(SplitState::kPreAlloc);
    RETURN_NOT_OK(preAllocPartitionBuffers(partition2RowCount_, partitionUsed_));
    END_TIMING();
  }

  setSplitState(SplitState::kSplit);
  RETURN_NOT_OK(splitRowVector(rv));

  if (doEvict) {
    RETURN_NOT_OK(tryEvict(memLimit));
  }
  return arrow::Status::OK();
}

arrow::Status VeloxHashShuffleWriterV2::preAllocPartitionBuffers(
    std::vector<uint32_t>& partition2RowCount,
    std::vector<uint32_t>& partitionUsed) {
  for (auto& pid : partitionUsed) {
    auto pRowNum = partition2RowCount[pid];
    RETURN_NOT_OK(allocatePartitionBufferV2(pid, pRowNum));
    batchNumRows_[pid].push_back(pRowNum);
  }
  return arrow::Status::OK();
}

arrow::Status VeloxHashShuffleWriterV2::allocateValidityBufferV2(
    uint32_t col,
    uint32_t partitionId,
    int32_t bytesNeeded) {
  if (inputHasNull_[col]) {
    auto& partitionValidityBuffer = partitionValidityBuffers_[col][partitionId];
    if (partitionValidityBuffer == nullptr) {
      ARROW_ASSIGN_OR_RAISE(
          auto validityBuffer, arrow::AllocateResizableBuffer(bytesNeeded, partitionBufferPool_.get()));
      memset(validityBuffer->mutable_data(), 0xFF, validityBuffer->capacity());
      partitionValidityAddrs_[col][partitionId] = validityBuffer->mutable_data();
      partitionValidityBuffer = std::move(validityBuffer);
    } else if (partitionValidityBuffer->size() < bytesNeeded) {
      int32_t oldSize = static_cast<int32_t>(partitionValidityBuffer->size());
      RETURN_NOT_OK(partitionValidityBuffer->Resize(bytesNeeded));
      auto delta = partitionValidityBuffer->capacity() - oldSize;
      memset(partitionValidityBuffer->mutable_data() + oldSize, 0xFF, delta);
      partitionValidityAddrs_[col][partitionId] = partitionValidityBuffer->mutable_data();
    } else if (partitionValidityBuffer->mutable_data() != partitionValidityAddrs_[col][partitionId]) {
      VELOX_CHECK(partitionValidityAddrs_[col][partitionId] == nullptr);
      partitionValidityAddrs_[col][partitionId] = partitionValidityBuffer->mutable_data();
    }
  }
  return arrow::Status::OK();
}

arrow::Status VeloxHashShuffleWriterV2::allocatePartitionBufferV2(uint32_t partitionId, uint32_t newSize) {
  SCOPED_TIMER(cpuWallTimingList_[CpuWallTimingAllocateBuffer]);
  uint32_t booleanColumnIndex = 0;
  uint32_t currentRowNum = partitionBufferBase_[partitionId];

  for (auto i = 0; i < fixedWidthColumnCount_; ++i) {
    auto columnType = schema_->field(simpleColumnIndices_[i])->type()->id();
    if (columnType == arrow::BooleanType::type_id) {
      int32_t bytesNeeded = arrow::bit_util::BytesForBits(currentRowNum + newSize);
      auto& booleanValueBuffer = partitionBooleanValueBuffers_[booleanColumnIndex][partitionId];
      if (booleanValueBuffer == nullptr) {
        auto numBytes = bytesNeeded * 4;
        ARROW_ASSIGN_OR_RAISE(
            booleanValueBuffer, arrow::AllocateResizableBuffer(numBytes, partitionBufferPool_.get()));
        VELOX_CHECK(partitionFixedWidthValueAddrsVector_[i][partitionId].empty());
        partitionFixedWidthValueAddrsVector_[i][partitionId].push_back(booleanValueBuffer->mutable_data());
      } else if (booleanValueBuffer->size() < bytesNeeded) {
        int32_t oldSize = static_cast<int32_t>(booleanValueBuffer->size());
        auto numBytes = std::max(bytesNeeded, 2 * oldSize);
        RETURN_NOT_OK(booleanValueBuffer->Resize(numBytes));
        VELOX_CHECK_EQ(partitionFixedWidthValueAddrsVector_[i][partitionId].size(), 1);
        partitionFixedWidthValueAddrsVector_[i][partitionId][0] = booleanValueBuffer->mutable_data();
      }
      booleanColumnIndex++;
    } else {
      int64_t valueBufferSize = static_cast<int64_t>(newSize) * fixedColValueSize_[i];
      uint8_t* valueBuffer = nullptr;
      RETURN_NOT_OK(allocateBuffer(valueBufferSize, &valueBuffer));
      partitionFixedWidthValueAddrsVector_[i][partitionId].push_back(valueBuffer);
    }
  }

  int32_t bytesNeeded = arrow::bit_util::BytesForBits(currentRowNum + newSize);
  for (auto i = 0; i < static_cast<int>(simpleColumnIndices_.size()); ++i) {
    RETURN_NOT_OK(allocateValidityBufferV2(i, partitionId, bytesNeeded));
  }
  return arrow::Status::OK();
}

arrow::Status VeloxHashShuffleWriterV2::splitFixedWidthValueBufferV2(const facebook::velox::RowVector& rv) {
  for (auto col = 0; col < fixedWidthColumnCount_; ++col) {
    auto colIdx = simpleColumnIndices_[col];
    auto& column = rv.childAt(colIdx);
    const uint8_t* srcAddr = static_cast<const uint8_t*>(column->valuesAsVoid());
    const auto& dstAddrs = partitionFixedWidthValueAddrsVector_[col];

    switch (arrow::bit_width(arrowColumnTypes_[colIdx]->id())) {
      case 0:
        break;
      case 1:
        RETURN_NOT_OK(splitBoolTypeV2(srcAddr, dstAddrs));
        break;
      case 8:
        RETURN_NOT_OK(splitFixedTypeV2<uint8_t>(srcAddr, dstAddrs));
        break;
      case 16:
        RETURN_NOT_OK(splitFixedTypeV2<uint16_t>(srcAddr, dstAddrs));
        break;
      case 32:
        RETURN_NOT_OK(splitFixedTypeV2<uint32_t>(srcAddr, dstAddrs));
        break;
      case 64: {
        if (column->type()->kind() == facebook::velox::TypeKind::TIMESTAMP) {
          RETURN_NOT_OK(splitFixedTypeV2<facebook::velox::int128_t>(srcAddr, dstAddrs));
        } else {
          RETURN_NOT_OK(splitFixedTypeV2<uint64_t>(srcAddr, dstAddrs));
        }
      } break;
      case 128: {
        if (column->type()->isShortDecimal()) {
          RETURN_NOT_OK(splitFixedTypeV2<int64_t>(srcAddr, dstAddrs));
        } else if (column->type()->isLongDecimal()) {
          RETURN_NOT_OK(splitFixedTypeV2<facebook::velox::int128_t>(srcAddr, dstAddrs));
        } else {
          return arrow::Status::Invalid(
              "Column type " + schema_->field(colIdx)->type()->ToString() + " is not supported.");
        }
      } break;
      default:
        return arrow::Status::Invalid(
            "Column type " + schema_->field(colIdx)->type()->ToString() + " is not fixed width");
    }
  }
  return arrow::Status::OK();
}

arrow::Status VeloxHashShuffleWriterV2::splitBoolTypeV2(
    const uint8_t* srcAddr,
    const std::vector<std::vector<uint8_t*>>& dstAddrs) {
  // Boolean buffers are a single growing ResizableBuffer; write at absolute partitionBufferBase_.
  for (auto& pid : partitionUsed_) {
    auto* dstaddr = dstAddrs[pid][0];
    if (dstaddr == nullptr) {
      continue;
    }
    auto r = partition2RowOffsetBase_[pid];
    auto size = partition2RowOffsetBase_[pid + 1];
    auto dstOffset = partitionBufferBase_[pid];
    auto dstOffsetInByte = (8 - (dstOffset & 0x7)) & 0x7;
    auto dstIdxByte = dstOffsetInByte;
    auto dst = dstaddr[dstOffset >> 3];

    for (; r < size && dstIdxByte > 0; r++, dstIdxByte--) {
      auto srcOffset = rowOffset2RowId_[r];
      auto src = srcAddr[srcOffset >> 3];
      src = src >> (srcOffset & 7) | 0xfe;
#if defined(__x86_64__)
      src = __rolb(src, 8 - dstIdxByte);
#else
      src = rotateLeft(src, (8 - dstIdxByte));
#endif
      dst = dst & src;
    }
    dstaddr[dstOffset >> 3] = dst;
    if (r == size) {
      continue;
    }
    dstOffset += dstOffsetInByte;
    for (; r + 8 < size; r += 8) {
      uint8_t src = 0;
      auto srcOffset = rowOffset2RowId_[r];
      src = srcAddr[srcOffset >> 3];
      dst = src >> (srcOffset & 7) | 0xfe;

      srcOffset = rowOffset2RowId_[r + 1];
      src = srcAddr[srcOffset >> 3];
      dst &= src >> (srcOffset & 7) << 1 | 0xfd;

      srcOffset = rowOffset2RowId_[r + 2];
      src = srcAddr[srcOffset >> 3];
      dst &= src >> (srcOffset & 7) << 2 | 0xfb;

      srcOffset = rowOffset2RowId_[r + 3];
      src = srcAddr[srcOffset >> 3];
      dst &= src >> (srcOffset & 7) << 3 | 0xf7;

      srcOffset = rowOffset2RowId_[r + 4];
      src = srcAddr[srcOffset >> 3];
      dst &= src >> (srcOffset & 7) << 4 | 0xef;

      srcOffset = rowOffset2RowId_[r + 5];
      src = srcAddr[srcOffset >> 3];
      dst &= src >> (srcOffset & 7) << 5 | 0xdf;

      srcOffset = rowOffset2RowId_[r + 6];
      src = srcAddr[srcOffset >> 3];
      dst &= src >> (srcOffset & 7) << 6 | 0xbf;

      srcOffset = rowOffset2RowId_[r + 7];
      src = srcAddr[srcOffset >> 3];
      dst &= src >> (srcOffset & 7) << 7 | 0x7f;

      dstaddr[dstOffset >> 3] = dst;
      dstOffset += 8;
    }
    dst = 0xff;
    dstIdxByte = 0;
    for (; r < size; r++, dstIdxByte++) {
      auto srcOffset = rowOffset2RowId_[r];
      auto src = srcAddr[srcOffset >> 3];
      src = src >> (srcOffset & 7) | 0xfe;
#if defined(__x86_64__)
      src = __rolb(src, dstIdxByte);
#else
      src = rotateLeft(src, dstIdxByte);
#endif
      dst = dst & src;
    }
    dstaddr[dstOffset >> 3] = dst;
  }
  return arrow::Status::OK();
}

arrow::Status VeloxHashShuffleWriterV2::splitValidityBufferV2(const facebook::velox::RowVector& rv) {
  for (size_t col = 0; col < simpleColumnIndices_.size(); ++col) {
    auto colIdx = simpleColumnIndices_[col];
    auto& column = rv.childAt(colIdx);
    if (vectorHasNull(column)) {
      auto& dstAddrs = partitionValidityAddrs_[col];
      auto srcAddr = reinterpret_cast<const uint8_t*>(column->mutableRawNulls());
      // Validity is a single ResizableBuffer spanning all batches; reuse V1 bit-split with absolute base.
      RETURN_NOT_OK(splitBoolType(srcAddr, dstAddrs));
    }
  }
  return arrow::Status::OK();
}

arrow::Status VeloxHashShuffleWriterV2::splitBinaryTypeV2(
    uint32_t binaryIdx,
    const facebook::velox::FlatVector<facebook::velox::StringView>& src,
    std::vector<std::vector<BinaryBuf>>& dst) {
  auto rawValues = src.rawValues();

  for (auto& pid : partitionUsed_) {
    auto rowOffsetBase = partition2RowOffsetBase_[pid];
    auto numRows = partition2RowOffsetBase_[pid + 1] - rowOffsetBase;
    uint64_t lengthBufferLength = numRows * sizeof(BinaryArrayLengthBufferType);

    dst[pid].emplace_back(BinaryBuf(nullptr, nullptr, lengthBufferLength, 0));
    auto& binaryBuf = dst[pid].back();

    RETURN_NOT_OK(allocateBuffer(lengthBufferLength, &binaryBuf.lengthPtr));
    auto* dstLengthBase = reinterpret_cast<BinaryArrayLengthBufferType*>(binaryBuf.lengthPtr);

    for (auto i = 0; i < static_cast<int>(numRows); i++) {
      auto rowId = rowOffset2RowId_[rowOffsetBase + i];
      auto& stringView = rawValues[rowId];
      auto stringLen = src.isNullAt(rowId) ? 0 : stringView.size();
      dstLengthBase[i] = stringLen;
      binaryBuf.valueOffset += stringLen;
    }

    if (binaryBuf.valueOffset) {
      RETURN_NOT_OK(allocateBuffer(binaryBuf.valueOffset, &binaryBuf.valuePtr));
      variableMemoryUsage_[pid] += lengthBufferLength + binaryBuf.valueOffset;
    } else {
      variableMemoryUsage_[pid] += lengthBufferLength;
      continue;
    }

    uint64_t offset = 0;
    for (auto i = 0; i < static_cast<int>(numRows); i++) {
      auto rowId = rowOffset2RowId_[rowOffsetBase + i];
      auto& stringView = rawValues[rowId];
      auto stringLen = src.isNullAt(rowId) ? 0 : stringView.size();
      if (stringLen) {
        gluten::fastCopy(binaryBuf.valuePtr + offset, stringView.data(), stringLen);
        offset += stringLen;
      }
    }
    VELOX_CHECK_EQ(offset, binaryBuf.valueOffset);
  }
  return arrow::Status::OK();
}

arrow::Status VeloxHashShuffleWriterV2::splitBinaryArray(const facebook::velox::RowVector& rv) {
  for (auto col = fixedWidthColumnCount_; col < simpleColumnIndices_.size(); ++col) {
    auto binaryIdx = col - fixedWidthColumnCount_;
    auto& dstAddrs = partitionBinaryAddrsVector_[binaryIdx];
    auto colIdx = simpleColumnIndices_[col];
    auto column = rv.childAt(colIdx)->asFlatVector<facebook::velox::StringView>();
    RETURN_NOT_OK(splitBinaryTypeV2(binaryIdx, *column, dstAddrs));
  }
  return arrow::Status::OK();
}

arrow::Status VeloxHashShuffleWriterV2::splitRowVector(const facebook::velox::RowVector& rv) {
  SCOPED_TIMER(cpuWallTimingList_[CpuWallTimingSplitRV]);

  RETURN_NOT_OK(splitFixedWidthValueBufferV2(rv));
  RETURN_NOT_OK(splitValidityBufferV2(rv));
  RETURN_NOT_OK(splitBinaryArray(rv));
  RETURN_NOT_OK(splitComplexType(rv));

  for (auto& pid : partitionUsed_) {
    partitionBufferBase_[pid] += partition2RowCount_[pid];
    partitionBufferBaseInBatches_[pid] += partition2RowCount_[pid];
    if (partitionBufferBase_[pid] >= static_cast<uint32_t>(options_.bufferSize) ||
        (hasComplexType_ && arenas_[pid] && arenas_[pid]->size() > kMaxComplexTypePageSize)) {
      fullBatch_[pid] = true;
    }
  }
  return arrow::Status::OK();
}

arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>> VeloxHashShuffleWriterV2::assembleBuffersOneBatch(
    uint32_t partitionId) {
  SCOPED_TIMER(cpuWallTimingList_[CpuWallTimingCreateRbFromBuffer]);

  const auto& batchRows = batchNumRows_[partitionId];
  const uint32_t batchCount = batchRows.size();
  auto numRows = partitionBufferBase_[partitionId];
  VELOX_CHECK(batchCount == 1 && numRows == batchRows[0]);

  if (numRows == 0) {
    return std::vector<std::shared_ptr<arrow::Buffer>>{};
  }

  const auto lengthBytes = numRows * kSizeOfBinaryArrayLengthBuffer;
  uint64_t validityBytes = arrow::bit_util::BytesForBits(numRows);

  auto fixedWidthIdx = 0;
  auto binaryIdx = 0;
  auto numFields = schema_->num_fields();

  std::vector<std::shared_ptr<arrow::Buffer>> allBuffers;
  allBuffers.reserve(fixedWidthColumnCount_ * 2 + binaryColumnIndices_.size() * 3 + hasComplexType_);

  assembleBufferPool_.reset();

  for (int i = 0; i < numFields; ++i) {
    switch (arrowColumnTypes_[i]->id()) {
      case arrow::BinaryType::type_id:
      case arrow::StringType::type_id: {
        if (partitionValidityAddrs_[fixedWidthColumnCount_ + binaryIdx][partitionId] != nullptr) {
          allBuffers.push_back(arrow::SliceBuffer(
              partitionValidityBuffers_[fixedWidthColumnCount_ + binaryIdx][partitionId], 0, validityBytes));
        } else {
          allBuffers.push_back(nullptr);
        }

        const auto& binaryBufs = partitionBinaryAddrsVector_[binaryIdx][partitionId];
        if (binaryBufs.size() == 1) {
          allBuffers.push_back(std::make_shared<arrow::Buffer>(binaryBufs[0].lengthPtr, lengthBytes));
          if (binaryBufs[0].valueOffset > 0) {
            allBuffers.push_back(std::make_shared<arrow::Buffer>(binaryBufs[0].valuePtr, binaryBufs[0].valueOffset));
          } else {
            allBuffers.push_back(zeroLengthNullBuffer());
          }
        } else {
          uint8_t* lengthBuffer = nullptr;
          uint64_t valueLength = 0;
          uint64_t valueOffset = 0;
          RETURN_NOT_OK(allocateAssembledBuffer(lengthBytes, &lengthBuffer));
          for (auto n = 0; n < static_cast<int>(binaryBufs.size()); ++n) {
            gluten::fastCopy(lengthBuffer + valueOffset, binaryBufs[n].lengthPtr, binaryBufs[n].valueCapacity);
            valueOffset += binaryBufs[n].valueCapacity;
            valueLength += binaryBufs[n].valueOffset;
          }
          allBuffers.push_back(std::make_shared<arrow::Buffer>(lengthBuffer, lengthBytes));

          if (valueLength > 0) {
            uint8_t* valueBuffer = nullptr;
            valueOffset = 0;
            RETURN_NOT_OK(allocateAssembledBuffer(valueLength, &valueBuffer));
            for (uint32_t n = 0; n < binaryBufs.size(); ++n) {
              gluten::fastCopy(valueBuffer + valueOffset, binaryBufs[n].valuePtr, binaryBufs[n].valueOffset);
              valueOffset += binaryBufs[n].valueOffset;
            }
            allBuffers.push_back(std::make_shared<arrow::Buffer>(valueBuffer, valueLength));
          } else {
            allBuffers.push_back(zeroLengthNullBuffer());
          }
        }

        partitionBinaryAddrsVector_[binaryIdx][partitionId].clear();
        partitionValidityAddrs_[fixedWidthColumnCount_ + binaryIdx][partitionId] = nullptr;
        binaryIdx++;
        break;
      }
      case arrow::StructType::type_id:
      case arrow::MapType::type_id:
      case arrow::ListType::type_id:
      case arrow::NullType::type_id:
        break;
      default: {
        if (partitionValidityAddrs_[fixedWidthIdx][partitionId] != nullptr) {
          allBuffers.push_back(
              arrow::SliceBuffer(partitionValidityBuffers_[fixedWidthIdx][partitionId], 0, validityBytes));
        } else {
          allBuffers.push_back(nullptr);
        }

        if (arrowColumnTypes_[i]->id() == arrow::BooleanType::type_id) {
          allBuffers.push_back(std::make_shared<arrow::Buffer>(
              partitionFixedWidthValueAddrsVector_[fixedWidthIdx][partitionId][0], validityBytes));
        } else {
          auto fixedLen = fixedColValueSize_[fixedWidthIdx];
          const auto& fixedValues = partitionFixedWidthValueAddrsVector_[fixedWidthIdx][partitionId];
          allBuffers.push_back(std::make_shared<arrow::Buffer>(fixedValues[0], fixedLen * numRows));
          partitionFixedWidthValueAddrsVector_[fixedWidthIdx][partitionId].clear();
        }

        partitionValidityAddrs_[fixedWidthIdx][partitionId] = nullptr;
        fixedWidthIdx++;
        break;
      }
    }
  }

  if (hasComplexType_ && complexTypeData_[partitionId] != nullptr) {
    auto flushBuffer = complexTypeFlushBuffer_[partitionId];
    auto serializedSize = complexTypeData_[partitionId]->maxSerializedSize();
    if (flushBuffer == nullptr) {
      ARROW_ASSIGN_OR_RAISE(flushBuffer, arrow::AllocateResizableBuffer(serializedSize, partitionBufferPool_.get()));
      complexTypeFlushBuffer_[partitionId] = flushBuffer;
    } else if (serializedSize > flushBuffer->capacity()) {
      RETURN_NOT_OK(flushBuffer->Reserve(serializedSize));
    }
    auto valueBuffer = arrow::SliceMutableBuffer(flushBuffer, 0, serializedSize);
    auto output = std::make_shared<arrow::io::FixedSizeBufferWriter>(valueBuffer);
    facebook::velox::serializer::presto::PrestoOutputStreamListener listener;
    ArrowFixedSizeBufferOutputStream out(output, &listener);
    complexTypeData_[partitionId]->flush(&out);
    allBuffers.emplace_back(valueBuffer);
    complexTypeData_[partitionId] = nullptr;
    arenas_[partitionId] = nullptr;
  }

  partitionBufferBase_[partitionId] = 0;
  batchNumRows_[partitionId].clear();
  return allBuffers;
}

arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>> VeloxHashShuffleWriterV2::assembleBuffersGeneral(
    uint32_t partitionId) {
  const auto& batchRows = batchNumRows_[partitionId];
  const uint32_t batchCount = batchRows.size();
  if (batchCount == 1) {
    return assembleBuffersOneBatch(partitionId);
  }

  SCOPED_TIMER(cpuWallTimingList_[CpuWallTimingCreateRbFromBuffer]);
  auto numRows = partitionBufferBase_[partitionId];
  if (numRows == 0) {
    return std::vector<std::shared_ptr<arrow::Buffer>>{};
  }

  const auto lengthBytes = numRows * kSizeOfBinaryArrayLengthBuffer;
  uint64_t validityBytes = arrow::bit_util::BytesForBits(numRows);

  auto fixedWidthIdx = 0;
  auto binaryIdx = 0;
  auto numFields = schema_->num_fields();

  std::vector<std::shared_ptr<arrow::Buffer>> allBuffers;
  allBuffers.reserve(fixedWidthColumnCount_ * 2 + binaryColumnIndices_.size() * 3 + hasComplexType_);

  assembleBufferPool_.reset();
  for (int i = 0; i < numFields; ++i) {
    switch (arrowColumnTypes_[i]->id()) {
      case arrow::BinaryType::type_id:
      case arrow::StringType::type_id: {
        if (partitionValidityAddrs_[fixedWidthColumnCount_ + binaryIdx][partitionId] != nullptr) {
          allBuffers.push_back(arrow::SliceBuffer(
              partitionValidityBuffers_[fixedWidthColumnCount_ + binaryIdx][partitionId], 0, validityBytes));
        } else {
          allBuffers.push_back(nullptr);
        }

        const auto& binaryBufs = partitionBinaryAddrsVector_[binaryIdx][partitionId];
        uint8_t* lengthBuffer = nullptr;
        uint64_t valueLength = 0;
        uint64_t valueOffset = 0;
        RETURN_NOT_OK(allocateAssembledBuffer(lengthBytes, &lengthBuffer));

        for (auto n = 0; n < static_cast<int>(binaryBufs.size()); ++n) {
          gluten::fastCopy(lengthBuffer + valueOffset, binaryBufs[n].lengthPtr, binaryBufs[n].valueCapacity);
          valueOffset += binaryBufs[n].valueCapacity;
          valueLength += binaryBufs[n].valueOffset;
        }
        allBuffers.push_back(std::make_shared<arrow::Buffer>(lengthBuffer, lengthBytes));

        if (valueLength > 0) {
          uint8_t* valueBuffer = nullptr;
          valueOffset = 0;
          RETURN_NOT_OK(allocateAssembledBuffer(valueLength, &valueBuffer));
          for (uint32_t n = 0; n < binaryBufs.size(); ++n) {
            gluten::fastCopy(valueBuffer + valueOffset, binaryBufs[n].valuePtr, binaryBufs[n].valueOffset);
            valueOffset += binaryBufs[n].valueOffset;
          }
          allBuffers.push_back(std::make_shared<arrow::Buffer>(valueBuffer, valueLength));
        } else {
          allBuffers.push_back(zeroLengthNullBuffer());
        }

        partitionBinaryAddrsVector_[binaryIdx][partitionId].clear();
        partitionValidityAddrs_[fixedWidthColumnCount_ + binaryIdx][partitionId] = nullptr;
        binaryIdx++;
        break;
      }
      case arrow::StructType::type_id:
      case arrow::MapType::type_id:
      case arrow::ListType::type_id:
      case arrow::NullType::type_id:
        break;
      default: {
        if (partitionValidityAddrs_[fixedWidthIdx][partitionId] != nullptr) {
          allBuffers.push_back(
              arrow::SliceBuffer(partitionValidityBuffers_[fixedWidthIdx][partitionId], 0, validityBytes));
        } else {
          allBuffers.push_back(nullptr);
        }

        if (arrowColumnTypes_[i]->id() == arrow::BooleanType::type_id) {
          allBuffers.push_back(std::make_shared<arrow::Buffer>(
              partitionFixedWidthValueAddrsVector_[fixedWidthIdx][partitionId][0], validityBytes));
        } else {
          uint64_t fixedLen = fixedColValueSize_[fixedWidthIdx];
          const auto& fixedValues = partitionFixedWidthValueAddrsVector_[fixedWidthIdx][partitionId];
          uint8_t* valueBuffer = nullptr;
          uint64_t valueOffset = 0;
          RETURN_NOT_OK(allocateAssembledBuffer(fixedLen * numRows, &valueBuffer));
          for (uint32_t n = 0; n < batchCount; ++n) {
            uint64_t valueLen = batchRows[n] * fixedLen;
            VELOX_CHECK_NE(valueLen, 0);
            gluten::fastCopy(valueBuffer + valueOffset, fixedValues[n], valueLen);
            valueOffset += valueLen;
          }
          VELOX_CHECK_EQ(valueOffset, fixedLen * numRows);
          allBuffers.push_back(std::make_shared<arrow::Buffer>(valueBuffer, valueOffset));
          partitionFixedWidthValueAddrsVector_[fixedWidthIdx][partitionId].clear();
        }

        partitionValidityAddrs_[fixedWidthIdx][partitionId] = nullptr;
        fixedWidthIdx++;
        break;
      }
    }
  }

  if (hasComplexType_ && complexTypeData_[partitionId] != nullptr) {
    auto flushBuffer = complexTypeFlushBuffer_[partitionId];
    auto serializedSize = complexTypeData_[partitionId]->maxSerializedSize();
    if (flushBuffer == nullptr) {
      ARROW_ASSIGN_OR_RAISE(flushBuffer, arrow::AllocateResizableBuffer(serializedSize, partitionBufferPool_.get()));
      complexTypeFlushBuffer_[partitionId] = flushBuffer;
    } else if (serializedSize > flushBuffer->capacity()) {
      RETURN_NOT_OK(flushBuffer->Reserve(serializedSize));
    }
    auto valueBuffer = arrow::SliceMutableBuffer(flushBuffer, 0, serializedSize);
    auto output = std::make_shared<arrow::io::FixedSizeBufferWriter>(valueBuffer);
    facebook::velox::serializer::presto::PrestoOutputStreamListener listener;
    ArrowFixedSizeBufferOutputStream out(output, &listener);
    complexTypeData_[partitionId]->flush(&out);
    allBuffers.emplace_back(valueBuffer);
    complexTypeData_[partitionId] = nullptr;
    arenas_[partitionId] = nullptr;
  }

  partitionBufferBase_[partitionId] = 0;
  batchNumRows_[partitionId].clear();
  return allBuffers;
}

arrow::Status VeloxHashShuffleWriterV2::evictPartitionBuffers(uint32_t partitionId, Evict::type evictType) {
  auto numRows = partitionBufferBase_[partitionId];
  if (numRows > 0) {
    ARROW_ASSIGN_OR_RAISE(auto buffers, assembleBuffersGeneral(partitionId));
    if (!buffers.empty()) {
      auto payload = std::make_unique<InMemoryPayload>(numRows, &isValidityBuffer_, std::move(buffers));
      RETURN_NOT_OK(partitionWriter_->hashEvict(partitionId, std::move(payload), evictType, false, hasComplexType_));
      RETURN_NOT_OK(resetValidityBuffer(partitionId));
    }
  }
  return arrow::Status::OK();
}

arrow::Status VeloxHashShuffleWriterV2::evictPartitionBuffers(uint32_t partitionId, bool /*reuseBuffers*/) {
  return evictPartitionBuffers(partitionId, Evict::kCacheNoMerge);
}

arrow::Status VeloxHashShuffleWriterV2::resetValidityBuffer(uint32_t partitionId) {
  std::for_each(partitionValidityBuffers_.begin(), partitionValidityBuffers_.end(), [partitionId](auto& bufs) {
    if (bufs[partitionId] != nullptr && bufs[partitionId]->size() > 0) {
      memset(bufs[partitionId]->mutable_data(), 0xff, bufs[partitionId]->capacity());
    }
  });
  return arrow::Status::OK();
}

arrow::Status VeloxHashShuffleWriterV2::evictFullPartitions() {
  SCOPED_TIMER(cpuWallTimingList_[CpuWallTimingEvictPartition]);
  for (const auto& [pid, dummy] : fullBatch_) {
    RETURN_NOT_OK(evictPartitionBuffers(pid, Evict::kCacheNoMerge));
    variableMemoryUsage_[pid] = 0;
  }
  if (!variableMemoryUsage_.empty()) {
    maxVariableMemoryUsage_ = *std::max_element(variableMemoryUsage_.begin(), variableMemoryUsage_.end());
  }
  fullBatch_.clear();
  return arrow::Status::OK();
}

arrow::Result<bool> VeloxHashShuffleWriterV2::sequentialEvictAllPartitions() {
  uint32_t pid = 0;
  bool evicted = false;
  for (; pid < numPartitions_; ++pid) {
    if (partitionBufferBase_[pid]) {
      break;
    }
  }
  if (pid < numPartitions_) {
    evicted = true;
    RETURN_NOT_OK(partitionWriter_->startClearPayLoadCacheSequential());
    for (pid = 0; pid < numPartitions_; ++pid) {
      RETURN_NOT_OK(evictPartitionBuffers(pid, Evict::kCacheNoMerge));
      RETURN_NOT_OK(partitionWriter_->clearSpecificPayLoadCache(pid));
    }
    RETURN_NOT_OK(partitionWriter_->stopClearPayLoadCacheSequential());
  }
  return evicted;
}

arrow::Status VeloxHashShuffleWriterV2::tryEvict(int64_t memLimit) {
  EvictGuard evictGuard{evictState_};
  RETURN_NOT_OK(evictFullPartitions());
  constexpr int64_t kMinReserveBufferSize = 8 * 1024 * 1024;
  if (requestSpill_ ||
      static_cast<int64_t>(std::max(maxVariableMemoryUsage_, static_cast<uint64_t>(kMinReserveBufferSize))) >=
          memLimit / 4) {
    bufferPool_.shrink();
    ARROW_ASSIGN_OR_RAISE(auto ret, sequentialEvictAllPartitions());
    if (!ret && partitionWriter_->canSpill()) {
      RETURN_NOT_OK(partitionWriter_->evictPayLoadCache());
    }
    bufferPool_.clear();
    std::fill(variableMemoryUsage_.begin(), variableMemoryUsage_.end(), 0);
    maxVariableMemoryUsage_ = 0;
  }
  requestSpill_ = false;
  fullBatch_.clear();
  return arrow::Status::OK();
}

arrow::Status VeloxHashShuffleWriterV2::stop() {
  if (!batches_.empty()) {
    if (batches_.size() == 1) {
      RETURN_NOT_OK(splitSingleBatch(std::numeric_limits<int64_t>::max()));
    } else {
      RETURN_NOT_OK(splitBatches(std::numeric_limits<int64_t>::max()));
    }
  }
  setSplitState(SplitState::kStopEvict);
  bufferPool_.shrink();
  ARROW_ASSIGN_OR_RAISE(auto ret, sequentialEvictAllPartitions());
  (void)ret;
  {
    SCOPED_TIMER(cpuWallTimingList_[CpuWallTimingStop]);
    setSplitState(SplitState::kStop);
    RETURN_NOT_OK(partitionWriter_->stop(&metrics_));
    bufferPool_.clear();
    assembleBufferPool_.clear();
  }
  stat();
  return arrow::Status::OK();
}

arrow::Status VeloxHashShuffleWriterV2::reclaimFixedSize(int64_t size, int64_t* actual) {
  if (evictState_ == EvictState::kUnevictable || splitState_ == SplitState::kStop) {
    *actual = 0;
    return arrow::Status::OK();
  }
  EvictGuard evictGuard{evictState_};

  int64_t reclaimed = 0;
  if (splitState_ == SplitState::kInit) {
    RETURN_NOT_OK(partitionWriter_->reclaimFixedSizeNoMerge(size - reclaimed, &reclaimed));
    if (reclaimed < size) {
      ARROW_ASSIGN_OR_RAISE(auto evicted, evictPartitionBuffersMinSize(size - reclaimed));
      reclaimed += evicted;
    }
  } else {
    requestSpill_ = true;
  }
  *actual = reclaimed;
  return arrow::Status::OK();
}

arrow::Result<int64_t> VeloxHashShuffleWriterV2::evictPartitionBuffersMinSize(int64_t size) {
  onBeforeEvictPartitionBuffersMinSize();
  int64_t before = partitionBufferPool_->bytes_allocated() + static_cast<int64_t>(bufferPool_.reservedBytes());
  ARROW_ASSIGN_OR_RAISE(auto ret, sequentialEvictAllPartitions());
  (void)ret;
  onAfterEvictPartitionBuffersMinSize();
  int64_t after = partitionBufferPool_->bytes_allocated() + static_cast<int64_t>(bufferPool_.reservedBytes());
  return std::max<int64_t>(0, before - after);
}

} // namespace gluten
