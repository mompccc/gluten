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

#include "VeloxAdaptiveHashShuffleWriter.h"

#include <glog/logging.h>

#include "VeloxHashShuffleWriter.h"
#include "VeloxHashShuffleWriterSelector.h"
#include "VeloxHashShuffleWriterV2.h"
#include "memory/VeloxColumnarBatch.h"

namespace gluten {

arrow::Result<std::shared_ptr<VeloxShuffleWriter>> VeloxAdaptiveHashShuffleWriter::create(
    uint32_t numPartitions,
    std::unique_ptr<PartitionWriter> partitionWriter,
    ShuffleWriterOptions options,
    std::shared_ptr<facebook::velox::memory::MemoryPool> veloxPool,
    arrow::MemoryPool* arrowPool) {
  return std::shared_ptr<VeloxAdaptiveHashShuffleWriter>(new VeloxAdaptiveHashShuffleWriter(
      numPartitions, std::move(partitionWriter), std::move(options), std::move(veloxPool), arrowPool));
}

VeloxAdaptiveHashShuffleWriter::VeloxAdaptiveHashShuffleWriter(
    uint32_t numPartitions,
    std::unique_ptr<PartitionWriter> partitionWriter,
    ShuffleWriterOptions options,
    std::shared_ptr<facebook::velox::memory::MemoryPool> veloxPool,
    arrow::MemoryPool* arrowPool)
    : VeloxShuffleWriter(numPartitions, nullptr, std::move(options), std::move(veloxPool), arrowPool),
      pendingPartitionWriter_(std::move(partitionWriter)),
      arrowPool_(arrowPool) {}

arrow::Status VeloxAdaptiveHashShuffleWriter::ensureDelegate(
    const std::shared_ptr<ColumnarBatch>& cb,
    int64_t memLimit) {
  if (delegate_) {
    return arrow::Status::OK();
  }
  auto veloxBatch = VeloxColumnarBatch::from(veloxPool_.get(), cb);
  auto rv = veloxBatch->getFlattenedRowVector();
  const auto numRows = rv->size();
  const auto flatSize = static_cast<int64_t>(rv->estimateFlatSize());
  const auto numColumnsExcludePid =
      rv->childrenSize() > 0 ? static_cast<uint32_t>(rv->childrenSize() - 1) : 0;
  const auto preAlloc = calculatePreallocRowCount(
      numRows, flatSize, memLimit, static_cast<uint32_t>(numPartitions_), options_.bufferSize);
  auto type = decideVeloxHashShuffleWriterType(options_, numColumnsExcludePid, numPartitions_, preAlloc);
  LOG(INFO) << "[VeloxAdaptiveHashShuffleWriter] selected type="
            << (type == ShuffleWriterType::kHashShuffleV2 ? "hash_v2" : "hash")
            << " preAllocRowCount=" << preAlloc << " numPartitions=" << numPartitions_
            << " numRows=" << numRows << " flatSize=" << flatSize;

  // Call concrete creators directly to avoid recursing into Adaptive via
  // VeloxShuffleWriter::create(kHashShuffle).
  if (type == ShuffleWriterType::kHashShuffleV2) {
    ARROW_ASSIGN_OR_RAISE(
        delegate_,
        VeloxHashShuffleWriterV2::create(
            numPartitions_,
            std::move(pendingPartitionWriter_),
            options_,
            veloxPool_,
            arrowPool_));
  } else {
    ARROW_ASSIGN_OR_RAISE(
        delegate_,
        VeloxHashShuffleWriter::create(
            numPartitions_,
            std::move(pendingPartitionWriter_),
            options_,
            veloxPool_,
            arrowPool_));
  }
  return arrow::Status::OK();
}

arrow::Status VeloxAdaptiveHashShuffleWriter::write(std::shared_ptr<ColumnarBatch> cb, int64_t memLimit) {
  RETURN_NOT_OK(ensureDelegate(cb, memLimit));
  return delegate_->write(std::move(cb), memLimit);
}

arrow::Status VeloxAdaptiveHashShuffleWriter::stop() {
  if (!delegate_) {
    return arrow::Status::OK();
  }
  RETURN_NOT_OK(delegate_->stop());
  metrics_.totalBytesWritten = delegate_->totalBytesWritten();
  metrics_.totalBytesEvicted = delegate_->totalBytesEvicted();
  metrics_.totalBytesToEvict = delegate_->totalBytesToEvict();
  metrics_.totalWriteTime = delegate_->totalWriteTime();
  metrics_.totalEvictTime = delegate_->totalEvictTime();
  metrics_.totalCompressTime = delegate_->totalCompressTime();
  metrics_.partitionLengths = delegate_->partitionLengths();
  metrics_.rawPartitionLengths = delegate_->rawPartitionLengths();
  return arrow::Status::OK();
}

arrow::Status VeloxAdaptiveHashShuffleWriter::reclaimFixedSize(int64_t size, int64_t* actual) {
  if (!delegate_) {
    *actual = 0;
    return arrow::Status::OK();
  }
  return delegate_->reclaimFixedSize(size, actual);
}

int64_t VeloxAdaptiveHashShuffleWriter::peakBytesAllocated() const {
  if (!delegate_) {
    return 0;
  }
  return delegate_->peakBytesAllocated();
}

const uint64_t VeloxAdaptiveHashShuffleWriter::cachedPayloadSize() const {
  if (!delegate_) {
    return 0;
  }
  return delegate_->cachedPayloadSize();
}

} // namespace gluten
