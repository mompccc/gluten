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

#include <fstream>
#include <unistd.h>

#include <glog/logging.h>

namespace gluten {

namespace {

constexpr const char* kV2MarkerFile = "/tmp/gluten_hash_shuffle_v2.log";

void logV2WriterActivated(int64_t taskAttemptId, uint32_t numPartitions, int32_t bufferSize) {
  LOG(INFO) << "[VeloxHashShuffleWriterV2] activated"
            << " taskAttemptId=" << taskAttemptId << " pid=" << getpid()
            << " numPartitions=" << numPartitions << " bufferSize=" << bufferSize
            << " evict=kCacheNoMerge";
  std::ofstream marker(kV2MarkerFile, std::ios::app);
  if (marker) {
    marker << "taskAttemptId=" << taskAttemptId << " pid=" << getpid()
           << " partitions=" << numPartitions << " bufferSize=" << bufferSize << '\n';
  }
}

} // namespace

VeloxHashShuffleWriterV2::~VeloxHashShuffleWriterV2() {
  bufferPool_.clear();
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
      bumpMemoryPool_(&bufferPool_) {}

arrow::Result<std::shared_ptr<arrow::ResizableBuffer>> VeloxHashShuffleWriterV2::allocatePartitionResizableBuffer(
    int64_t size) {
  return arrow::AllocateResizableBuffer(size, &bumpMemoryPool_);
}

Evict::type VeloxHashShuffleWriterV2::hashEvictType() const {
  return Evict::kCacheNoMerge;
}

void VeloxHashShuffleWriterV2::onBeforeStopEvict() {
  bufferPool_.shrink();
}

void VeloxHashShuffleWriterV2::onAfterStop() {
  bufferPool_.clear();
}

void VeloxHashShuffleWriterV2::onBeforeEvictPartitionBuffersMinSize() {
  bufferPool_.shrink();
}

void VeloxHashShuffleWriterV2::onAfterEvictPartitionBuffersMinSize() {
  bufferPool_.clear();
}

} // namespace gluten
