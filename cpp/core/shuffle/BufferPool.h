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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <arrow/buffer.h>
#include <arrow/memory_pool.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <glog/logging.h>

#include "shuffle/ShuffleMemoryPool.h"

namespace gluten {

/// Bump allocator over fixed-size chunks backed by ShuffleMemoryPool.
/// Ported from bolt BoltShuffleWriterV2.h BufferPool; tuned for Kunpeng 920
/// (64B alignment = cache line size; 8MB chunk fits L3 sequential access).
class BufferPool {
 public:
  static constexpr int32_t kDefaultBufferSize = 8 << 20;
  static constexpr int32_t kDefaultBufferAlignment = 64;

  BufferPool() = default;
  explicit BufferPool(ShuffleMemoryPool* pool) : pool_(pool) {}

  arrow::Status allocateFixed(uint64_t size, uint8_t** out) {
    if (bytesInBuffer_ >= size) {
      *out = startOfBuffer_;
      startOfBuffer_ += size;
      bytesInBuffer_ -= size;
    } else if (size > kDefaultBufferSize) {
      return allocateLargeBuffer(size, out);
    } else {
      return allocateFromNextBuffer(size, out);
    }
    return arrow::Status::OK();
  }

  arrow::Status allocateFixedAligned(uint64_t size, uint8_t** out, int32_t alignment) {
    DCHECK_EQ(__builtin_popcount(static_cast<unsigned>(alignment)), 1)
        << "Alignment must be a power of 2";
    DCHECK_LE(alignment, kDefaultBufferAlignment);

    if (bytesInBuffer_ >= size) {
      auto paddingBytes = alignmentPadding(startOfBuffer_, alignment);
      auto alignedBytes = size + paddingBytes;
      if (bytesInBuffer_ >= alignedBytes) {
        *out = startOfBuffer_ + paddingBytes;
        startOfBuffer_ += alignedBytes;
        bytesInBuffer_ -= alignedBytes;
      } else {
        return allocateFromNextBuffer(size, out, alignment);
      }
    } else if (size > kDefaultBufferSize) {
      return allocateLargeBuffer(size, out);
    } else {
      return allocateFromNextBuffer(size, out, alignment);
    }
    return arrow::Status::OK();
  }

  /// Release chunks after currentBufferIndex_ (memory pressure path).
  /// Typically called after reset() rewinds the bump pointer to chunk[0].
  uint64_t shrink() {
    uint64_t shrunkBytes = 0;
    if (!buffers_.empty() && currentBufferIndex_ < static_cast<int32_t>(buffers_.size()) - 1) {
      for (auto index = currentBufferIndex_ + 1; index < static_cast<int32_t>(buffers_.size()); ++index) {
        pool_->Free(buffers_[index], kDefaultBufferSize, kDefaultBufferAlignment);
        buffers_[index] = nullptr;
      }
      shrunkBytes = kDefaultBufferSize * (buffers_.size() - currentBufferIndex_ - 1);
      reservedBytes_ -= shrunkBytes;
      buffers_.erase(buffers_.begin() + currentBufferIndex_ + 1, buffers_.end());
    }
    return shrunkBytes;
  }

  /// Free all chunks and large buffers.
  void clear() {
    for (auto* buffer : buffers_) {
      pool_->Free(buffer, kDefaultBufferSize, kDefaultBufferAlignment);
    }
    reservedBytes_ -= kDefaultBufferSize * buffers_.size();
    buffers_.clear();
    releaseLargeBuffers();
    DCHECK_EQ(reservedBytes_, 0);

    currentBufferIndex_ = -1;
    startOfBuffer_ = nullptr;
    bytesInBuffer_ = 0;
  }

  /// Rewind bump pointer to reuse retained chunks without freeing.
  void reset() {
    releaseLargeBuffers();
    if (!buffers_.empty()) {
      currentBufferIndex_ = 0;
      bytesInBuffer_ = kDefaultBufferSize;
      startOfBuffer_ = buffers_[0];
    }
  }

  uint64_t reservedBytes() const {
    return reservedBytes_;
  }

  static int32_t alignmentPadding(void* addr, int32_t alignment) {
    auto padding = reinterpret_cast<uintptr_t>(addr) % alignment;
    return padding == 0 ? 0 : alignment - padding;
  }

 private:
  struct InternalBuffer {
    uint8_t* data;
    uint64_t size;
  };

  void releaseLargeBuffers() {
    for (const auto& buffer : largeBuffers_) {
      reservedBytes_ -= buffer.size;
      pool_->Free(buffer.data, buffer.size, kDefaultBufferAlignment);
    }
    largeBuffers_.clear();
  }

  arrow::Status allocateLargeBuffer(uint64_t size, uint8_t** out) {
    ARROW_RETURN_NOT_OK(pool_->Allocate(size, kDefaultBufferAlignment, out));
    largeBuffers_.push_back(InternalBuffer{*out, size});
    reservedBytes_ += size;
    return arrow::Status::OK();
  }

  arrow::Status allocateFromNextBuffer(uint64_t size, uint8_t** out, int32_t alignment = kDefaultBufferAlignment) {
    if (currentBufferIndex_ == static_cast<int32_t>(buffers_.size()) - 1) {
      ARROW_RETURN_NOT_OK(pool_->Allocate(kDefaultBufferSize, kDefaultBufferAlignment, out));
      buffers_.push_back(*out);
      reservedBytes_ += kDefaultBufferSize;
    }
    ++currentBufferIndex_;
    auto* chunk = buffers_[currentBufferIndex_];
    if (alignment > 1) {
      auto padding = alignmentPadding(chunk, alignment);
      *out = chunk + padding;
      startOfBuffer_ = *out + size;
      bytesInBuffer_ = kDefaultBufferSize - padding - size;
    } else {
      *out = chunk;
      startOfBuffer_ = chunk + size;
      bytesInBuffer_ = kDefaultBufferSize - size;
    }
    return arrow::Status::OK();
  }

  std::vector<uint8_t*> buffers_;
  std::vector<InternalBuffer> largeBuffers_;
  int32_t currentBufferIndex_ = -1;
  uint64_t bytesInBuffer_{0};
  uint64_t reservedBytes_{0};
  uint8_t* startOfBuffer_{nullptr};
  ShuffleMemoryPool* pool_{nullptr};
};

/// Arrow MemoryPool adapter: Allocate/Reallocate delegate to BufferPool bump;
/// Free is no-op (chunk lifecycle managed by BufferPool::shrink/clear).
class BumpMemoryPool final : public arrow::MemoryPool {
 public:
  explicit BumpMemoryPool(BufferPool* pool) : pool_(pool) {}

  arrow::Status Allocate(int64_t size, int64_t alignment, uint8_t** out) override {
    if (alignment > 1 && alignment <= BufferPool::kDefaultBufferAlignment) {
      return pool_->allocateFixedAligned(size, out, static_cast<int32_t>(alignment));
    }
    return pool_->allocateFixed(size, out);
  }

  arrow::Status Reallocate(int64_t oldSize, int64_t newSize, int64_t alignment, uint8_t** ptr) override {
    if (newSize <= oldSize) {
      return arrow::Status::OK();
    }
    uint8_t* newPtr = nullptr;
    if (alignment > 1 && alignment <= BufferPool::kDefaultBufferAlignment) {
      ARROW_RETURN_NOT_OK(pool_->allocateFixedAligned(newSize, &newPtr, static_cast<int32_t>(alignment)));
    } else {
      ARROW_RETURN_NOT_OK(pool_->allocateFixed(newSize, &newPtr));
    }
    if (oldSize > 0 && *ptr != nullptr) {
      std::memcpy(newPtr, *ptr, oldSize);
    }
    *ptr = newPtr;
    return arrow::Status::OK();
  }

  void Free(uint8_t* /*buffer*/, int64_t /*size*/, int64_t /*alignment*/) override {}

  int64_t bytes_allocated() const override {
    return static_cast<int64_t>(pool_->reservedBytes());
  }

  int64_t max_memory() const override {
    return static_cast<int64_t>(pool_->reservedBytes());
  }

  std::string backend_name() const override {
    return "BumpMemoryPool";
  }

  int64_t total_bytes_allocated() const override {
    return static_cast<int64_t>(pool_->reservedBytes());
  }

  int64_t num_allocations() const override {
    return 0;
  }

 private:
  BufferPool* pool_;
};

/// Tag subclass: uses Arrow ResizableBuffer + BumpMemoryPool for correct Resize.
class BumpResizableBuffer : public arrow::ResizableBuffer {
 public:
  static arrow::Result<std::shared_ptr<BumpResizableBuffer>> Allocate(BumpMemoryPool* pool, int64_t size) {
    return std::shared_ptr<BumpResizableBuffer>(new BumpResizableBuffer(size, pool));
  }

 private:
  BumpResizableBuffer(int64_t size, BumpMemoryPool* pool) : arrow::ResizableBuffer(size, pool) {}
};

} // namespace gluten
