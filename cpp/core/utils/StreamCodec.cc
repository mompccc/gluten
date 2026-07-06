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

#include <arrow/status.h>
#include <arrow/util/logging.h>
#include <lz4frame.h>
#include <zstd.h>

#include <cstring>

namespace gluten {
namespace {

class ZstdStreamCompressor final : public StreamCompressor {
 public:
  explicit ZstdStreamCompressor(int compressionLevel) : compressionLevel_(compressionLevel) {
    cstream_ = ZSTD_createCStream();
    ARROW_CHECK(cstream_ != nullptr);
    reset();
  }

  ~ZstdStreamCompressor() override {
    if (cstream_) {
      ZSTD_freeCStream(cstream_);
    }
  }

  StreamCompressResult compress(
      const uint8_t* input,
      int64_t inputLen,
      uint8_t* output,
      int64_t outputLen) override {
    ZSTD_inBuffer inBuf{input, static_cast<size_t>(inputLen), 0};
    ZSTD_outBuffer outBuf{output, static_cast<size_t>(outputLen), 0};
    size_t ret = ZSTD_compressStream(cstream_, &outBuf, &inBuf);
    if (ZSTD_isError(ret)) {
      ARROW_LOG(WARNING) << "ZSTD_compressStream failed: " << ZSTD_getErrorName(ret);
    }
    return {static_cast<int64_t>(inBuf.pos), static_cast<int64_t>(outBuf.pos)};
  }

  StreamEndResult end(uint8_t* output, int64_t outputLen) override {
    ZSTD_outBuffer outBuf{output, static_cast<size_t>(outputLen), 0};
    size_t ret = ZSTD_endStream(cstream_, &outBuf);
    if (ZSTD_isError(ret)) {
      ARROW_LOG(WARNING) << "ZSTD_endStream failed: " << ZSTD_getErrorName(ret);
    }
    // ret == 0 means end complete; otherwise more output remains.
    return {static_cast<int64_t>(outBuf.pos), ret == 0};
  }

  void reset() override {
    size_t ret = ZSTD_CCtx_reset(cstream_, ZSTD_reset_session_only);
    if (ZSTD_isError(ret)) {
      ARROW_LOG(WARNING) << "ZSTD_CCtx_reset failed: " << ZSTD_getErrorName(ret);
      return;
    }
    ret = ZSTD_CCtx_setParameter(cstream_, ZSTD_c_compressionLevel, compressionLevel_);
    if (ZSTD_isError(ret)) {
      ARROW_LOG(WARNING) << "ZSTD_c_compressionLevel failed: " << ZSTD_getErrorName(ret);
    }
  }

  int64_t recommendedOutputSize(int64_t inputSize) const override {
    return static_cast<int64_t>(ZSTD_compressBound(static_cast<size_t>(inputSize)));
  }

 private:
  int compressionLevel_;
  ZSTD_CStream* cstream_;
};

// LZ4 Frame streaming compressor, based on LZ4F_compressBegin/Update/End.
// Modeled on bolt's Lz4FrameStreamCompressor. Single-threaded (LZ4 has no
// nbWorkers equivalent — see analysis; parallelism is not applicable here).
// Output is a standard LZ4 frame, byte-compatible with Arrow one-shot
// LZ4_FRAME Decompress.
class Lz4FrameStreamCompressor final : public StreamCompressor {
 public:
  explicit Lz4FrameStreamCompressor(int compressionLevel) : compressionLevel_(compressionLevel) {
    memset(&prefs_, 0, sizeof(prefs_));
    prefs_.compressionLevel = compressionLevel;
    init();
  }

  ~Lz4FrameStreamCompressor() override {
    if (cctx_ != nullptr) {
      LZ4F_freeCompressionContext(cctx_);
    }
  }

  StreamCompressResult compress(
      const uint8_t* input,
      int64_t inputLen,
      uint8_t* output,
      int64_t outputLen) override {
    size_t dstCap = static_cast<size_t>(outputLen);
    size_t bytesWritten = 0;

    // Write the frame header on the first compress call.
    if (!headerWritten_) {
      if (dstCap < LZ4F_HEADER_SIZE_MAX) {
        // Output too small for header; report no progress so caller retries.
        return {0, 0};
      }
      size_t h = LZ4F_compressBegin(cctx_, output, dstCap, &prefs_);
      if (LZ4F_isError(h)) {
        ARROW_LOG(WARNING) << "LZ4F_compressBegin failed: " << LZ4F_getErrorName(h);
        return {0, 0};
      }
      output += h;
      dstCap -= h;
      bytesWritten += h;
      headerWritten_ = true;
    }

    size_t ret = LZ4F_compressUpdate(
        cctx_, output, dstCap, input, static_cast<size_t>(inputLen), nullptr);
    if (LZ4F_isError(ret)) {
      ARROW_LOG(WARNING) << "LZ4F_compressUpdate failed: " << LZ4F_getErrorName(ret);
      return {0, static_cast<int64_t>(bytesWritten)};
    }
    bytesWritten += ret;
    // LZ4F_compressUpdate consumes all input when output has enough room.
    return {inputLen, static_cast<int64_t>(bytesWritten)};
  }

  StreamEndResult end(uint8_t* output, int64_t outputLen) override {
    size_t ret = LZ4F_compressEnd(cctx_, output, static_cast<size_t>(outputLen), nullptr);
    if (LZ4F_isError(ret)) {
      ARROW_LOG(WARNING) << "LZ4F_compressEnd failed: " << LZ4F_getErrorName(ret);
      return {0, true};
    }
    // LZ4F_compressEnd writes the frame footer in a single call.
    return {static_cast<int64_t>(ret), true};
  }

  void reset() override {
    if (cctx_ != nullptr) {
      LZ4F_freeCompressionContext(cctx_);
      cctx_ = nullptr;
    }
    init();
  }

  int64_t recommendedOutputSize(int64_t inputSize) const override {
    return static_cast<int64_t>(LZ4F_compressBound(static_cast<size_t>(inputSize), &prefs_)) +
        LZ4F_HEADER_SIZE_MAX;
  }

 private:
  void init() {
    LZ4F_errorCode_t ret = LZ4F_createCompressionContext(&cctx_, LZ4F_VERSION);
    ARROW_CHECK(!LZ4F_isError(ret));
    headerWritten_ = false;
  }

  int compressionLevel_;
  LZ4F_cctx* cctx_{nullptr};
  LZ4F_preferences_t prefs_;
  bool headerWritten_{false};
};

} // namespace

std::unique_ptr<StreamCompressor> StreamCompressor::create(
    arrow::Compression::type compressionType,
    int compressionLevel) {
  switch (compressionType) {
    case arrow::Compression::ZSTD:
      return std::make_unique<ZstdStreamCompressor>(compressionLevel);
    // LZ4_FRAME: intentionally NOT using streaming. Flame-graph profiling
    // showed LZ4 streaming (strategy 1: compress-then-write) regresses ~2.6%
    // vs one-shot, because LZ4 compression is so fast that the extra buffer
    // indirection becomes net overhead. LZ4 falls back to one-shot
    // compressBuffer. (ZSTD streaming is retained; ZSTD compression is slow
    // enough that streaming overhead is relatively negligible, and it paves
    // the way for RowVector mode.)
    default:
      return nullptr; // caller falls back to one-shot
  }
}

} // namespace gluten
