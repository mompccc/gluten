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
#include <zstd.h>

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

} // namespace

std::unique_ptr<StreamCompressor> StreamCompressor::create(
    arrow::Compression::type compressionType,
    int compressionLevel) {
  switch (compressionType) {
    case arrow::Compression::ZSTD:
      return std::make_unique<ZstdStreamCompressor>(compressionLevel);
    // LZ4_FRAME: streaming support added when lz4frame.h is available
    // (see plan Task 5). Until then LZ4 falls back to one-shot compressBuffer.
    default:
      return nullptr; // caller falls back to one-shot
  }
}

} // namespace gluten
