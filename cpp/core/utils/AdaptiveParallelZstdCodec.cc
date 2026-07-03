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

#include "utils/AdaptiveParallelZstdCodec.h"

#include <arrow/status.h>
#include <arrow/util/logging.h>
#include <glog/logging.h>
#include <zstd.h>

namespace gluten {
namespace {

// Parallel compression kicks in when a single buffer >= 2MB (matches bolt
// AdaptiveParallelZstdCodec::kParallelCompressionThreshold).
constexpr int64_t kParallelCompressionThreshold = 2 * 1024 * 1024;
constexpr int kWorkerNumber = 2;

// Unified zstd error helper: logs at WARNING and returns IOError Status.
// Returns OK when ret is not an error, so callers can use RETURN_NOT_OK.
arrow::Status ZSTDError(size_t ret, const char* prefixMsg) {
  if (!ZSTD_isError(ret)) {
    return arrow::Status::OK();
  }
  ARROW_LOG(WARNING) << prefixMsg << ZSTD_getErrorName(ret);
  return arrow::Status::IOError(prefixMsg, ZSTD_getErrorName(ret));
}

class AdaptiveParallelZstdCodec final : public arrow::util::Codec {
 public:
  explicit AdaptiveParallelZstdCodec(int compressionLevel)
      : compressionLevel_(resolveLevel(compressionLevel)) {}

  ~AdaptiveParallelZstdCodec() override {
    if (initCCtx_) {
      ZSTD_freeCCtx(singleCCtx_);
      ZSTD_freeCCtx(parallelCCtx_);
    }
  }

  arrow::Compression::type compression_type() const override {
    return arrow::Compression::ZSTD;
  }

  arrow::Result<int64_t> Compress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputLen,
      uint8_t* output) override {
    RETURN_NOT_OK(initCCtx());
    // Route by per-buffer input length: parallel for large buffers, single-thread otherwise.
    ZSTD_CCtx* cctx =
        (inputLen >= kParallelCompressionThreshold) ? parallelCCtx_ : singleCCtx_;
    size_t ret = ZSTD_compress2(
        cctx, output, static_cast<size_t>(outputLen), input, static_cast<size_t>(inputLen));
    RETURN_NOT_OK(ZSTDError(ret, "ZSTD compression failed: "));
    return static_cast<int64_t>(ret);
  }

  arrow::Result<int64_t> Decompress(
      int64_t inputLen,
      const uint8_t* input,
      int64_t outputLen,
      uint8_t* output) override {
    if (output == nullptr) {
      // Some zstd versions demand a valid pointer for 0-byte output:
      // https://github.com/facebook/zstd/issues/1385
      static uint8_t emptyBuffer;
      DCHECK_EQ(outputLen, 0);
      output = &emptyBuffer;
    }
    size_t ret = ZSTD_decompress(
        output, static_cast<size_t>(outputLen), input, static_cast<size_t>(inputLen));
    RETURN_NOT_OK(ZSTDError(ret, "ZSTD decompression failed: "));
    if (static_cast<int64_t>(ret) != outputLen) {
      return arrow::Status::IOError("Corrupt ZSTD compressed data.");
    }
    return static_cast<int64_t>(ret);
  }

  int64_t MaxCompressedLen(int64_t inputLen, const uint8_t* ARROW_ARG_UNUSED(input)) override {
    DCHECK_GE(inputLen, 0);
    return ZSTD_compressBound(static_cast<size_t>(inputLen));
  }

  arrow::Result<std::shared_ptr<arrow::util::Compressor>> MakeCompressor() override {
    return arrow::Status::NotImplemented(
        "Streaming compression unsupported with AdaptiveParallelZstdCodec");
  }

  arrow::Result<std::shared_ptr<arrow::util::Decompressor>> MakeDecompressor() override {
    return arrow::Status::NotImplemented(
        "Streaming decompression unsupported with AdaptiveParallelZstdCodec");
  }

  int minimum_compression_level() const override {
    return ZSTD_minCLevel();
  }
  int maximum_compression_level() const override {
    return ZSTD_maxCLevel();
  }
  int default_compression_level() const override {
    return ZSTD_CLEVEL_DEFAULT;
  }
  int compression_level() const override {
    return compressionLevel_;
  }

 private:
  static int resolveLevel(int compressionLevel) {
    // arrow::util::kUseDefaultCompressionLevel is a sentinel (negative);
    // map it to zstd's default level so ZSTD_CCtx_setParameter gets a valid value.
    return compressionLevel == arrow::util::kUseDefaultCompressionLevel ? ZSTD_CLEVEL_DEFAULT
                                                                        : compressionLevel;
  }

  arrow::Status initCCtx() {
    if (initCCtx_) {
      return arrow::Status::OK();
    }
    // Single-threaded context.
    singleCCtx_ = ZSTD_createCCtx();
    if (singleCCtx_ == nullptr) {
      return arrow::Status::IOError("ZSTD_createCCtx failed for single-threaded context.");
    }
    RETURN_NOT_OK(ZSTDError(
        ZSTD_CCtx_setParameter(singleCCtx_, ZSTD_c_compressionLevel, compressionLevel_),
        "ZSTD_c_compressionLevel on singleCCtx: "));

    // Parallel context (nbWorkers=2). nbWorkers is a CCtx-level parameter and
    // switching it per-call is costly, so a dedicated CCtx is kept.
    parallelCCtx_ = ZSTD_createCCtx();
    if (parallelCCtx_ == nullptr) {
      return arrow::Status::IOError("ZSTD_createCCtx failed for parallel context.");
    }
    RETURN_NOT_OK(ZSTDError(
        ZSTD_CCtx_setParameter(parallelCCtx_, ZSTD_c_compressionLevel, compressionLevel_),
        "ZSTD_c_compressionLevel on parallelCCtx: "));
    RETURN_NOT_OK(ZSTDError(
        ZSTD_CCtx_setParameter(parallelCCtx_, ZSTD_c_nbWorkers, kWorkerNumber),
        "ZSTD_c_nbWorkers on parallelCCtx: "));

    initCCtx_ = true;
    return arrow::Status::OK();
  }

  int compressionLevel_;
  ZSTD_CCtx* singleCCtx_{nullptr};
  ZSTD_CCtx* parallelCCtx_{nullptr};
  bool initCCtx_{false};
};

} // namespace

std::unique_ptr<arrow::util::Codec> makeAdaptiveParallelZstdCodec(int compressionLevel) {
  return std::unique_ptr<arrow::util::Codec>(new AdaptiveParallelZstdCodec(compressionLevel));
}

std::unique_ptr<arrow::util::Codec> makeDefaultAdaptiveParallelZstdCodec() {
  return makeAdaptiveParallelZstdCodec(arrow::util::kUseDefaultCompressionLevel);
}

} // namespace gluten
