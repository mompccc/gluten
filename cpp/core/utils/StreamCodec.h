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

#include <arrow/util/compression.h>

#include <cstdint>
#include <memory>

namespace gluten {

/// Result of a streaming compress operation.
struct StreamCompressResult {
  int64_t bytesRead;    ///< bytes consumed from input
  int64_t bytesWritten; ///< bytes written to output
};

/// Result of a streaming end operation.
struct StreamEndResult {
  int64_t bytesWritten;
  bool noMoreOutput; ///< true when end is complete
};

/// Streaming compressor interface for incremental compression.
///
/// Used by Payload.cc::compressAndFlush to compress a buffer via the streaming
/// API (ZSTD_compressStream / LZ4F_compressUpdate) instead of one-shot Compress.
/// Output is a standard frame, byte-compatible with one-shot Decompress.
class StreamCompressor {
 public:
  virtual ~StreamCompressor() = default;

  /// Compress input incrementally. Returns bytes consumed/written.
  /// If bytesRead == 0 and inputLen > 0, output buffer was too small; caller
  /// should flush output and retry with the same input.
  virtual StreamCompressResult compress(
      const uint8_t* input,
      int64_t inputLen,
      uint8_t* output,
      int64_t outputLen) = 0;

  /// End the compression stream, writing frame footer.
  /// Returns noMoreOutput=true when fully ended. May need multiple calls.
  /// After end(), reset() is required before reuse.
  virtual StreamEndResult end(uint8_t* output, int64_t outputLen) = 0;

  /// Reset to initial state for a new stream.
  virtual void reset() = 0;

  /// Upper bound on output size for given input size (for pre-allocation).
  virtual int64_t recommendedOutputSize(int64_t inputSize) const = 0;

  /// Factory. Returns nullptr for unsupported codec types (caller falls back
  /// to one-shot). Supports ZSTD. LZ4_FRAME supported when lz4frame.h available.
  static std::unique_ptr<StreamCompressor> create(
      arrow::Compression::type compressionType,
      int compressionLevel);
};

} // namespace gluten
