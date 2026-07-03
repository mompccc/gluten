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

#include <memory>

namespace gluten {

/// Adaptive parallel ZSTD codec for columnar shuffle.
///
/// Maintains two ZSTD_CCtx: one single-threaded, one with ZSTD_c_nbWorkers=2.
/// Routes per Compress() call based on inputLen >= kParallelCompressionThreshold
/// (2MB). Output is a standard zstd frame regardless of nbWorkers, so the read
/// side (standard ZSTD_decompress) needs no changes and is byte-compatible with
/// Arrow's default ZSTD codec.
std::unique_ptr<arrow::util::Codec> makeAdaptiveParallelZstdCodec(int compressionLevel);

/// Default factory: uses arrow::util::kUseDefaultCompressionLevel, which the
/// codec maps to ZSTD_CLEVEL_DEFAULT (3).
std::unique_ptr<arrow::util::Codec> makeDefaultAdaptiveParallelZstdCodec();

} // namespace gluten
