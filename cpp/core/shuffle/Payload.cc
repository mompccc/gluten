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
#include "shuffle/Payload.h"

#include <arrow/buffer.h>
#include <arrow/io/memory.h>
#include <arrow/util/bitmap.h>
#include <iostream>
#include <numeric>

#include "shuffle/Options.h"
#include "shuffle/Utils.h"
#include "utils/Exception.h"
#include "utils/StreamCodec.h"
#include "utils/Timer.h"

namespace gluten {
namespace {

static const Payload::Type kCompressedType = gluten::BlockPayload::kCompressed;
static const Payload::Type kUncompressedType = gluten::BlockPayload::kUncompressed;

static constexpr int64_t kZeroLengthBuffer = 0;
static constexpr int64_t kNullBuffer = -1;
static constexpr int64_t kUncompressedBuffer = -2;

template <typename T>
void write(uint8_t** dst, T data) {
  memcpy(*dst, &data, sizeof(T));
  *dst += sizeof(T);
}

template <typename T>
T* advance(uint8_t** dst) {
  auto ptr = reinterpret_cast<T*>(*dst);
  *dst += sizeof(T);
  return ptr;
}

arrow::Result<uint8_t> readType(arrow::io::InputStream* inputStream) {
  uint8_t type;
  ARROW_ASSIGN_OR_RAISE(auto bytes, inputStream->Read(sizeof(Payload::Type), &type));
  if (bytes == 0) {
    // Reach EOS.
    return 0;
  }
  return type;
}

arrow::Result<int64_t> compressBuffer(
    const std::shared_ptr<arrow::Buffer>& buffer,
    uint8_t* output,
    int64_t outputLength,
    arrow::util::Codec* codec) {
  auto outputPtr = &output;
  if (!buffer) {
    write<int64_t>(outputPtr, kNullBuffer);
    return sizeof(int64_t);
  }
  if (buffer->size() == 0) {
    write<int64_t>(outputPtr, kZeroLengthBuffer);
    return sizeof(int64_t);
  }
  static const int64_t kCompressedBufferHeaderLength = 2 * sizeof(int64_t);
  auto* compressedLengthPtr = advance<int64_t>(outputPtr);
  write(outputPtr, static_cast<int64_t>(buffer->size()));
  ARROW_ASSIGN_OR_RAISE(
      auto compressedLength, codec->Compress(buffer->size(), buffer->data(), outputLength, *outputPtr));
  if (compressedLength >= buffer->size()) {
    // Write uncompressed buffer.
    memcpy(*outputPtr, buffer->data(), buffer->size());
    *compressedLengthPtr = kUncompressedBuffer;
    return kCompressedBufferHeaderLength + buffer->size();
  }
  *compressedLengthPtr = static_cast<int64_t>(compressedLength);
  return kCompressedBufferHeaderLength + compressedLength;
}

// Concatenate column buffers into [lengthBuffer, valueBuffer] for RowVector
// mode. Aligned with bolt BlockPayload::concatBuffer (Payload.cpp:343-407).
//
// lengthBuffer (int64[bufferCount+3]):
//   [0] = total uncompressed size of valueBuffer
//   [1] = bufferCount (number of source buffers)
//   [2] = hasComplexType (always 0 in this version; complex types fall back
//         to BUFFER mode before reaching here)
//   [3..] = each buffer's size, or kNullBuffer/kZeroLengthBuffer sentinel
//
// valueBuffer = contiguous concat of non-null non-empty source buffers.
arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>> concatBuffersRowVector(
    std::vector<std::shared_ptr<arrow::Buffer>>&& buffers,
    arrow::MemoryPool* pool) {
  auto sourceBuffers = std::move(buffers);
  auto bufferCount = static_cast<int64_t>(sourceBuffers.size());

  // Allocate length buffer.
  ARROW_ASSIGN_OR_RAISE(
      auto lengthBuffer, arrow::AllocateResizableBuffer((bufferCount + 3) * sizeof(int64_t), pool));
  auto* lengthPtr = reinterpret_cast<int64_t*>(lengthBuffer->mutable_data());

  // Compute total uncompressed size.
  int64_t totalSize = std::accumulate(sourceBuffers.begin(), sourceBuffers.end(), 0LL, [](auto sum, const auto& b) {
    return b ? sum + b->size() : sum;
  });

  // Allocate value buffer.
  ARROW_ASSIGN_OR_RAISE(auto valueBuffer, arrow::AllocateResizableBuffer(totalSize, pool));
  auto* valuePtr = valueBuffer->mutable_data();

  // Fill length buffer + concat value buffer.
  int64_t pos = 0;
  lengthPtr[pos++] = totalSize;
  lengthPtr[pos++] = bufferCount;
  lengthPtr[pos++] = 0; // hasComplexType = 0 (complex types use BUFFER mode)
  int64_t offset = 0;
  for (const auto& buffer : sourceBuffers) {
    if (!buffer) {
      lengthPtr[pos++] = kNullBuffer;
    } else if (buffer->size() == 0) {
      lengthPtr[pos++] = kZeroLengthBuffer;
    } else {
      std::memcpy(valuePtr + offset, buffer->data(), buffer->size());
      lengthPtr[pos++] = static_cast<int64_t>(buffer->size());
      offset += buffer->size();
    }
  }
  return std::vector<std::shared_ptr<arrow::Buffer>>{
      std::move(lengthBuffer), std::move(valueBuffer)};
}

// Compress a buffer via the streaming API into a newly allocated buffer.
// Returns the compressed bytes, or nullptr if streaming is unsupported for this
// codec type (caller falls back to one-shot compressBuffer).
arrow::Result<std::shared_ptr<arrow::Buffer>> compressBufferStreaming(
    const std::shared_ptr<arrow::Buffer>& buffer,
    arrow::util::Codec* codec,
    arrow::MemoryPool* pool) {
  auto streamCompressor = StreamCompressor::create(codec->compression_type(), codec->compression_level());
  if (streamCompressor == nullptr) {
    // Unsupported codec type; caller will use one-shot path.
    // Return a null shared_ptr with OK status (not an error).
    return std::shared_ptr<arrow::Buffer>{};
  }

  ARROW_ASSIGN_OR_RAISE(
      auto innerStream,
      arrow::io::BufferOutputStream::Create(
          streamCompressor->recommendedOutputSize(buffer->size()), pool));

  const uint8_t* inPtr = buffer->data();
  int64_t remaining = buffer->size();
  // Chunked output buffer.
  constexpr int64_t kChunk = 64 * 1024;
  std::vector<uint8_t> outChunk(static_cast<size_t>(kChunk));

  // Feed all input.
  while (remaining > 0) {
    auto r = streamCompressor->compress(inPtr, remaining, outChunk.data(), kChunk);
    if (r.bytesRead == 0 && remaining > 0) {
      // Output buffer full; flush what was written and retry.
      RETURN_NOT_OK(innerStream->Write(outChunk.data(), r.bytesWritten));
      continue;
    }
    inPtr += r.bytesRead;
    remaining -= r.bytesRead;
    if (r.bytesWritten > 0) {
      RETURN_NOT_OK(innerStream->Write(outChunk.data(), r.bytesWritten));
    }
  }
  // End stream.
  while (true) {
    auto e = streamCompressor->end(outChunk.data(), kChunk);
    if (e.bytesWritten > 0) {
      RETURN_NOT_OK(innerStream->Write(outChunk.data(), e.bytesWritten));
    }
    if (e.noMoreOutput) {
      break;
    }
  }
  return innerStream->Finish();
}

arrow::Status compressAndFlush(
    const std::shared_ptr<arrow::Buffer>& buffer,
    arrow::io::OutputStream* outputStream,
    arrow::util::Codec* codec,
    arrow::MemoryPool* pool,
    int64_t& compressTime,
    int64_t& writeTime) {
  if (!buffer) {
    ScopedTimer timer(&writeTime);
    RETURN_NOT_OK(outputStream->Write(&kNullBuffer, sizeof(int64_t)));
    return arrow::Status::OK();
  }
  if (buffer->size() == 0) {
    ScopedTimer timer(&writeTime);
    RETURN_NOT_OK(outputStream->Write(&kZeroLengthBuffer, sizeof(int64_t)));
    return arrow::Status::OK();
  }
  ScopedTimer timer(&compressTime);

  // Try streaming compression first (ZSTD, LZ4 when available).
  ARROW_ASSIGN_OR_RAISE(auto compressedBuffer, compressBufferStreaming(buffer, codec, pool));

  int64_t compressedSize;
  if (compressedBuffer != nullptr) {
    // Streaming path succeeded.
    compressedSize = static_cast<int64_t>(compressedBuffer->size());
    if (compressedSize >= buffer->size()) {
      // Negative optimization: store uncompressed.
      timer.switchTo(&writeTime);
      int64_t header[2] = {kUncompressedBuffer, static_cast<int64_t>(buffer->size())};
      RETURN_NOT_OK(outputStream->Write(header, sizeof(header)));
      RETURN_NOT_OK(outputStream->Write(buffer->data(), buffer->size()));
      return arrow::Status::OK();
    }
    timer.switchTo(&writeTime);
    int64_t header[2] = {compressedSize, static_cast<int64_t>(buffer->size())};
    RETURN_NOT_OK(outputStream->Write(header, sizeof(header)));
    RETURN_NOT_OK(outputStream->Write(compressedBuffer->data(), compressedSize));
    return arrow::Status::OK();
  }

  // Fallback: one-shot path (for codecs without streaming support, e.g. GZIP/QAT).
  auto maxCompressedLength = codec->MaxCompressedLen(buffer->size(), buffer->data());
  ARROW_ASSIGN_OR_RAISE(
      auto compressed, arrow::AllocateResizableBuffer(sizeof(int64_t) * 2 + maxCompressedLength, pool));
  auto output = compressed->mutable_data();
  ARROW_ASSIGN_OR_RAISE(compressedSize, compressBuffer(buffer, output, maxCompressedLength, codec));

  timer.switchTo(&writeTime);
  RETURN_NOT_OK(outputStream->Write(compressed->data(), compressedSize));
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Buffer>>
readUncompressedBuffer(arrow::io::InputStream* inputStream, arrow::MemoryPool* pool, int64_t& deserializedTime) {
  ScopedTimer timer(&deserializedTime);
  int64_t bufferLength;
  RETURN_NOT_OK(inputStream->Read(sizeof(int64_t), &bufferLength));
  if (bufferLength == kNullBuffer) {
    return nullptr;
  }
  ARROW_ASSIGN_OR_RAISE(auto buffer, arrow::AllocateResizableBuffer(bufferLength, pool));
  RETURN_NOT_OK(inputStream->Read(bufferLength, buffer->mutable_data()));
  return buffer;
}

arrow::Result<std::shared_ptr<arrow::Buffer>> readCompressedBuffer(
    arrow::io::InputStream* inputStream,
    const std::shared_ptr<arrow::util::Codec>& codec,
    arrow::MemoryPool* pool,
    int64_t& deserializeTime,
    int64_t& decompressTime) {
  ScopedTimer timer(&deserializeTime);
  int64_t compressedLength;
  RETURN_NOT_OK(inputStream->Read(sizeof(int64_t), &compressedLength));
  if (compressedLength == kNullBuffer) {
    return nullptr;
  }
  if (compressedLength == kZeroLengthBuffer) {
    return zeroLengthNullBuffer();
  }

  int64_t uncompressedLength;
  RETURN_NOT_OK(inputStream->Read(sizeof(int64_t), &uncompressedLength));
  if (compressedLength == kUncompressedBuffer) {
    ARROW_ASSIGN_OR_RAISE(auto uncompressed, arrow::AllocateResizableBuffer(uncompressedLength, pool));
    RETURN_NOT_OK(inputStream->Read(uncompressedLength, uncompressed->mutable_data()));
    return uncompressed;
  }
  ARROW_ASSIGN_OR_RAISE(auto compressed, arrow::AllocateResizableBuffer(compressedLength, pool));
  RETURN_NOT_OK(inputStream->Read(compressedLength, compressed->mutable_data()));

  timer.switchTo(&decompressTime);
  ARROW_ASSIGN_OR_RAISE(auto output, arrow::AllocateResizableBuffer(uncompressedLength, pool));
  RETURN_NOT_OK(codec->Decompress(compressedLength, compressed->data(), uncompressedLength, output->mutable_data()));
  return output;
}

} // namespace

Payload::Payload(Payload::Type type, uint32_t numRows, const std::vector<bool>* isValidityBuffer)
    : type_(type), numRows_(numRows), isValidityBuffer_(isValidityBuffer) {}

std::string Payload::toString() const {
  static std::string kUncompressedString = "Payload::kUncompressed";
  static std::string kCompressedString = "Payload::kCompressed";
  static std::string kToBeCompressedString = "Payload::kToBeCompressed";

  if (type_ == kUncompressed) {
    return kUncompressedString;
  }
  if (type_ == kCompressed) {
    return kCompressedString;
  }
  return kToBeCompressedString;
}

arrow::Result<std::unique_ptr<BlockPayload>> BlockPayload::fromBuffers(
    Payload::Type payloadType,
    uint32_t numRows,
    std::vector<std::shared_ptr<arrow::Buffer>> buffers,
    const std::vector<bool>* isValidityBuffer,
    arrow::MemoryPool* pool,
    arrow::util::Codec* codec,
    std::shared_ptr<arrow::Buffer> compressed,
    PayloadMode mode) {
  if (payloadType == Payload::Type::kCompressed) {
    Timer compressionTime;
    compressionTime.start();
    // RowVector mode: concat all buffers into length+value first, reducing
    // compress calls from O(buffers) to O(1)=2. Aligned with bolt concatBuffer.
    if (mode == PayloadMode::kRowVector) {
      ARROW_RETURN_IF(compressed != nullptr, arrow::Status::Invalid("RowVector mode does not support pre-allocated compressed buffer."));
      std::vector<std::shared_ptr<arrow::Buffer>> concatBuffers;
      ARROW_ASSIGN_OR_RAISE(concatBuffers, concatBuffersRowVector(std::move(buffers), pool));
      buffers = std::move(concatBuffers);
    }
    // Compress.
    auto maxLength = maxCompressedLength(buffers, codec);
    std::shared_ptr<arrow::Buffer> compressedBuffer;
    uint8_t* output;
    if (compressed) {
      ARROW_RETURN_IF(
          compressed->size() < maxLength,
          arrow::Status::Invalid(
              "Compressed buffer length < maxCompressedLength. (", compressed->size(), " vs ", maxLength, ")"));
      output = const_cast<uint8_t*>(compressed->data());
    } else {
      ARROW_ASSIGN_OR_RAISE(compressedBuffer, arrow::AllocateResizableBuffer(maxLength, pool));
      output = compressedBuffer->mutable_data();
    }

    int64_t actualLength = 0;
    // Compress buffers one by one.
    for (auto& buffer : buffers) {
      auto availableLength = maxLength - actualLength;
      // Release buffer after compression.
      ARROW_ASSIGN_OR_RAISE(auto compressedSize, compressBuffer(std::move(buffer), output, availableLength, codec));
      output += compressedSize;
      actualLength += compressedSize;
    }

    ARROW_RETURN_IF(actualLength < 0, arrow::Status::Invalid("Writing compressed buffer out of bound."));
    if (compressed) {
      compressedBuffer = std::make_shared<arrow::Buffer>(compressed->data(), actualLength);
    } else {
      RETURN_NOT_OK(std::dynamic_pointer_cast<arrow::ResizableBuffer>(compressedBuffer)->Resize(actualLength));
    }
    compressionTime.stop();
    auto payload = std::unique_ptr<BlockPayload>(
        new BlockPayload(Type::kCompressed, numRows, {compressedBuffer}, isValidityBuffer, pool, codec, mode));
    payload->setCompressionTime(compressionTime.realTimeUsed());
    return payload;
  }
  return std::unique_ptr<BlockPayload>(
      new BlockPayload(payloadType, numRows, std::move(buffers), isValidityBuffer, pool, codec, mode));
}

arrow::Status BlockPayload::serialize(arrow::io::OutputStream* outputStream) {
  switch (type_) {
    case Type::kUncompressed: {
      ScopedTimer timer(&writeTime_);
      RETURN_NOT_OK(outputStream->Write(&kUncompressedType, sizeof(Type)));
      RETURN_NOT_OK(outputStream->Write(&numRows_, sizeof(uint32_t)));
      auto mode = static_cast<uint8_t>(PayloadMode::kBuffer);
      RETURN_NOT_OK(outputStream->Write(&mode, sizeof(uint8_t)));
      uint32_t numBuffers = buffers_.size();
      RETURN_NOT_OK(outputStream->Write(&numBuffers, sizeof(uint32_t)));
      for (auto& buffer : buffers_) {
        if (!buffer) {
          RETURN_NOT_OK(outputStream->Write(&kNullBuffer, sizeof(int64_t)));
          continue;
        }
        int64_t bufferSize = buffer->size();
        RETURN_NOT_OK(outputStream->Write(&bufferSize, sizeof(int64_t)));
        if (bufferSize > 0) {
          RETURN_NOT_OK(outputStream->Write(std::move(buffer)));
        }
      }
    } break;
    case Type::kToBeCompressed: {
      {
        ScopedTimer timer(&writeTime_);
        RETURN_NOT_OK(outputStream->Write(&kCompressedType, sizeof(Type)));
        RETURN_NOT_OK(outputStream->Write(&numRows_, sizeof(uint32_t)));
        auto mode = static_cast<uint8_t>(mode_);
        RETURN_NOT_OK(outputStream->Write(&mode, sizeof(uint8_t)));
        uint32_t numBuffers = buffers_.size();
        RETURN_NOT_OK(outputStream->Write(&numBuffers, sizeof(uint32_t)));
      }
      for (auto& buffer : buffers_) {
        RETURN_NOT_OK(compressAndFlush(std::move(buffer), outputStream, codec_, pool_, compressTime_, writeTime_));
      }
    } break;
    case Type::kCompressed: {
      ScopedTimer timer(&writeTime_);
      RETURN_NOT_OK(outputStream->Write(&kCompressedType, sizeof(Type)));
      RETURN_NOT_OK(outputStream->Write(&numRows_, sizeof(uint32_t)));
      auto mode = static_cast<uint8_t>(mode_);
      RETURN_NOT_OK(outputStream->Write(&mode, sizeof(uint8_t)));
      // For RowVector mode, buffers_[0] already contains the packed
      // compress(lengthBuffer)+compress(valueBuffer) blob; write it directly
      // (no numBuffers prefix — count is derived from schema on read).
      // For BUFFER mode, write numBuffers then the blob.
      if (mode_ == PayloadMode::kBuffer) {
        uint32_t buffers = numBuffers();
        RETURN_NOT_OK(outputStream->Write(&buffers, sizeof(uint32_t)));
      }
      RETURN_NOT_OK(outputStream->Write(std::move(buffers_[0])));
    } break;
    case Type::kRaw: {
      ScopedTimer timer(&writeTime_);
      RETURN_NOT_OK(outputStream->Write(std::move(buffers_[0])));
    } break;
  }
  buffers_.clear();
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Buffer>> BlockPayload::readBufferAt(uint32_t pos) {
  if (type_ == Type::kCompressed) {
    return arrow::Status::Invalid("Cannot read buffer from compressed BlockPayload.");
  }
  if (type_ == Type::kRaw && pos != 0) {
    return arrow::Status::Invalid("Read buffer pos from raw should only be 0, but got " + std::to_string(pos));
  }
  return std::move(buffers_[pos]);
}

arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>> BlockPayload::deserializeRowVectorModeBuffers(
    arrow::io::InputStream* inputStream,
    const std::shared_ptr<arrow::util::Codec>& codec,
    arrow::MemoryPool* pool,
    int64_t& deserializeTime,
    int64_t& decompressTime) {
  // Read length buffer (compressed) then value buffer (compressed).
  // Aligned with bolt BlockPayload::deserializeRowVectorModeBuffers.
  std::shared_ptr<arrow::Buffer> lengthBuffer;
  ARROW_ASSIGN_OR_RAISE(
      lengthBuffer, readCompressedBuffer(inputStream, codec, pool, deserializeTime, decompressTime));
  ARROW_RETURN_IF(
      lengthBuffer == nullptr,
      arrow::Status::Invalid("RowVector mode length buffer should not be nullptr"));

  std::shared_ptr<arrow::Buffer> valueBuffer;
  ARROW_ASSIGN_OR_RAISE(
      valueBuffer, readCompressedBuffer(inputStream, codec, pool, deserializeTime, decompressTime));
  ARROW_RETURN_IF(
      valueBuffer == nullptr,
      arrow::Status::Invalid("RowVector mode value buffer should not be nullptr"));

  const auto* lengthPtr = reinterpret_cast<const int64_t*>(lengthBuffer->data());
  int64_t uncompressLength = lengthPtr[0];
  int64_t bufferCount = lengthPtr[1];
  int64_t hasComplexType = lengthPtr[2];
  ARROW_RETURN_IF(
      uncompressLength != valueBuffer->size(),
      arrow::Status::Invalid(
          "RowVector uncompressLength " + std::to_string(uncompressLength) +
          " != valueBuffer size " + std::to_string(valueBuffer->size())));

  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.reserve(static_cast<size_t>(bufferCount));
  int64_t bufferOffset = 0;
  for (auto i = 3; i < bufferCount + 3; ++i) {
    if (lengthPtr[i] == kNullBuffer) {
      buffers.push_back(nullptr);
    } else if (lengthPtr[i] == kZeroLengthBuffer) {
      buffers.push_back(zeroLengthNullBuffer());
    } else {
      buffers.push_back(arrow::SliceBuffer(valueBuffer, bufferOffset, lengthPtr[i]));
      bufferOffset += lengthPtr[i];
    }
  }
  ARROW_RETURN_IF(
      bufferOffset != uncompressLength,
      arrow::Status::Invalid(
          "RowVector accumulated length " + std::to_string(bufferOffset) +
          " != uncompressLength " + std::to_string(uncompressLength)));
  if (hasComplexType) {
    // Read the separately-compressed complex-type buffer (this version always
    // falls back to BUFFER mode for complex types, so this should not trigger;
    // included for forward compatibility).
    buffers.emplace_back();
    ARROW_ASSIGN_OR_RAISE(
        buffers.back(), readCompressedBuffer(inputStream, codec, pool, deserializeTime, decompressTime));
  }
  return buffers;
}

arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>> BlockPayload::deserialize(
    arrow::io::InputStream* inputStream,
    const std::shared_ptr<arrow::util::Codec>& codec,
    arrow::MemoryPool* pool,
    uint32_t& numRows,
    int64_t& deserializeTime,
    int64_t& decompressTime) {
  auto timer = std::make_unique<ScopedTimer>(&deserializeTime);
  static const std::vector<std::shared_ptr<arrow::Buffer>> kEmptyBuffers{};
  ARROW_ASSIGN_OR_RAISE(auto type, readType(inputStream));
  if (type == 0) {
    numRows = 0;
    return kEmptyBuffers;
  }
  RETURN_NOT_OK(inputStream->Read(sizeof(uint32_t), &numRows));
  // Read mode byte (new header field).
  uint8_t modeByte;
  RETURN_NOT_OK(inputStream->Read(sizeof(uint8_t), &modeByte));
  auto mode = static_cast<PayloadMode>(modeByte);
  uint32_t numBuffers = 0;
  if (mode != PayloadMode::kRowVector) {
    RETURN_NOT_OK(inputStream->Read(sizeof(uint32_t), &numBuffers));
  }
  timer.reset();

  bool isCompressionEnabled = type == Type::kCompressed;

  // RowVector mode dispatch.
  if (isCompressionEnabled && mode == PayloadMode::kRowVector) {
    return deserializeRowVectorModeBuffers(inputStream, codec, pool, deserializeTime, decompressTime);
  }

  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.reserve(numBuffers);
  for (auto i = 0; i < numBuffers; ++i) {
    buffers.emplace_back();
    if (isCompressionEnabled) {
      ARROW_ASSIGN_OR_RAISE(
          buffers.back(), readCompressedBuffer(inputStream, codec, pool, deserializeTime, decompressTime));
    } else {
      ARROW_ASSIGN_OR_RAISE(buffers.back(), readUncompressedBuffer(inputStream, pool, deserializeTime));
    }
  }
  return buffers;
}

void BlockPayload::setCompressionTime(int64_t compressionTime) {
  compressTime_ = compressionTime;
}

int64_t BlockPayload::rawSize() {
  return getBufferSize(buffers_);
}

int64_t BlockPayload::maxCompressedLength(
    const std::vector<std::shared_ptr<arrow::Buffer>>& buffers,
    arrow::util::Codec* codec) {
  // Compressed buffer layout: | buffer1 compressedLength | buffer1 uncompressedLength | buffer1 | ...
  const auto metadataLength = sizeof(int64_t) * 2 * buffers.size();
  int64_t totalCompressedLength =
      std::accumulate(buffers.begin(), buffers.end(), 0LL, [&](auto sum, const auto& buffer) {
        if (!buffer) {
          return sum;
        }
        return sum + codec->MaxCompressedLen(buffer->size(), buffer->data());
      });
  return metadataLength + totalCompressedLength;
}

arrow::Result<std::unique_ptr<InMemoryPayload>> InMemoryPayload::merge(
    std::unique_ptr<InMemoryPayload> source,
    std::unique_ptr<InMemoryPayload> append,
    arrow::MemoryPool* pool) {
  auto mergedRows = source->numRows() + append->numRows();
  auto isValidityBuffer = source->isValidityBuffer();

  auto numBuffers = append->numBuffers();
  ARROW_RETURN_IF(
      numBuffers != source->numBuffers(), arrow::Status::Invalid("Number of merging buffers doesn't match."));
  std::vector<std::shared_ptr<arrow::Buffer>> merged;
  merged.resize(numBuffers);
  for (size_t i = 0; i < numBuffers; ++i) {
    ARROW_ASSIGN_OR_RAISE(auto sourceBuffer, source->readBufferAt(i));
    ARROW_ASSIGN_OR_RAISE(auto appendBuffer, append->readBufferAt(i));
    if (isValidityBuffer->at(i)) {
      if (!sourceBuffer) {
        if (!appendBuffer) {
          merged[i] = nullptr;
        } else {
          ARROW_ASSIGN_OR_RAISE(
              auto buffer, arrow::AllocateResizableBuffer(arrow::bit_util::BytesForBits(mergedRows), pool));
          // Source is null, fill all true.
          arrow::bit_util::SetBitsTo(buffer->mutable_data(), 0, source->numRows(), true);
          // Write append bits.
          arrow::internal::CopyBitmap(
              appendBuffer->data(), 0, append->numRows(), buffer->mutable_data(), source->numRows());
          merged[i] = std::move(buffer);
        }
      } else {
        // Because sourceBuffer can be resized, need to save buffer size in advance.
        auto sourceBufferSize = sourceBuffer->size();
        auto resizable = std::dynamic_pointer_cast<arrow::ResizableBuffer>(sourceBuffer);
        auto mergedBytes = arrow::bit_util::BytesForBits(mergedRows);
        if (resizable) {
          // If source is resizable, resize and reuse source.
          RETURN_NOT_OK(resizable->Resize(mergedBytes));
        } else {
          // Otherwise copy source.
          ARROW_ASSIGN_OR_RAISE(resizable, arrow::AllocateResizableBuffer(mergedBytes, pool));
          memcpy(resizable->mutable_data(), sourceBuffer->data(), sourceBufferSize);
        }
        if (!appendBuffer) {
          arrow::bit_util::SetBitsTo(resizable->mutable_data(), source->numRows(), append->numRows(), true);
        } else {
          arrow::internal::CopyBitmap(
              appendBuffer->data(), 0, append->numRows(), resizable->mutable_data(), source->numRows());
        }
        merged[i] = std::move(resizable);
      }
    } else {
      if (appendBuffer->size() == 0) {
        merged[i] = std::move(sourceBuffer);
      } else {
        // Because sourceBuffer can be resized, need to save buffer size in advance.
        auto sourceBufferSize = sourceBuffer->size();
        auto mergedSize = sourceBufferSize + appendBuffer->size();
        auto resizable = std::dynamic_pointer_cast<arrow::ResizableBuffer>(sourceBuffer);
        if (resizable) {
          // If source is resizable, resize and reuse source.
          RETURN_NOT_OK(resizable->Resize(mergedSize));
        } else {
          // Otherwise copy source.
          ARROW_ASSIGN_OR_RAISE(resizable, arrow::AllocateResizableBuffer(mergedSize, pool));
          memcpy(resizable->mutable_data(), sourceBuffer->data(), sourceBufferSize);
        }
        // Copy append.
        memcpy(resizable->mutable_data() + sourceBufferSize, appendBuffer->data(), appendBuffer->size());
        merged[i] = std::move(resizable);
      }
    }
  }
  return std::make_unique<InMemoryPayload>(mergedRows, isValidityBuffer, std::move(merged));
}

arrow::Result<std::unique_ptr<BlockPayload>> InMemoryPayload::toBlockPayload(
    Payload::Type payloadType,
    arrow::MemoryPool* pool,
    arrow::util::Codec* codec,
    std::shared_ptr<arrow::Buffer> compressed,
    PayloadMode mode) {
  return BlockPayload::fromBuffers(
      payloadType, numRows_, std::move(buffers_), isValidityBuffer_, pool, codec, std::move(compressed), mode);
}

arrow::Status InMemoryPayload::serialize(arrow::io::OutputStream* outputStream) {
  return arrow::Status::Invalid("Cannot serialize InMemoryPayload.");
}

arrow::Result<std::shared_ptr<arrow::Buffer>> InMemoryPayload::readBufferAt(uint32_t index) {
  return std::move(buffers_[index]);
}

arrow::Status InMemoryPayload::copyBuffers(arrow::MemoryPool* pool) {
  for (auto& buffer : buffers_) {
    if (!buffer) {
      continue;
    }
    if (buffer->size() == 0) {
      buffer = zeroLengthNullBuffer();
      continue;
    }
    ARROW_ASSIGN_OR_RAISE(auto copy, arrow::AllocateResizableBuffer(buffer->size(), pool));
    memcpy(copy->mutable_data(), buffer->data(), buffer->size());
    buffer = std::move(copy);
  }
  return arrow::Status::OK();
}

int64_t InMemoryPayload::rawSize() {
  return getBufferSize(buffers_);
}

UncompressedDiskBlockPayload::UncompressedDiskBlockPayload(
    Type type,
    uint32_t numRows,
    const std::vector<bool>* isValidityBuffer,
    arrow::io::InputStream*& inputStream,
    uint64_t rawSize,
    arrow::MemoryPool* pool,
    arrow::util::Codec* codec)
    : Payload(type, numRows, isValidityBuffer),
      inputStream_(inputStream),
      rawSize_(rawSize),
      pool_(pool),
      codec_(codec) {}

arrow::Result<std::shared_ptr<arrow::Buffer>> UncompressedDiskBlockPayload::readBufferAt(uint32_t index) {
  return arrow::Status::Invalid("Cannot read buffer from UncompressedDiskBlockPayload.");
}

arrow::Status UncompressedDiskBlockPayload::serialize(arrow::io::OutputStream* outputStream) {
  ARROW_RETURN_IF(
      inputStream_ == nullptr, arrow::Status::Invalid("inputStream_ is uninitialized before calling serialize()."));

  if (codec_ == nullptr || type_ == Payload::kUncompressed) {
    ARROW_ASSIGN_OR_RAISE(auto block, inputStream_->Read(rawSize_));
    RETURN_NOT_OK(outputStream->Write(block));
    return arrow::Status::OK();
  }

  ARROW_RETURN_IF(
      type_ != Payload::kToBeCompressed,
      arrow::Status::Invalid(
          "Invalid payload type: " + std::to_string(type_) +
          ", should be either Payload::kUncompressed or Payload::kToBeCompressed"));
  RETURN_NOT_OK(outputStream->Write(&kCompressedType, sizeof(kCompressedType)));
  RETURN_NOT_OK(outputStream->Write(&numRows_, sizeof(uint32_t)));
  // Write mode byte (BUFFER mode for disk-spill re-compress path; RowVector is
  // not applied here since buffers are read back one-by-one from disk).
  auto modeByte = static_cast<uint8_t>(PayloadMode::kBuffer);
  RETURN_NOT_OK(outputStream->Write(&modeByte, sizeof(uint8_t)));

  ARROW_ASSIGN_OR_RAISE(auto startPos, inputStream_->Tell());

  // Discard original header: type(1) | numRows(4) | mode(1) | numBuffers(4).
  Payload::Type type;
  uint32_t numRows;
  uint8_t diskMode;
  ARROW_ASSIGN_OR_RAISE(auto bytes, inputStream_->Read(sizeof(Payload::Type), &type));
  ARROW_ASSIGN_OR_RAISE(bytes, inputStream_->Read(sizeof(uint32_t), &numRows));
  ARROW_ASSIGN_OR_RAISE(bytes, inputStream_->Read(sizeof(uint8_t), &diskMode));
  uint32_t numBuffers = 0;
  ARROW_ASSIGN_OR_RAISE(bytes, inputStream_->Read(sizeof(uint32_t), &numBuffers));
  ARROW_RETURN_IF(bytes == 0 || numBuffers == 0, arrow::Status::Invalid("Cannot serialize payload with 0 buffers."));
  RETURN_NOT_OK(outputStream->Write(&numBuffers, sizeof(uint32_t)));

  // Advance Payload::Type, rows, mode and numBuffers.
  auto readPos = startPos + sizeof(Payload::Type) + sizeof(uint32_t) + sizeof(uint8_t) + sizeof(uint32_t);
  while (readPos - startPos < rawSize_) {
    ARROW_ASSIGN_OR_RAISE(auto uncompressed, readUncompressedBuffer());
    ARROW_ASSIGN_OR_RAISE(readPos, inputStream_->Tell());
    RETURN_NOT_OK(compressAndFlush(std::move(uncompressed), outputStream, codec_, pool_, compressTime_, writeTime_));
  }
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Buffer>> UncompressedDiskBlockPayload::readUncompressedBuffer() {
  ScopedTimer timer(&writeTime_);
  readPos_++;
  int64_t bufferLength;
  RETURN_NOT_OK(inputStream_->Read(sizeof(int64_t), &bufferLength));
  if (bufferLength == kNullBuffer) {
    return nullptr;
  }
  if (bufferLength == 0) {
    return zeroLengthNullBuffer();
  }
  ARROW_ASSIGN_OR_RAISE(auto buffer, inputStream_->Read(bufferLength));
  return buffer;
}

int64_t UncompressedDiskBlockPayload::rawSize() {
  return rawSize_;
}

CompressedDiskBlockPayload::CompressedDiskBlockPayload(
    uint32_t numRows,
    const std::vector<bool>* isValidityBuffer,
    arrow::io::InputStream*& inputStream,
    int64_t rawSize,
    arrow::MemoryPool* /* pool */)
    : Payload(Type::kCompressed, numRows, isValidityBuffer), inputStream_(inputStream), rawSize_(rawSize) {}

arrow::Status CompressedDiskBlockPayload::serialize(arrow::io::OutputStream* outputStream) {
  ARROW_RETURN_IF(
      inputStream_ == nullptr, arrow::Status::Invalid("inputStream_ is uninitialized before calling serialize()."));
  ScopedTimer timer(&writeTime_);
  ARROW_ASSIGN_OR_RAISE(auto block, inputStream_->Read(rawSize_));
  RETURN_NOT_OK(outputStream->Write(block));
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Buffer>> CompressedDiskBlockPayload::readBufferAt(uint32_t index) {
  return arrow::Status::Invalid("Cannot read buffer from CompressedDiskBlockPayload.");
}

int64_t CompressedDiskBlockPayload::rawSize() {
  return rawSize_;
}
} // namespace gluten
