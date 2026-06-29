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

#include <stdint.h>

#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
#include <arm_sve.h>
#endif

namespace gluten {

inline uint8_t extractBitsToByte(const uint8_t* srcAddr, uint32_t* offset) {
  uint8_t dst = 0;
  uint8_t src = 0;
  auto srcOffset = offset[0];
  src = srcAddr[srcOffset >> 3];
  dst = src >> (srcOffset & 7) | 0xfe;

  srcOffset = offset[1];
  src = srcAddr[srcOffset >> 3];
  dst &= src >> (srcOffset & 7) << 1 | 0xfd;

  srcOffset = offset[2];
  src = srcAddr[srcOffset >> 3];
  dst &= src >> (srcOffset & 7) << 2 | 0xfb;

  srcOffset = offset[3];
  src = srcAddr[srcOffset >> 3];
  dst &= src >> (srcOffset & 7) << 3 | 0xf7;

  srcOffset = offset[4];
  src = srcAddr[srcOffset >> 3];
  dst &= src >> (srcOffset & 7) << 4 | 0xef;

  srcOffset = offset[5];
  src = srcAddr[srcOffset >> 3];
  dst &= src >> (srcOffset & 7) << 5 | 0xdf;

  srcOffset = offset[6];
  src = srcAddr[srcOffset >> 3];
  dst &= src >> (srcOffset & 7) << 6 | 0xbf;

  srcOffset = offset[7];
  src = srcAddr[srcOffset >> 3];
  dst &= src >> (srcOffset & 7) << 7 | 0x7f;

  return dst;
}

#if defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
namespace detail {

// GCC SVE ACLE uses overloaded svld1_gather_index; Clang uses svld1_gather_u32index.
inline svuint32_t sveGatherU32(
    svbool_t pg,
    const uint32_t* base,
    svuint32_t indices) {
#if defined(__clang__)
  return svld1_gather_u32index(pg, base, indices);
#else
  return svld1_gather_index(pg, base, indices);
#endif
}

// Portable predicate for [start, end) u32 lanes across GCC/Clang and any SVE VL.
inline svbool_t sveWhileLtU32(uint32_t start, uint32_t end) {
#if defined(__clang__)
  return svwhilelt_b32_u32(start, end);
#else
  return svwhilelt_b32(start, end);
#endif
}

// Process exactly 8 bit offsets; loop handles VL < 8 (e.g. Kunpeng 920 128-bit SVE).
inline uint8_t extractBitsToByteSve(const uint8_t* srcAddr, uint32_t* offset) {
  uint8_t result = 0;
  uint32_t i = 0;
  while (i < 8) {
    const svbool_t pg = sveWhileLtU32(i, 8);
    const svuint32_t offsetVec = svld1_u32(pg, offset + i);
    const svuint32_t indices = svlsr_n_u32_x(pg, offsetVec, 5);
    const svuint32_t srcNullVec =
        sveGatherU32(pg, reinterpret_cast<const uint32_t*>(srcAddr), indices);
    const svuint32_t offsetIn4Byte = svand_n_u32_x(pg, offsetVec, 0x1F);
    const svuint32_t bitVec = svlsr_u32_x(pg, srcNullVec, offsetIn4Byte);
    const svuint32_t shifts = svadd_n_u32_x(pg, svindex_u32(0, 1), i);
    const svuint32_t contribution =
        svlsl_u32_x(pg, svand_n_u32_x(pg, bitVec, 1), shifts);
    result |= static_cast<uint8_t>(svaddv_u32(pg, contribution));
    i += svcntw();
  }
  return result;
}

} // namespace detail
#endif

inline uint8_t extractBitsToByteSimd(const uint8_t* srcAddr, uint32_t* offset) {
#if defined(__x86_64__)
  __m256i offsetVec = _mm256_loadu_si256((__m256i*)offset);
  __m256i srcNullVec = _mm256_i32gather_epi32(
      (const int*)srcAddr, _mm256_srli_epi32(offsetVec, 5), 4);
  __m256i offsetIn4ByteVec = _mm256_and_si256(
      offsetVec,
      _mm256_set_epi32(0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F));
  __m256i bitVec = _mm256_srav_epi32(srcNullVec, offsetIn4ByteVec);
  __m256i bitInSignVec = _mm256_slli_epi32(bitVec, 31);
  return (uint8_t)_mm256_movemask_ps(_mm256_cvtepi32_ps(bitInSignVec));
#elif defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
  return detail::extractBitsToByteSve(srcAddr, offset);
#else
  return extractBitsToByte(srcAddr, offset);
#endif
}

} // namespace gluten
