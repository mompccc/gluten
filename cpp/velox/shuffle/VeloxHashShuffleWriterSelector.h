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

#include "shuffle/Options.h"
#include "shuffle/Partitioning.h"

namespace gluten {

/// Runtime hash writer variant selection (bolt decideBoltShuffleWriterType).
/// Returns kHashShuffle (V1) or kHashShuffleV2. RowBased writer is out of scope.
inline ShuffleWriterType decideVeloxHashShuffleWriterType(
    const ShuffleWriterOptions& options,
    uint32_t numColumnsExcludePid,
    uint32_t numPartitions,
    uint32_t preAllocRowCount) {
  if (options.partitioning == Partitioning::kSingle) {
    return ShuffleWriterType::kHashShuffle;
  }

  constexpr uint32_t kV1PartitionThresholdL1 = 10000;
  constexpr uint32_t kV1PartitionThresholdL2 = 50000;
  constexpr uint32_t kV1PreAllocSizeL1 = 20;
  constexpr uint32_t kV1PreAllocSizeL2 = 10;

  if (numPartitions >= kV1PartitionThresholdL1 && preAllocRowCount > kV1PreAllocSizeL1) {
    return ShuffleWriterType::kHashShuffle;
  }
  if (numPartitions >= kV1PartitionThresholdL2 && preAllocRowCount > kV1PreAllocSizeL2) {
    return ShuffleWriterType::kHashShuffle;
  }
  if (preAllocRowCount > static_cast<uint32_t>(options.useV2PreallocSizeThreshold)) {
    return ShuffleWriterType::kHashShuffle;
  }

  (void)numColumnsExcludePid;
  return ShuffleWriterType::kHashShuffleV2;
}

} // namespace gluten
