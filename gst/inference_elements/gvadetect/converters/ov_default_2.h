/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/
#pragma once

#include "converters/converter.h"

namespace DetectionPlugin {
namespace Converters {

class OVDefault2Converter : public Converter {
  public:
    bool process(const std::map<std::string, InferenceBackend::OutputBlob::Ptr> &output_blobs,
                 const std::vector<std::shared_ptr<InferenceFrame>> &frames, GstStructure *detection_result,
                 double confidence_threshold, GValueArray *labels) override;
};
} // namespace Converters
} // namespace DetectionPlugin
