/*******************************************************************************
 * Copyright (C) 2020-2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "converters/ov_default_2.h"

#include "gstgvadetect.h"

#include "gva_utils.h"
#include "inference_backend/logger.h"

using namespace DetectionPlugin;
using namespace Converters;

/**
 * Applies inference results to the buffer. Extracting data from each resulting blob,
 * adding ROI to the corresponding frame and addting metas to detection_result.
 *
 * @param[in] output_blobs - blobs containing inference results.
 * @param[in] frames - frames processed during inference.
 * @param[in] detection_result - detection tensor to attach meta in.
 * @param[in] confidence_threshold - value between 0 and 1 determining the accuracy of inference results to be handled.
 * @param[in] labels - GValueArray containing layers info from output_blobs.
 *
 * @return true if everything processed without exceptions.
 *
 * @throw std::invalid_argument when either blobs are invalid or their info is invalid.
 */
bool OVDefault2Converter::process(const std::map<std::string, InferenceBackend::OutputBlob::Ptr> &output_blobs,
                                  const std::vector<std::shared_ptr<InferenceFrame>> &frames,
                                  GstStructure *detection_result, double confidence_threshold,
                                  GValueArray *labels_list) {
    ITT_TASK(__FUNCTION__);
    try {
        if (not detection_result)
            throw std::invalid_argument("detection_result tensor is nullptr");

        if (output_blobs.size() < 2) {
            throw std::runtime_error("Choosen wrong converter.");
        }

        if (frames.size() != 1) {
            throw std::runtime_error("Converter does not support batch size.");
        }

        InferenceBackend::OutputBlob::Ptr bboxes = nullptr;
        InferenceBackend::OutputBlob::Ptr labels = nullptr;

        static constexpr size_t supported_bbox_size = 5;

        for (const auto &output_blob : output_blobs) {
            // Check whether we can handle this blob instead iterator
            InferenceBackend::OutputBlob::Ptr blob = output_blob.second;
            if (not blob)
                throw std::invalid_argument("Output blob is nullptr");

            auto dims = blob->GetDims();
            size_t dims_size = dims.size();

            if (dims_size > 1 or output_blob.first == "boxes") {

                size_t object_size = dims[1];
                if (object_size != supported_bbox_size)
                    throw std::invalid_argument("Object size dimension of output blob is set to " +
                                                std::to_string(object_size) + ", but only " +
                                                std::to_string(supported_bbox_size) + " supported");

                bboxes = blob;
            } else if (dims_size == 1 or output_blob.first == "labels") {
                labels = blob;
            }
        }

        if (bboxes == nullptr or labels == nullptr) {
            throw std::runtime_error("Nothing to parse.");
        }

        const auto &bboxes_dims = bboxes->GetDims();

        const float *bboxes_data = reinterpret_cast<const float *>(bboxes->GetData());
        if (not bboxes_data)
            throw std::invalid_argument("Output blob data is nullptr");

        const uint64_t *labels_data = reinterpret_cast<const uint64_t *>(labels->GetData());
        if (not labels_data)
            throw std::invalid_argument("Output blob data is nullptr");

        size_t max_proposal_count = static_cast<size_t>(bboxes_dims[0]);

        for (size_t i = 0; i < max_proposal_count; ++i) {
            const double confidence = bboxes_data[i * supported_bbox_size + 4];
            /* discard inference results that do not match 'confidence_threshold' */
            if (confidence < confidence_threshold) {
                continue;
            }

            const uint64_t label_id = static_cast<uint64_t>(labels_data[i]);

            const float bbox_x = bboxes_data[i * supported_bbox_size + 0] / input_info.width;
            const float bbox_y = bboxes_data[i * supported_bbox_size + 1] / input_info.height;
            const float bbox_w = bboxes_data[i * supported_bbox_size + 2] / input_info.width - bbox_x;
            const float bbox_h = bboxes_data[i * supported_bbox_size + 3] / input_info.height - bbox_y;

            addRoi(frames[0], bbox_x, bbox_y, bbox_w, bbox_h, label_id, confidence,
                   gst_structure_copy(detection_result),
                   labels_list); // each ROI gets its own copy, which is then
                                 // owned by GstVideoRegionOfInterestMeta
        }

    } catch (const std::exception &e) {
        std::throw_with_nested(std::runtime_error("Failed to do OV2 post-processing"));
    }
    return true;
}
