/*******************************************************************************
 * Copyright (C) 2018-2020 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * Reidentification gallery implementation based on smart classroom demo
 * See https://github.com/opencv/open_model_zoo/tree/2018/demos/smart_classroom_demo
 * Differences:
 * Store features in separate feature file instead of embedding into images
 * Adapted code style to match with Video Analytics GStreamer* plugins project
 * Fixed warnings
 ******************************************************************************/

#pragma once

#include "inference_backend/safe_arithmetic.h"

#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>

#include <opencv2/core/core.hpp>

#include <map>
#include <string>
#include <vector>

struct GalleryObject {
    const std::vector<cv::Mat> embeddings;
    const std::string label;
    const int id;

    std::vector<float> embedding_sizes;

    GalleryObject(const std::vector<cv::Mat> &embeddings, const std::string &label, int id)
        : embeddings(embeddings), label(label), id(id) {
        embedding_sizes.clear();

        for (const auto &embedding : embeddings) {
            float size = safe_convert<float>(embedding.dot(embedding));
            embedding_sizes.push_back(size);
        }
    }
};

class EmbeddingsGallery {
  public:
    static const std::string unknown_label;
    static const int unknown_id;
    EmbeddingsGallery(GstBaseTransform *base_transform, std::string ids_list, double threshold);
    size_t size() const;
    std::vector<std::pair<int, float>> GetIDsByEmbeddings(const std::vector<cv::Mat> &embeddings) const;
    std::string GetLabelByID(int id) const;
    std::vector<std::string> GetIDToLabelMap() const;

  protected:
    float ComputeCosineDistance(const cv::Mat &descr1, const cv::Mat &descr2, float reference_emb_size) const {
        float xx = safe_convert<float>(descr1.dot(descr1));
        float yy = reference_emb_size;
        float xy = safe_convert<float>(descr1.dot(descr2));
        float norm = sqrt(xx * yy) + 1e-6f;
        float cosine_similarity = xy / norm;
        return cosine_similarity;
    }

  private:
    std::vector<int> idx_to_id;
    double reid_threshold;
    std::vector<GalleryObject> identities;
};
