#pragma once

#include "botsort/types.hpp"
#include "boxmot/trackers/base/reid_onnx.hpp"

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <vector>

namespace botsort {

// Reuse the shared base implementation; per-tracker glue is just the
// detection-to-rect conversion below.
using OnnxReIdModel = boxmot::trackers::base::OnnxReIdModel;
using boxmot::trackers::base::MaybeCreateOnnxReIdModel;

// Compute embeddings for a batch of BoTSORT detections, mapping OBB rows to
// their enclosing AABB before crop extraction.
std::vector<Eigen::VectorXf> GetReIdFeatures(
    const OnnxReIdModel& model,
    const std::vector<Detection>& detections,
    const cv::Mat& image
);

}  // namespace botsort
