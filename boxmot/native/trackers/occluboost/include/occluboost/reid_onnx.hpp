#pragma once

#include "occluboost/types.hpp"
#include "boxmot/trackers/base/reid_onnx.hpp"

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <vector>

namespace occluboost {

// Reuse the shared base implementation; OBB detections are warped to
// straightened axis-aligned crops via the shared per-detection helper, AABB
// detections are clamped and cropped directly.
using OnnxReIdModel = boxmot::trackers::base::OnnxReIdModel;
using boxmot::trackers::base::MaybeCreateOnnxReIdModel;

std::vector<Eigen::VectorXf> GetReIdFeatures(
    const OnnxReIdModel& model,
    const std::vector<Detection>& detections,
    const cv::Mat& image
);

}  // namespace occluboost
