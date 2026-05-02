#pragma once

#include "botsort/types.hpp"
#include "boxmot/trackers/base/reid_onnx.hpp"

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <vector>

namespace botsort {

// Reuse the shared base implementation; OBB detections are warped to
// straightened axis-aligned crops, AABB ones are clamped and cropped
// directly. See ``boxmot::trackers::base::GetReIdFeaturesForDetections``.
using OnnxReIdModel = boxmot::trackers::base::OnnxReIdModel;
using TimedReIdFeatures = boxmot::trackers::base::TimedReIdFeatures;
using boxmot::trackers::base::MaybeCreateOnnxReIdModel;

// Compute embeddings for a batch of BoTSORT detections, dispatching AABB and
// OBB rows through the shared shared-base helper.
std::vector<Eigen::VectorXf> GetReIdFeatures(
    const OnnxReIdModel& model,
    const std::vector<Detection>& detections,
    const cv::Mat& image
);

// Same as ``GetReIdFeatures`` but also returns per-phase wall-clock timings
// so the tracker can fan them out through its C ABI.
TimedReIdFeatures GetReIdFeaturesTimed(
    const OnnxReIdModel& model,
    const std::vector<Detection>& detections,
    const cv::Mat& image
);

}  // namespace botsort
