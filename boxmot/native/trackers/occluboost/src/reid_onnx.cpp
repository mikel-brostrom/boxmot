#include "occluboost/reid_onnx.hpp"

namespace occluboost {

TimedReIdFeatures GetReIdFeaturesTimed(
    const OnnxReIdModel& model,
    const std::vector<Detection>& detections,
    const cv::Mat& image
) {
    return boxmot::trackers::base::GetReIdFeaturesForDetections(model, detections, image);
}

std::vector<Eigen::VectorXf> GetReIdFeatures(
    const OnnxReIdModel& model,
    const std::vector<Detection>& detections,
    const cv::Mat& image
) {
    return GetReIdFeaturesTimed(model, detections, image).features;
}

}  // namespace occluboost

