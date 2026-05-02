#include "botsort/reid_onnx.hpp"

namespace botsort {

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

}  // namespace botsort

