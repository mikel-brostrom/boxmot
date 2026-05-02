#include "occluboost/reid_onnx.hpp"

namespace occluboost {

std::vector<Eigen::VectorXf> GetReIdFeatures(
    const OnnxReIdModel& model,
    const std::vector<Detection>& detections,
    const cv::Mat& image
) {
    return boxmot::trackers::base::GetReIdFeaturesForDetections(model, detections, image);
}

}  // namespace occluboost

