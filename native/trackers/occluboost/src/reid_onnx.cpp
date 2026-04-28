#include "occluboost/reid_onnx.hpp"

namespace occluboost {

std::vector<Eigen::VectorXf> GetReIdFeatures(
    const OnnxReIdModel& model,
    const std::vector<Detection>& detections,
    const cv::Mat& image
) {
    std::vector<cv::Rect> boxes;
    boxes.reserve(detections.size());
    const cv::Size image_size = image.size();
    for (const auto& detection : detections) {
        boxes.push_back(boxmot::trackers::base::ClampBoxToImage(detection.xyxy, image_size));
    }
    return model.GetFeaturesForBoxes(boxes, image);
}

}  // namespace occluboost
