#include "botsort/reid_onnx.hpp"

namespace botsort {

std::vector<Eigen::VectorXf> GetReIdFeatures(
    const OnnxReIdModel& model,
    const std::vector<Detection>& detections,
    const cv::Mat& image
) {
    std::vector<cv::Rect> boxes;
    boxes.reserve(detections.size());
    const cv::Size image_size = image.size();
    for (const auto& detection : detections) {
        const Eigen::Vector4d xyxy = detection.is_obb
            ? boxmot::trackers::base::ObbToEnclosingXyxy(detection.xywha)
            : detection.xyxy;
        boxes.push_back(boxmot::trackers::base::ClampBoxToImage(xyxy, image_size));
    }
    return model.GetFeaturesForBoxes(boxes, image);
}

}  // namespace botsort
