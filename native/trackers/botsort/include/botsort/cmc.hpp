#pragma once

#include "botsort/types.hpp"

#include <opencv2/core.hpp>

#include <memory>

namespace botsort {

class CameraMotionCompensator {
public:
    virtual ~CameraMotionCompensator() = default;
    virtual cv::Mat Apply(const cv::Mat& image, const std::vector<Detection>& detections) = 0;

protected:
    cv::Mat Preprocess(const cv::Mat& image, bool grayscale, float scale) const;
};

std::unique_ptr<CameraMotionCompensator> CreateCameraMotionCompensator(const std::string& method);

}  // namespace botsort

