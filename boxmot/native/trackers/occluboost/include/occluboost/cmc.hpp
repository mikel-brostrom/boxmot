#pragma once

#include "occluboost/types.hpp"

#include <opencv2/core.hpp>

#include <memory>

namespace occluboost {

class CameraMotionCompensator {
public:
    virtual ~CameraMotionCompensator() = default;
    virtual cv::Mat Apply(const cv::Mat& image, const std::vector<Detection>& detections) = 0;

protected:
    cv::Mat Preprocess(const cv::Mat& image, bool grayscale, float scale) const;
};

std::unique_ptr<CameraMotionCompensator> CreateCameraMotionCompensator(const std::string& method);

}  // namespace occluboost
