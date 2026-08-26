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
    static cv::Mat GenerateMask(
        const cv::Size& size,
        const std::vector<Detection>& detections,
        double scale_x,
        double scale_y
    );
};

namespace detail {

bool IsValidAffine(
    const cv::Mat& affine,
    double min_abs_determinant = 1.0e-6,
    double max_abs_determinant = 1.0e6
);

}  // namespace detail

std::unique_ptr<CameraMotionCompensator> CreateCameraMotionCompensator(const std::string& method);

}  // namespace botsort
