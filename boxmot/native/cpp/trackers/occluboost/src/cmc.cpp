#include "occluboost/cmc.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace occluboost {

cv::Mat CameraMotionCompensator::Preprocess(const cv::Mat& image, const bool grayscale, const float scale) const {
    cv::Mat output = image;
    if (grayscale && image.channels() == 3) {
        cv::cvtColor(image, output, cv::COLOR_BGR2GRAY);
    } else if (grayscale && image.channels() == 4) {
        cv::cvtColor(image, output, cv::COLOR_BGRA2GRAY);
    }
    if (scale > 0.0F && scale != 1.0F) {
        cv::resize(output, output, cv::Size(), scale, scale, cv::INTER_LINEAR);
    }
    return output;
}

cv::Mat CameraMotionCompensator::GenerateMask(
    const cv::Size& size,
    const std::vector<Detection>& detections,
    const double scale_x,
    const double scale_y
) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
    const int safe_x1 = static_cast<int>(0.02 * static_cast<double>(size.width));
    const int safe_y1 = static_cast<int>(0.02 * static_cast<double>(size.height));
    const int safe_x2 = static_cast<int>(0.98 * static_cast<double>(size.width));
    const int safe_y2 = static_cast<int>(0.98 * static_cast<double>(size.height));
    if (safe_x2 > safe_x1 && safe_y2 > safe_y1) {
        cv::rectangle(
            mask,
            cv::Rect(safe_x1, safe_y1, safe_x2 - safe_x1, safe_y2 - safe_y1),
            cv::Scalar(255),
            cv::FILLED
        );
    }
    for (const auto& detection : detections) {
        if (detection.is_obb) {
            const auto& box = detection.xywha;
            const cv::RotatedRect rect(
                cv::Point2f(static_cast<float>(box[0]), static_cast<float>(box[1])),
                cv::Size2f(
                    std::max(static_cast<float>(box[2]), 1.0e-4F),
                    std::max(static_cast<float>(box[3]), 1.0e-4F)
                ),
                static_cast<float>(box[4] * 180.0 / CV_PI)
            );
            std::array<cv::Point2f, 4> points{};
            rect.points(points.data());
            std::vector<cv::Point> polygon;
            polygon.reserve(points.size());
            for (const auto& point : points) {
                polygon.emplace_back(
                    cvRound(static_cast<double>(point.x) * scale_x),
                    cvRound(static_cast<double>(point.y) * scale_y)
                );
            }
            cv::fillConvexPoly(mask, polygon, cv::Scalar(0));
            continue;
        }

        const auto& box = detection.xyxy;
        const int box_x1 = std::clamp(static_cast<int>(box[0] * scale_x), 0, size.width);
        const int box_y1 = std::clamp(static_cast<int>(box[1] * scale_y), 0, size.height);
        const int box_x2 = std::clamp(static_cast<int>(box[2] * scale_x), 0, size.width);
        const int box_y2 = std::clamp(static_cast<int>(box[3] * scale_y), 0, size.height);
        if (box_x2 > box_x1 && box_y2 > box_y1) {
            cv::rectangle(
                mask,
                cv::Rect(box_x1, box_y1, box_x2 - box_x1, box_y2 - box_y1),
                cv::Scalar(0),
                cv::FILLED
            );
        }
    }
    return mask;
}

namespace {

std::pair<double, double> ResizeScale(const cv::Mat& image, const cv::Mat& resized) {
    if (image.empty() || resized.empty()) {
        return {1.0, 1.0};
    }
    return {
        static_cast<double>(resized.cols) / static_cast<double>(image.cols),
        static_cast<double>(resized.rows) / static_cast<double>(image.rows),
    };
}

cv::Mat RestoreAffineScale(const cv::Mat& transform, const double scale_x, const double scale_y) {
    cv::Mat transform64;
    transform.convertTo(transform64, CV_64F);
    cv::Mat homogeneous = cv::Mat::eye(3, 3, CV_64F);
    transform64.copyTo(homogeneous(cv::Rect(0, 0, 3, 2)));
    const cv::Mat scale = (cv::Mat_<double>(3, 3) <<
        scale_x, 0.0, 0.0,
        0.0, scale_y, 0.0,
        0.0, 0.0, 1.0);
    cv::Mat restored = scale.inv() * homogeneous * scale;
    cv::Mat result;
    restored(cv::Rect(0, 0, 3, 2)).convertTo(result, CV_32F);
    return result;
}

bool IsValidAffine(
    const cv::Mat& affine,
    const double min_abs_determinant = 1.0e-6,
    const double max_abs_determinant = 1.0e6
) {
    if (affine.rows != 2 || affine.cols != 3) {
        return false;
    }

    cv::Mat affine64;
    affine.convertTo(affine64, CV_64F);
    for (int row = 0; row < affine64.rows; ++row) {
        for (int column = 0; column < affine64.cols; ++column) {
            if (!std::isfinite(affine64.at<double>(row, column))) {
                return false;
            }
        }
    }
    const double determinant = (
        affine64.at<double>(0, 0) * affine64.at<double>(1, 1)
        - affine64.at<double>(0, 1) * affine64.at<double>(1, 0)
    );
    const double abs_determinant = std::abs(determinant);
    return std::isfinite(determinant)
        && abs_determinant >= min_abs_determinant
        && abs_determinant <= max_abs_determinant;
}

bool IsValidSofAffine(const cv::Mat& affine, const cv::Mat& inliers, const std::size_t match_count) {
    constexpr int kMinInliers = 8;
    constexpr double kMinInlierRatio = 0.2;
    if (inliers.empty() || match_count == 0) {
        return false;
    }
    const int inlier_count = cv::countNonZero(inliers);
    return inlier_count >= kMinInliers
        && static_cast<double>(inlier_count) / match_count >= kMinInlierRatio
        && IsValidAffine(affine);
}

class EccCmc final : public CameraMotionCompensator {
public:
    cv::Mat Apply(const cv::Mat& image, const std::vector<Detection>& detections) override {
        cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
        if (image.empty()) {
            return warp;
        }

        cv::Mat current = Preprocess(image, true, 0.15F);
        const auto [scale_x, scale_y] = ResizeScale(image, current);
        if (prev_image_.empty()) {
            prev_image_ = current;
            return warp;
        }

        try {
            const cv::Mat mask = GenerateMask(current.size(), detections, scale_x, scale_y);
            const double correlation = cv::findTransformECC(
                prev_image_,
                current,
                warp,
                cv::MOTION_TRANSLATION,
                cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 100, 1.0e-5),
                mask,
                1
            );
            if (!std::isfinite(correlation) || correlation < 0.5 || !IsValidAffine(warp)) {
                warp = cv::Mat::eye(2, 3, CV_32F);
            } else {
                warp = RestoreAffineScale(warp, scale_x, scale_y);
            }
        } catch (const cv::Exception& exception) {
            if (exception.code != cv::Error::StsNoConv) {
                throw;
            }
            warp = cv::Mat::eye(2, 3, CV_32F);
        }

        prev_image_ = current;
        return warp;
    }

private:
    cv::Mat prev_image_;
};

class SofCmc final : public CameraMotionCompensator {
public:
    cv::Mat Apply(const cv::Mat& image, const std::vector<Detection>& detections) override {
        cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
        if (image.empty()) {
            return warp;
        }

        cv::Mat current = Preprocess(image, true, 0.15F);
        const auto [scale_x, scale_y] = ResizeScale(image, current);
        if (!initialized_ || prev_frame_.empty() || prev_keypoints_.empty()) {
            Initialize(current, detections, scale_x, scale_y);
            return warp;
        }

        std::vector<cv::Point2f> next_keypoints;
        std::vector<unsigned char> status;
        std::vector<float> errors;
        cv::calcOpticalFlowPyrLK(
            prev_frame_,
            current,
            prev_keypoints_,
            next_keypoints,
            status,
            errors,
            cv::Size(21, 21),
            3,
            cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01)
        );

        if (next_keypoints.empty() || status.empty()) {
            Reset(current, detections, scale_x, scale_y);
            return warp;
        }

        std::vector<cv::Point2f> prev_valid;
        std::vector<cv::Point2f> next_valid;
        const std::size_t point_count = std::min({status.size(), prev_keypoints_.size(), next_keypoints.size()});
        for (std::size_t i = 0; i < point_count; ++i) {
            if (
                status[i]
                && std::isfinite(prev_keypoints_[i].x)
                && std::isfinite(prev_keypoints_[i].y)
                && std::isfinite(next_keypoints[i].x)
                && std::isfinite(next_keypoints[i].y)
            ) {
                prev_valid.push_back(prev_keypoints_[i]);
                next_valid.push_back(next_keypoints[i]);
            }
        }

        if (prev_valid.size() < 4U || next_valid.size() < 4U) {
            Reset(current, detections, scale_x, scale_y);
            return warp;
        }

        cv::Mat inliers;
        cv::Mat affine = cv::estimateAffinePartial2D(
            prev_valid,
            next_valid,
            inliers,
            cv::RANSAC,
            3.0
        );
        if (IsValidSofAffine(affine, inliers, prev_valid.size())) {
            affine.convertTo(warp, CV_32F);
            warp = RestoreAffineScale(warp, scale_x, scale_y);
        }

        std::vector<cv::Point2f> refreshed = DetectKeypoints(current, detections, scale_x, scale_y);
        if (refreshed.size() < 4U) {
            refreshed = next_valid;
        }
        prev_keypoints_ = std::move(refreshed);
        prev_frame_ = current;
        initialized_ = true;
        return warp;
    }

private:
    std::vector<cv::Point2f> DetectKeypoints(
        const cv::Mat& current,
        const std::vector<Detection>& detections,
        const double scale_x,
        const double scale_y
    ) const {
        std::vector<cv::Point2f> keypoints;
        const cv::Mat mask = GenerateMask(current.size(), detections, scale_x, scale_y);
        cv::goodFeaturesToTrack(current, keypoints, 1000, 0.01, 1.0, mask, 3, false, 0.04);
        return keypoints;
    }

    void Initialize(
        const cv::Mat& current,
        const std::vector<Detection>& detections,
        const double scale_x,
        const double scale_y
    ) {
        prev_keypoints_ = DetectKeypoints(current, detections, scale_x, scale_y);
        prev_frame_ = current;
        if (prev_keypoints_.size() < 4U) {
            initialized_ = false;
            return;
        }
        cv::cornerSubPix(
            prev_frame_,
            prev_keypoints_,
            cv::Size(5, 5),
            cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01)
        );
        initialized_ = true;
    }

    void Reset(
        const cv::Mat& current,
        const std::vector<Detection>& detections,
        const double scale_x,
        const double scale_y
    ) {
        prev_keypoints_ = DetectKeypoints(current, detections, scale_x, scale_y);
        prev_frame_ = current;
        initialized_ = prev_keypoints_.size() >= 4U;
    }

    cv::Mat prev_frame_;
    std::vector<cv::Point2f> prev_keypoints_;
    bool initialized_ = false;
};

}  // namespace

std::unique_ptr<CameraMotionCompensator> CreateCameraMotionCompensator(const std::string& method) {
    if (method.empty() || method == "none") {
        return nullptr;
    }
    if (method == "ecc") {
        return std::make_unique<EccCmc>();
    }
    if (method == "sof") {
        return std::make_unique<SofCmc>();
    }
    throw std::invalid_argument("Unsupported cmc_method: " + method);
}

}  // namespace occluboost
