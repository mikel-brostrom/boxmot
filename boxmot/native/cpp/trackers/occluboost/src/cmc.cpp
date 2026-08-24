#include "occluboost/cmc.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
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

namespace {

cv::Mat GenerateMask(
    const cv::Size& size,
    const std::vector<Detection>& detections,
    const float scale
) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
    cv::rectangle(
        mask,
        cv::Point(static_cast<int>(0.02F * size.width), static_cast<int>(0.02F * size.height)),
        cv::Point(static_cast<int>(0.98F * size.width), static_cast<int>(0.98F * size.height)),
        cv::Scalar(255),
        cv::FILLED
    );
    for (const auto& detection : detections) {
        if (detection.is_obb) {
            const auto& box = detection.xywha;
            const cv::RotatedRect rect(
                cv::Point2f(static_cast<float>(box[0]) * scale, static_cast<float>(box[1]) * scale),
                cv::Size2f(
                    std::max(static_cast<float>(box[2]) * scale, 1.0e-4F),
                    std::max(static_cast<float>(box[3]) * scale, 1.0e-4F)
                ),
                static_cast<float>(box[4] * 180.0 / CV_PI)
            );
            std::array<cv::Point2f, 4> points{};
            rect.points(points.data());
            std::vector<cv::Point> polygon;
            polygon.reserve(points.size());
            for (const auto& point : points) {
                polygon.emplace_back(cvRound(point.x), cvRound(point.y));
            }
            cv::fillConvexPoly(mask, polygon, cv::Scalar(0));
            continue;
        }

        const auto& box = detection.xyxy;
        const int x1 = std::clamp(cvRound(box[0] * scale), 0, size.width);
        const int y1 = std::clamp(cvRound(box[1] * scale), 0, size.height);
        const int x2 = std::clamp(cvRound(box[2] * scale), 0, size.width);
        const int y2 = std::clamp(cvRound(box[3] * scale), 0, size.height);
        if (x2 > x1 && y2 > y1) {
            cv::rectangle(mask, cv::Rect(x1, y1, x2 - x1, y2 - y1), cv::Scalar(0), cv::FILLED);
        }
    }
    return mask;
}

bool IsValidSofAffine(const cv::Mat& affine, const cv::Mat& inliers, const std::size_t match_count) {
    constexpr int kMinInliers = 8;
    constexpr double kMinInlierRatio = 0.2;
    constexpr double kMinAbsDeterminant = 1.0e-4;
    constexpr double kMaxAbsDeterminant = 1.0e4;

    if (affine.rows != 2 || affine.cols != 3 || inliers.empty() || match_count == 0) {
        return false;
    }
    const int inlier_count = cv::countNonZero(inliers);
    if (inlier_count < kMinInliers || static_cast<double>(inlier_count) / match_count < kMinInlierRatio) {
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
        && abs_determinant >= kMinAbsDeterminant
        && abs_determinant <= kMaxAbsDeterminant;
}

class EccCmc final : public CameraMotionCompensator {
public:
    cv::Mat Apply(const cv::Mat& image, const std::vector<Detection>& detections) override {
        cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
        if (image.empty()) {
            return warp;
        }

        cv::Mat current = Preprocess(image, true, 0.15F);
        if (prev_image_.empty()) {
            prev_image_ = current;
            return warp;
        }

        try {
            const cv::Mat mask = GenerateMask(current.size(), detections, 0.15F);
            // 30 iterations are ample for translation-only ECC at 0.15x scale; the
            // strict 1e-5 eps in the Python defaults is essentially never satisfied,
            // so capping iterations is the dominant speed win and the resulting warp
            // differs by sub-pixel amounts from the iter=100 result on MOT footage.
            cv::findTransformECC(
                prev_image_,
                current,
                warp,
                cv::MOTION_TRANSLATION,
                cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 1.0e-5),
                mask,
                1
            );
            warp.at<float>(0, 2) /= 0.15F;
            warp.at<float>(1, 2) /= 0.15F;
        } catch (const cv::Exception&) {
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
        if (prev_frame_.empty() || prev_keypoints_.empty()) {
            Refresh(current, detections);
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

        if (prev_valid.size() >= 4 && next_valid.size() >= 4) {
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
                warp.at<float>(0, 2) /= 0.15F;
                warp.at<float>(1, 2) /= 0.15F;
            }
        }

        Refresh(current, detections);
        return warp;
    }

private:
    void Refresh(const cv::Mat& current, const std::vector<Detection>& detections) {
        prev_keypoints_.clear();
        const cv::Mat mask = GenerateMask(current.size(), detections, 0.15F);
        cv::goodFeaturesToTrack(current, prev_keypoints_, 1000, 0.01, 1.0, mask);
        prev_frame_ = current;
    }

    cv::Mat prev_frame_;
    std::vector<cv::Point2f> prev_keypoints_;
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
