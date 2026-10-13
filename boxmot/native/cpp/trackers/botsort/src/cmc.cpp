#include "botsort/cmc.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace botsort {

cv::Mat CameraMotionCompensator::Preprocess(const cv::Mat& image, const bool grayscale, const int target_width) const {
    cv::Mat output = image;
    // Bound estimator work independently of capture resolution, and convert
    // only the pixels retained by downsampling. Never upscale small inputs.
    if (image.cols > target_width) {
        const double scale = static_cast<double>(target_width) / image.cols;
        const cv::Size size(target_width, std::max(1, cvRound(image.rows * scale)));
        cv::resize(output, output, size, 0.0, 0.0, cv::INTER_LINEAR);
    }
    if (grayscale && output.channels() == 3) {
        cv::cvtColor(output, output, cv::COLOR_BGR2GRAY);
    } else if (grayscale && output.channels() == 4) {
        cv::cvtColor(output, output, cv::COLOR_BGRA2GRAY);
    }
    // Small grayscale inputs bypass both allocation paths. Own their history
    // so a caller can refill the same capture buffer for the next frame.
    if (output.data == image.data) {
        output = output.clone();
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
    const int x1 = static_cast<int>(0.02 * static_cast<double>(size.width));
    const int y1 = static_cast<int>(0.02 * static_cast<double>(size.height));
    const int x2 = static_cast<int>(0.98 * static_cast<double>(size.width));
    const int y2 = static_cast<int>(0.98 * static_cast<double>(size.height));
    if (x2 > x1 && y2 > y1) {
        // Python assigns mask[y1:y2, x1:x2]. A rectangle built from two
        // cv::Point endpoints includes x2/y2 and changes SOF keypoint
        // selection at the lower/right border.
        cv::rectangle(mask, cv::Rect(x1, y1, x2 - x1, y2 - y1), cv::Scalar(255), cv::FILLED);
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
        // Match NumPy astype(int), which truncates rather than rounds.
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

bool detail::IsValidAffine(
    const cv::Mat& affine,
    const double min_abs_determinant,
    const double max_abs_determinant
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
    const cv::Mat scale = (cv::Mat_<double>(3, 3) << scale_x, 0.0, 0.0, 0.0, scale_y, 0.0, 0.0, 0.0, 1.0);
    cv::Mat restored = scale.inv() * homogeneous * scale;
    cv::Mat result;
    restored(cv::Rect(0, 0, 3, 2)).convertTo(result, CV_32F);
    return result;
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
        && detail::IsValidAffine(affine);
}

class EccCmc final : public CameraMotionCompensator {
public:
    cv::Mat Apply(const cv::Mat& image, const std::vector<Detection>& detections) override {
        cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
        if (image.empty()) {
            return warp;
        }

        cv::Mat current = Preprocess(image, true, 96);
        const auto [scale_x, scale_y] = ResizeScale(image, current);
        if (prev_image_.empty() || prev_input_size_ != image.size()) {
            prev_image_ = current;
            prev_input_size_ = image.size();
            return warp;
        }

        try {
            const cv::Mat mask = GenerateMask(current.size(), detections, scale_x, scale_y);
            const double correlation = cv::findTransformECC(
                prev_image_,
                current,
                warp,
                cv::MOTION_TRANSLATION,
                cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1.0e-3),
                mask,
                3
            );
            if (!std::isfinite(correlation) || correlation < 0.5 || !detail::IsValidAffine(warp)) {
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
    cv::Size prev_input_size_;
};

class SofCmc final : public CameraMotionCompensator {
public:
    cv::Mat Apply(const cv::Mat& image, const std::vector<Detection>& detections) override {
        cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
        if (image.empty()) {
            return warp;
        }

        cv::Mat current = Preprocess(image, true, 192);
        const auto [scale_x, scale_y] = ResizeScale(image, current);
        if (!initialized_ || prev_frame_.empty() || prev_keypoints_.empty() || prev_input_size_ != image.size()) {
            prev_input_size_ = image.size();
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
            cv::Size(11, 11),
            2,
            cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 0.03)
        );

        if (next_keypoints.empty() || status.empty()) {
            Reset(current, detections, scale_x, scale_y);
            return warp;
        }

        std::vector<cv::Point2f> prev_valid;
        std::vector<cv::Point2f> next_valid;
        const cv::Mat mask = GenerateMask(current.size(), detections, scale_x, scale_y);
        const std::size_t point_count = std::min({status.size(), prev_keypoints_.size(), next_keypoints.size()});
        prev_valid.reserve(point_count);
        next_valid.reserve(point_count);
        for (std::size_t i = 0; i < point_count; ++i) {
            if (
                status[i]
                && std::isfinite(prev_keypoints_[i].x)
                && std::isfinite(prev_keypoints_[i].y)
                && std::isfinite(next_keypoints[i].x)
                && std::isfinite(next_keypoints[i].y)
            ) {
                // Points reused between detector refreshes must remain on the
                // current background, outside moving detections and borders.
                const cv::Point point(cvRound(next_keypoints[i].x), cvRound(next_keypoints[i].y));
                if (
                    point.x >= 0 && point.x < mask.cols
                    && point.y >= 0 && point.y < mask.rows
                    && mask.at<unsigned char>(point) != 0
                ) {
                    prev_valid.push_back(prev_keypoints_[i]);
                    next_valid.push_back(next_keypoints[i]);
                }
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
        const bool accepted = IsValidSofAffine(affine, inliers, prev_valid.size());
        if (accepted) {
            affine.convertTo(warp, CV_32F);
            warp = RestoreAffineScale(warp, scale_x, scale_y);
        }

        ++frames_since_detection_;
        std::vector<cv::Point2f> refreshed;
        if (accepted) {
            refreshed.reserve(next_valid.size());
            for (std::size_t index = 0; index < next_valid.size(); ++index) {
                if (inliers.at<unsigned char>(static_cast<int>(index)) != 0) {
                    refreshed.push_back(next_valid[index]);
                }
            }
        }
        // Retain subpixel inlier locations, refreshing at most every five
        // frames or as soon as fewer than half the feature budget survives.
        if (!accepted || refreshed.size() < 24U || frames_since_detection_ >= 5) {
            cv::goodFeaturesToTrack(current, refreshed, 48, 0.01, 4.0, mask, 3, false, 0.04);
            frames_since_detection_ = 0;
        }
        prev_keypoints_ = std::move(refreshed);
        prev_frame_ = current;
        initialized_ = prev_keypoints_.size() >= 4U;
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
        cv::goodFeaturesToTrack(current, keypoints, 48, 0.01, 4.0, mask, 3, false, 0.04);
        return keypoints;
    }

    void Initialize(
        const cv::Mat& current,
        const std::vector<Detection>& detections,
        const double scale_x,
        const double scale_y
    ) {
        std::vector<cv::Point2f> keypoints = DetectKeypoints(current, detections, scale_x, scale_y);
        prev_frame_ = current;
        prev_keypoints_ = std::move(keypoints);
        if (prev_keypoints_.size() < 4U) {
            initialized_ = false;
            return;
        }
        // Subpixel refinement requires at least 2 * window + 5 pixels.
        if (prev_frame_.rows >= 15 && prev_frame_.cols >= 15) {
            cv::cornerSubPix(
                prev_frame_,
                prev_keypoints_,
                cv::Size(5, 5),
                cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 0.03)
            );
        }
        initialized_ = true;
        frames_since_detection_ = 0;
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
        frames_since_detection_ = 0;
    }

    cv::Mat prev_frame_;
    cv::Size prev_input_size_;
    std::vector<cv::Point2f> prev_keypoints_;
    bool initialized_ = false;
    int frames_since_detection_ = 0;
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

}  // namespace botsort
