#include "botsort/cmc.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <stdexcept>

namespace botsort {

cv::Mat CameraMotionCompensator::Preprocess(const cv::Mat& image, const bool grayscale, const float scale) const {
    cv::Mat output = image;
    if (grayscale && image.channels() == 3) {
        cv::cvtColor(image, output, cv::COLOR_BGR2GRAY);
    }
    if (scale > 0.0F && scale != 1.0F) {
        cv::resize(output, output, cv::Size(), scale, scale, cv::INTER_LINEAR);
    }
    return output;
}

namespace {

class EccCmc final : public CameraMotionCompensator {
public:
    cv::Mat Apply(const cv::Mat& image, const std::vector<Detection>&) override {
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
            cv::findTransformECC(
                prev_image_,
                current,
                warp,
                cv::MOTION_TRANSLATION,
                cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 100, 1.0e-5),
                cv::noArray(),
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
    cv::Mat Apply(const cv::Mat& image, const std::vector<Detection>&) override {
        cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
        if (image.empty()) {
            return warp;
        }

        cv::Mat current = Preprocess(image, true, 0.15F);
        if (prev_frame_.empty() || prev_keypoints_.empty()) {
            Refresh(current);
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
        for (std::size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                prev_valid.push_back(prev_keypoints_[i]);
                next_valid.push_back(next_keypoints[i]);
            }
        }

        if (prev_valid.size() >= 4 && next_valid.size() >= 4) {
            cv::Mat affine = cv::estimateAffinePartial2D(prev_valid, next_valid);
            if (!affine.empty()) {
                affine.convertTo(warp, CV_32F);
                warp.at<float>(0, 2) /= 0.15F;
                warp.at<float>(1, 2) /= 0.15F;
            }
        }

        Refresh(current);
        return warp;
    }

private:
    void Refresh(const cv::Mat& current) {
        prev_keypoints_.clear();
        cv::goodFeaturesToTrack(current, prev_keypoints_, 1000, 0.01, 1.0);
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

}  // namespace botsort
