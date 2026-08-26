#include "botsort/cmc.hpp"
#include "botsort/track.hpp"
#include "botsort/types.hpp"

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace {

class CmcMaskProbe final : public botsort::CameraMotionCompensator {
public:
    using botsort::CameraMotionCompensator::GenerateMask;

    cv::Mat Apply(const cv::Mat&, const std::vector<botsort::Detection>&) override {
        return {};
    }
};

}  // namespace

int main(const int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "affine-validity") {
        cv::Mat valid = cv::Mat::eye(2, 3, CV_64F);
        cv::Mat nan_translation = valid.clone();
        nan_translation.at<double>(0, 2) = std::numeric_limits<double>::quiet_NaN();
        cv::Mat inf_linear = valid.clone();
        inf_linear.at<double>(1, 1) = std::numeric_limits<double>::infinity();
        cv::Mat singular = valid.clone();
        singular.at<double>(1, 1) = 0.0;
        std::cout << botsort::detail::IsValidAffine(valid) << ' '
                  << botsort::detail::IsValidAffine(nan_translation) << ' '
                  << botsort::detail::IsValidAffine(inf_linear) << ' '
                  << botsort::detail::IsValidAffine(singular) << '\n';
        return 0;
    }

    if (argc > 1 && std::string(argv[1]) == "mask") {
        const cv::Mat mask = CmcMaskProbe::GenerateMask(cv::Size(100, 100), {}, 1.0, 1.0);
        std::cout << cv::countNonZero(mask) << ' '
                  << static_cast<int>(mask.at<unsigned char>(2, 2)) << ' '
                  << static_cast<int>(mask.at<unsigned char>(97, 97)) << ' '
                  << static_cast<int>(mask.at<unsigned char>(98, 97)) << ' '
                  << static_cast<int>(mask.at<unsigned char>(97, 98)) << '\n';
        return 0;
    }

    if (argc > 1 && std::string(argv[1]) == "aabb") {
        botsort::Detection detection;
        detection.is_obb = false;
        detection.xyxy << 80.0, 60.0, 160.0, 90.0;

        botsort::Track track(detection);
        track.mean.resize(8);
        track.mean << 120.0, 75.0, 80.0, 30.0, 2.0, -1.0, 0.8, -0.4;

        Eigen::Matrix<double, 8, 8> lower = Eigen::Matrix<double, 8, 8>::Identity();
        lower(1, 0) = 0.20;
        lower(2, 0) = -0.15;
        lower(5, 1) = 0.30;
        lower(6, 3) = -0.25;
        track.covariance = lower * lower.transpose();

        Eigen::Matrix2d linear;
        linear << 1.10, 0.25, -0.12, 0.85;
        track.ApplyAffine(linear, Eigen::Vector2d(7.0, -4.0));

        std::cout << std::setprecision(17);
        for (int index = 0; index < track.mean.size(); ++index) {
            std::cout << track.mean[index] << ' ';
        }
        for (int row = 0; row < track.covariance.rows(); ++row) {
            for (int column = 0; column < track.covariance.cols(); ++column) {
                std::cout << track.covariance(row, column) << ' ';
            }
        }
        std::cout << '\n';
        return 0;
    }

    botsort::Detection detection;
    detection.is_obb = true;
    detection.xywha << 120.0, 75.0, 50.0, 18.0, 0.35;

    botsort::Track track(detection);
    track.mean.resize(10);
    track.mean << 120.0, 75.0, 50.0, 18.0, 0.35, 2.0, -1.0, 0.8, -0.4, 0.05;

    Eigen::Matrix<double, 10, 10> lower = Eigen::Matrix<double, 10, 10>::Identity();
    lower(1, 0) = 0.20;
    lower(2, 0) = -0.15;
    lower(4, 2) = 0.12;
    lower(6, 1) = 0.30;
    lower(7, 3) = -0.25;
    lower(9, 4) = 0.40;
    track.covariance = lower * lower.transpose();

    Eigen::Matrix2d linear;
    if (argc > 1 && std::string(argv[1]) == "similarity") {
        constexpr double kScale = 1.08;
        constexpr double kAngle = 0.17;
        linear <<
            kScale * std::cos(kAngle), -kScale * std::sin(kAngle),
            kScale * std::sin(kAngle), kScale * std::cos(kAngle);
    } else {
        linear << 1.10, 0.25, -0.12, 0.85;
    }
    const Eigen::Vector2d translation(7.0, -4.0);
    track.ApplyAffine(linear, translation);

    std::cout << std::setprecision(17);
    for (int index = 0; index < track.mean.size(); ++index) {
        std::cout << track.mean[index] << ' ';
    }
    for (int row = 0; row < track.covariance.rows(); ++row) {
        for (int column = 0; column < track.covariance.cols(); ++column) {
            std::cout << track.covariance(row, column) << ' ';
        }
    }
    std::cout << '\n';
    return 0;
}
