#include "occluboost/track.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
    occluboost::Detection detection;
    detection.is_obb = true;
    detection.xywha << 120.0, 75.0, 50.0, 18.0, 0.35;
    detection.conf = 0.95F;
    occluboost::KalmanBoxTracker tracker(detection, 5);

    occluboost::KalmanFilterXYHR::Vector mean(10);
    mean << 120.0, 75.0, 18.0, 50.0 / 18.0, 0.35, 2.0, -1.0, 0.8, -0.4, 0.05;
    tracker.kf.mutable_mean() = mean;

    Eigen::Matrix<double, 10, 10> lower = Eigen::Matrix<double, 10, 10>::Zero();
    for (int row = 0; row < 10; ++row) {
        lower(row, row) = 1.0 + (0.2 * static_cast<double>(row));
        for (int column = 0; column < row; ++column) {
            lower(row, column) = 0.003 * static_cast<double>((row + 1) * (column + 1));
        }
    }
    tracker.kf.mutable_covariance() = lower * lower.transpose();

    constexpr double scale = 1.08;
    constexpr double angle = 0.17;
    Eigen::Matrix2d linear;
    linear << scale * std::cos(angle), -scale * std::sin(angle),
        scale * std::sin(angle), scale * std::cos(angle);
    const Eigen::Vector2d translation(7.0, -4.0);
    tracker.CameraUpdate(linear, translation);

    std::cout << std::setprecision(17);
    for (int index = 0; index < tracker.kf.mean().size(); ++index) {
        std::cout << tracker.kf.mean()[index] << ' ';
    }
    for (int row = 0; row < tracker.kf.covariance().rows(); ++row) {
        for (int column = 0; column < tracker.kf.covariance().cols(); ++column) {
            std::cout << tracker.kf.covariance()(row, column) << ' ';
        }
    }
    std::cout << '\n';
    return 0;
}
