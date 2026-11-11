#include "botsort/kalman_filter.hpp"
#include "bytetrack/kalman_filter.hpp"
#include "occluboost/kalman_filter.hpp"
#include "ocsort/kalman_filter.hpp"
#include "sfsort/tracker.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void Require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Filter>
void CheckStateless(const Filter& filter, const Eigen::VectorXd& measurement) {
    auto [mean, covariance] = filter.Initiate(measurement);
    mean[measurement.size()] = 2.0;
    const auto [short_mean, short_covariance] = filter.Predict(mean, covariance, 0.4);
    const auto [split_mean, split_covariance] = filter.Predict(short_mean, short_covariance, 0.6);
    const auto [full_mean, full_covariance] = filter.Predict(mean, covariance, 1.0);
    const auto [long_mean, long_covariance] = filter.Predict(mean, covariance, 3.0);
    Require(std::abs(long_mean[0] - full_mean[0] - 4.0) < 1.0e-9, "Elapsed transition ignored velocity");
    Require(long_covariance(0, 0) > full_covariance(0, 0), "Long gap failed to increase uncertainty");
    Require(split_mean.isApprox(full_mean, 1.0e-10), "Split transitions do not compose");
    Require(split_covariance.isApprox(full_covariance, 1.0e-10), "Integrated process noise does not compose");
    const auto [fixed_mean, fixed_covariance] = filter.Predict(mean, covariance);
    Require(fixed_mean.isApprox(full_mean), "Fixed and unit elapsed transition differ");
    Require(!fixed_covariance.isApprox(full_covariance, 1.0e-10), "Timed process noise reused discrete Q");
}

void CheckOccluBoost(const int dimensions) {
    Eigen::VectorXd measurement(dimensions);
    measurement.head(4) << 20.0, 30.0, 40.0, 0.5;
    if (dimensions == 5) {
        measurement[4] = 0.2;
    }
    occluboost::KalmanFilterXYHR filter;
    filter.Initiate(measurement);
    filter.mutable_mean()[dimensions] = 2.0;
    auto split = filter;
    auto full = filter;
    auto longer = filter;
    split.Predict(0.4);
    split.Predict(0.6);
    full.Predict(1.0);
    longer.Predict(3.0);
    Require(split.mean().isApprox(full.mean()), "OccluBoost transition does not compose");
    Require(split.covariance().isApprox(full.covariance()), "OccluBoost noise does not compose");
    Require(std::abs(longer.mean()[0] - full.mean()[0] - 4.0) < 1.0e-9, "OccluBoost ignored elapsed velocity");
    Require(longer.covariance()(0, 0) > full.covariance()(0, 0), "OccluBoost uncertainty did not grow");
}

void CheckOcSort(const bool obb) {
    ocsort::KalmanFilterXYSR filter(obb ? 9 : 7, obb ? 5 : 4);
    Eigen::VectorXd origin(obb ? 5 : 4);
    origin.head(4) << 20.0, 30.0, 800.0, 0.5;
    if (obb) {
        origin[4] = 0.2;
    }
    filter.x.head(origin.size()) = origin;
    filter.x[obb ? 5 : 4] = 2.0;
    auto split = filter;
    auto full = filter;
    split.Predict(0.4);
    split.Predict(0.6);
    full.Predict(1.0);
    Require(split.x.isApprox(full.x), "OC-SORT transition does not compose");
    Require(split.P.isApprox(full.P), "OC-SORT noise does not compose");

    // A missed-observation replay must produce the same posterior as explicit
    // virtual detections placed at the actual (unequal) capture intervals.
    auto gap = filter;
    auto observed = filter;
    const double intervals[] = {0.2, 0.5, 0.9};
    double elapsed = 0.0;
    for (const double dt : intervals) {
        elapsed += dt;
        const double fraction = elapsed / 1.6;
        Eigen::VectorXd measurement = origin;
        measurement[0] += 8.0 * fraction;
        measurement[1] += 3.0 * fraction;
        const double width = 20.0 + 2.0 * fraction;
        const double height = 40.0 + 4.0 * fraction;
        measurement[2] = width * height;
        measurement[3] = width / height;
        if (obb) {
            measurement[4] += 0.1 * fraction;
        }
        observed.Predict(dt);
        observed.Update(measurement);
        gap.Predict(dt);
        if (fraction < 0.999) {
            gap.UpdateMissing();
        } else {
            gap.Update(measurement);
        }
    }
    Require(gap.x.isApprox(observed.x, 1.0e-9), "OC-SORT recovery ignored elapsed interpolation");
    Require(gap.P.isApprox(observed.P, 1.0e-9), "OC-SORT recovery replayed wrong process covariance");
}

void CheckSfsort() {
    sfsort::SFSORTTracker::TrackData original;
    original.is_obb = true;
    original.xywha << 20.0, 30.0, 40.0, 20.0, 0.0;
    sfsort::Detection detection;
    detection.is_obb = true;
    detection.xywha << 20.0, 30.0, 40.0, 20.0, 0.4;
    original.Update(detection, 2, 0.8F);
    Require(std::abs(original.xywha[4] - 0.08) < 1.0e-7, "SFSORT fixed-step angular smoothing changed");
    original.Update(detection, 3, 0.8F);
    Require(std::abs(original.xywha[4] - 0.208) < 1.0e-7, "SFSORT fixed-step angular velocity changed");

    sfsort::SFSORTTracker tracker(sfsort::Config{});
    bool rejected = false;
    try {
        tracker.Update({}, {}, 0.5);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(rejected, "SFSORT accepted variable dt");
}

}  // namespace

int main() {
    try {
        Eigen::Vector4d xyah(20.0, 30.0, 0.5, 40.0);
        CheckStateless(bytetrack::KalmanFilterXYAH{}, xyah);
        for (const int dimensions : {4, 5}) {
            Eigen::VectorXd xywh(dimensions);
            xywh.head(4) << 20.0, 30.0, 20.0, 40.0;
            if (dimensions == 5) {
                xywh[4] = 0.2;
            }
            CheckStateless(bytetrack::KalmanFilterXYWH{dimensions}, xywh);
            CheckStateless(botsort::KalmanFilterXYWH{dimensions}, xywh);
            CheckOccluBoost(dimensions);
            CheckOcSort(dimensions == 5);
        }
        CheckSfsort();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
