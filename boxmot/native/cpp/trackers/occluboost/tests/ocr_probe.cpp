#include "occluboost/track.hpp"
#include "occluboost/tracker.hpp"

#include <opencv2/core.hpp>

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr double kTolerance = 1.0e-6;

occluboost::Detection MakeAabb(const double x, const int cls = 0) {
    occluboost::Detection detection;
    detection.xyxy << x, 50.0, x + 100.0, 100.0;
    detection.conf = 0.95F;
    detection.cls = cls;
    detection.embedding = Eigen::Vector2f(1.0F, 0.0F);
    return detection;
}

occluboost::Detection MakeObb(const double x, const int cls = 0) {
    occluboost::Detection detection;
    detection.is_obb = true;
    detection.xywha << x, 75.0, 100.0, 50.0, 0.2;
    detection.conf = 0.95F;
    detection.cls = cls;
    detection.embedding = Eigen::Vector2f(1.0F, 0.0F);
    return detection;
}

occluboost::Config RecoveryConfig() {
    occluboost::Config config;
    config.use_embeddings = false;
    config.cmc_method = "none";
    config.use_dlo_boost = false;
    config.use_duo_boost = false;
    config.use_second_pass = false;
    config.ams_enabled = false;
    config.det_thresh = 0.1F;
    config.iou_threshold = 0.9F;
    config.new_track_thresh = 0.1F;
    config.instant_confirm_thresh = 0.1F;
    config.obb_det_thresh = 0.1F;
    config.obb_iou_threshold = 0.9F;
    config.obb_new_track_thresh = 0.1F;
    config.obb_instant_confirm_thresh = 0.1F;
    config.confirm_hits = 1;
    config.min_hits = 0;
    config.max_age = 30;
    config.obb_max_age = 30;
    config.tentative_max_age = 30;
    config.min_box_area = 1;
    config.aspect_ratio_thresh = 20.0F;
    return config;
}

bool NearlyEqual(const double lhs, const double rhs) {
    return std::abs(lhs - rhs) <= kTolerance;
}

bool CheckObservationState() {
    occluboost::KalmanBoxTracker aabb_track(MakeAabb(0.0), 5);
    occluboost::KalmanBoxTracker obb_track(MakeObb(100.0), 5);
    if (aabb_track.HasLastObservation() || obb_track.HasLastObservation()) {
        std::cerr << "Birth detections must not be OCR-eligible observations.\n";
        return false;
    }

    occluboost::Detection aabb_update = MakeAabb(4.0);
    aabb_update.embedding = Eigen::Vector2f(0.0F, 1.0F);
    aabb_track.Update(aabb_update);
    obb_track.Update(MakeObb(104.0));
    if (!aabb_track.HasLastObservation() || !obb_track.HasLastObservation()) {
        std::cerr << "Matched real detections must become OCR-eligible observations.\n";
        return false;
    }
    if (!NearlyEqual(aabb_track.embedding()[0], 1.0) ||
        !NearlyEqual(aabb_track.embedding()[1], 0.0)) {
        std::cerr << "Geometry-only updates must not change the track embedding.\n";
        return false;
    }

    Eigen::Matrix2d aabb_rotation;
    aabb_rotation << 0.0, -1.0, 1.0, 0.0;
    aabb_track.CameraUpdate(aabb_rotation, Eigen::Vector2d(60.0, 5.0));
    const Eigen::Vector4d warped_aabb = aabb_track.last_observation_xyxy();
    const Eigen::Vector4d expected_aabb(-40.0, 9.0, 10.0, 109.0);
    if (!warped_aabb.isApprox(expected_aabb, kTolerance)) {
        std::cerr << "AABB last observation did not follow CMC.\n";
        return false;
    }

    constexpr double scale = 1.2;
    constexpr double angle = 0.3;
    Eigen::Matrix2d obb_similarity;
    obb_similarity << scale * std::cos(angle), -scale * std::sin(angle), scale * std::sin(angle),
        scale * std::cos(angle);
    const Eigen::Vector2d obb_translation(7.0, -4.0);
    const Eigen::Matrix<double, 5, 1> before_obb = obb_track.last_observation_xywha();
    obb_track.CameraUpdate(obb_similarity, obb_translation);
    const Eigen::Matrix<double, 5, 1> warped_obb = obb_track.last_observation_xywha();
    const Eigen::Vector2d expected_center =
        (obb_similarity * before_obb.head<2>()) + obb_translation;
    if (!warped_obb.head<2>().isApprox(expected_center, kTolerance) ||
        !NearlyEqual(warped_obb[2], scale * before_obb[2]) ||
        !NearlyEqual(warped_obb[3], scale * before_obb[3]) ||
        !NearlyEqual(warped_obb[4], before_obb[4] + angle)) {
        std::cerr << "OBB last observation did not follow CMC.\n";
        return false;
    }
    return true;
}

bool CheckAabbRecovery(const bool change_class) {
    occluboost::OccluBoostTracker tracker(RecoveryConfig());
    const cv::Mat image = cv::Mat::zeros(200, 240, CV_8UC3);
    const auto first = tracker.Update({MakeAabb(0.0)}, image);
    tracker.Update({MakeAabb(2.0)}, image);
    tracker.Update({MakeAabb(4.0)}, image);
    for (int frame = 0; frame < 4; ++frame) {
        tracker.Update({}, image);
    }
    const auto recovered = tracker.Update({MakeAabb(4.0, change_class ? 1 : 0)}, image);
    if (first.size() != 1 || recovered.size() != 1) {
        return false;
    }
    return change_class ? recovered.front().id != first.front().id
                        : recovered.front().id == first.front().id;
}

bool CheckObbRecovery(const bool change_class) {
    occluboost::OccluBoostTracker tracker(RecoveryConfig());
    const cv::Mat image = cv::Mat::zeros(200, 240, CV_8UC3);
    const auto first = tracker.Update({MakeObb(100.0)}, image);
    tracker.Update({MakeObb(102.0)}, image);
    tracker.Update({MakeObb(104.0)}, image);
    for (int frame = 0; frame < 4; ++frame) {
        tracker.Update({}, image);
    }
    const auto recovered = tracker.Update({MakeObb(104.0, change_class ? 1 : 0)}, image);
    if (first.size() != 1 || recovered.size() != 1) {
        return false;
    }
    return change_class ? recovered.front().id != first.front().id
                        : recovered.front().id == first.front().id;
}

}  // namespace

int main() {
    if (!CheckObservationState()) {
        return 1;
    }
    if (!CheckAabbRecovery(false) || !CheckObbRecovery(false)) {
        std::cerr << "Observation-centric recovery did not preserve the track ID.\n";
        return 2;
    }
    if (!CheckAabbRecovery(true) || !CheckObbRecovery(true)) {
        std::cerr << "Observation-centric recovery ignored the class-equality gate.\n";
        return 3;
    }
    return 0;
}
