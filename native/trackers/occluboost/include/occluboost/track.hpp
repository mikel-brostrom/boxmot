#pragma once

#include "occluboost/kalman_filter.hpp"
#include "occluboost/types.hpp"

#include <Eigen/Dense>

#include <deque>
#include <memory>

namespace occluboost {

// Per-track Kalman bookkeeping that mirrors boosttrack.KalmanBoxTracker plus
// OccluBoost's BotSort-style activation gate and OccluTrack AMS state.
class KalmanBoxTracker {
public:
    using Ptr = std::shared_ptr<KalmanBoxTracker>;

    KalmanBoxTracker(const Detection& detection, int max_obs);

    static void ResetCount();
    static int NextId();

    // Advance the predict step: bumps age + time_since_update; resets
    // hit_streak if a frame was missed; returns the predicted [x1,y1,x2,y2].
    Eigen::Vector4d Predict();

    // Update with a new detection (no AMS damping).
    void Update(const Detection& detection);

    // OccluBoost AMS-aware update wrapper used by all matching passes.
    // alpha=1.0 reproduces a standard update.
    void UpdateWithAlpha(const Detection& detection, double alpha);

    // Apply 2x3 affine warp from camera-motion compensation in image space.
    void CameraUpdate(const Eigen::Matrix2d& linear, const Eigen::Vector2d& translation);

    // Embedding maintenance.
    void UpdateEmbedding(const Eigen::VectorXf& emb, double alpha);
    [[nodiscard]] bool HasEmbedding() const { return embedding_.size() > 0; }
    [[nodiscard]] const Eigen::VectorXf& embedding() const { return embedding_; }

    // Convert internal state to bbox forms.
    [[nodiscard]] Eigen::Vector4d xyxy() const;  // [x1, y1, x2, y2]
    [[nodiscard]] Eigen::Vector4d xywh() const;  // [cx, cy, w, h]

    // BoostTrack track-confidence decay (coef^(missed) heuristic).
    [[nodiscard]] double GetConfidence(double coef = 0.9) const;

    // OccluBoost AMS suppression buffer accessors.
    std::deque<Eigen::Vector4d>& ams_buffer() { return ams_buffer_; }
    const std::deque<Eigen::Vector4d>& ams_buffer() const { return ams_buffer_; }

    int id = 0;
    int age = 0;
    int hit_streak = 0;
    int time_since_update = 0;
    bool is_activated = false;
    float conf = 0.0F;
    int cls = 0;
    int det_ind = -1;

    KalmanFilterXYHR kf;

private:
    static int count_;
    int max_obs_;
    Eigen::VectorXf embedding_;
    std::deque<Eigen::Vector4d> ams_buffer_;
};

// Helpers shared with the tracker.
Eigen::Vector4d XyxyToZ(const Eigen::Vector4d& xyxy);  // [x, y, h, r]
Eigen::Vector4d ZToXyxy(const Eigen::Vector4d& z);     // [x1, y1, x2, y2]
Eigen::Vector4d XyxyToCxcywh(const Eigen::Vector4d& xyxy);

}  // namespace occluboost
