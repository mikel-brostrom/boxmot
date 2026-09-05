#pragma once

#include "botsort/kalman_filter.hpp"
#include "botsort/types.hpp"

#include <memory>
#include <unordered_map>

namespace botsort {

class Track {
public:
    using Ptr = std::shared_ptr<Track>;

    explicit Track(const Detection& detection);

    void Activate(const KalmanFilterXYWH& kalman_filter,
                  int frame_id,
                  std::int64_t track_id);
    void ReActivate(const Track& new_track,
                    const KalmanFilterXYWH& kalman_filter,
                    int frame_id);
    void Update(const Track& new_track, const KalmanFilterXYWH& kalman_filter, int frame_id);
    void Predict(const KalmanFilterXYWH& kalman_filter);
    void ApplyAffine(const Eigen::Matrix2d& linear, const Eigen::Vector2d& translation);

    void MarkLost() { state = TrackState::kLost; }
    void MarkRemoved() { state = TrackState::kRemoved; }

    Eigen::Vector4d xyxy() const;
    Eigen::Vector4d xywh() const;
    Eigen::Matrix<double, 5, 1> xywha() const;
    bool UsesObb() const;

    bool HasSmoothFeature() const { return smooth_feat_.size() > 0; }
    const Eigen::VectorXf& smooth_feat() const { return smooth_feat_; }
    const Eigen::VectorXf& curr_feat() const { return curr_feat_; }

    bool is_activated = false;
    TrackState state = TrackState::kNew;
    std::int64_t id = 0;
    int frame_id = 0;
    int start_frame = 0;
    int tracklet_len = 0;
    float conf = 0.0F;
    std::int64_t cls = 0;
    std::int64_t det_ind = -1;

    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;

private:
    void UpdateFeatures(const Eigen::VectorXf& feat);
    void UpdateClass(std::int64_t cls_id, float confidence);
    static Eigen::VectorXf Normalize(const Eigen::VectorXf& feat);
    Eigen::VectorXd Measurement() const;

    bool is_obb_ = false;
    Eigen::Vector4d xywh_ = Eigen::Vector4d::Zero();
    Eigen::Matrix<double, 5, 1> xywha_ = Eigen::Matrix<double, 5, 1>::Zero();
    Eigen::VectorXf smooth_feat_;
    Eigen::VectorXf curr_feat_;
    std::unordered_map<std::int64_t, float> cls_hist_;
    float alpha_ = 0.9F;
};

}  // namespace botsort
