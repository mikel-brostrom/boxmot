#pragma once

#include "bytetrack/kalman_filter.hpp"
#include "bytetrack/types.hpp"

#include <memory>

namespace bytetrack {

class Track {
public:
    using Ptr = std::shared_ptr<Track>;

    explicit Track(const Detection& detection);

    static void ResetCount();
    static int NextId();

    void Activate(const KalmanFilterXYAH& kalman_filter, int frame_id);
    void Activate(const KalmanFilterXYWH& kalman_filter, int frame_id);
    void ReActivate(const Track& new_track, const KalmanFilterXYAH& kalman_filter, int frame_id, bool new_id = false);
    void ReActivate(const Track& new_track, const KalmanFilterXYWH& kalman_filter, int frame_id, bool new_id = false);
    void Update(const Track& new_track, const KalmanFilterXYAH& kalman_filter, int frame_id);
    void Update(const Track& new_track, const KalmanFilterXYWH& kalman_filter, int frame_id);
    void Predict(const KalmanFilterXYAH& kalman_filter);
    void Predict(const KalmanFilterXYWH& kalman_filter);

    Eigen::Vector4d xyxy() const;
    Eigen::Vector4d xyah() const;
    Eigen::Matrix<double, 5, 1> xywha() const;
    bool UsesObb() const;

    bool is_activated = false;
    TrackState state = TrackState::kNew;
    int id = 0;
    int frame_id = 0;
    int start_frame = 0;
    int tracklet_len = 0;
    float conf = 0.0F;
    int cls = 0;
    int det_ind = -1;

    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;

private:
    bool is_obb_ = false;
    Eigen::Vector4d xywh_ = Eigen::Vector4d::Zero();
    Eigen::Vector4d xyah_ = Eigen::Vector4d::Zero();
    Eigen::Matrix<double, 5, 1> xywha_ = Eigen::Matrix<double, 5, 1>::Zero();

    static int count_;
};

}  // namespace bytetrack
