#include "bytetrack/track.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>

namespace bytetrack {

int Track::count_ = 0;

namespace {

constexpr double kPi = 3.14159265358979323846;

double WrapAngle(const double angle) {
    const double period = 2.0 * kPi;
    return std::fmod(std::fmod(angle + kPi, period) + period, period) - kPi;
}

Eigen::Vector4d XyxyToXywh(const Eigen::Vector4d& xyxy) {
    const double width = std::max(xyxy[2] - xyxy[0], 1.0e-4);
    const double height = std::max(xyxy[3] - xyxy[1], 1.0e-4);
    return Eigen::Vector4d(
        xyxy[0] + (width / 2.0),
        xyxy[1] + (height / 2.0),
        width,
        height
    );
}

Eigen::Vector4d XywhToXyah(const Eigen::Vector4d& xywh) {
    const double width = std::max(xywh[2], 1.0e-4);
    const double height = std::max(xywh[3], 1.0e-4);
    return Eigen::Vector4d(xywh[0], xywh[1], width / height, height);
}

Eigen::Vector4d XyahToXyxy(const Eigen::Vector4d& xyah) {
    const double height = std::max(xyah[3], 1.0e-4);
    const double width = std::max(xyah[2] * height, 1.0e-4);
    const double half_w = width / 2.0;
    const double half_h = height / 2.0;
    return Eigen::Vector4d(
        xyah[0] - half_w,
        xyah[1] - half_h,
        xyah[0] + half_w,
        xyah[1] + half_h
    );
}

cv::RotatedRect RotatedRectFromXywha(const Eigen::Matrix<double, 5, 1>& box) {
    return cv::RotatedRect(
        cv::Point2f(static_cast<float>(box[0]), static_cast<float>(box[1])),
        cv::Size2f(
            static_cast<float>(std::max(box[2], 1.0e-4)),
            static_cast<float>(std::max(box[3], 1.0e-4))
        ),
        static_cast<float>(box[4] * 180.0 / kPi)
    );
}

Eigen::Vector4d ObbToXyxy(const Eigen::Matrix<double, 5, 1>& box) {
    const cv::RotatedRect rect = RotatedRectFromXywha(box);
    std::array<cv::Point2f, 4> corners{};
    rect.points(corners.data());

    double x1 = std::numeric_limits<double>::infinity();
    double y1 = std::numeric_limits<double>::infinity();
    double x2 = -std::numeric_limits<double>::infinity();
    double y2 = -std::numeric_limits<double>::infinity();
    for (const auto& point : corners) {
        x1 = std::min(x1, static_cast<double>(point.x));
        y1 = std::min(y1, static_cast<double>(point.y));
        x2 = std::max(x2, static_cast<double>(point.x));
        y2 = std::max(y2, static_cast<double>(point.y));
    }

    Eigen::Vector4d xyxy;
    xyxy << x1, y1, x2, y2;
    return xyxy;
}

}  // namespace

Track::Track(const Detection& detection)
    : conf(detection.conf),
      cls(detection.cls),
      det_ind(detection.det_ind),
      is_obb_(detection.is_obb) {
    if (is_obb_) {
        xywha_ = detection.xywha;
        xywha_[2] = std::max(xywha_[2], 1.0e-4);
        xywha_[3] = std::max(xywha_[3], 1.0e-4);
        xywha_[4] = WrapAngle(xywha_[4]);
    } else {
        xywh_ = XyxyToXywh(detection.xyxy);
        xyah_ = XywhToXyah(xywh_);
    }
}

void Track::ResetCount() {
    count_ = 0;
}

int Track::NextId() {
    ++count_;
    return count_;
}

void Track::Activate(const KalmanFilterXYAH& kalman_filter, const int current_frame_id) {
    id = NextId();
    const auto [new_mean, new_covariance] = kalman_filter.Initiate(xyah_);
    mean = new_mean;
    covariance = new_covariance;
    tracklet_len = 0;
    state = TrackState::kTracked;
    is_activated = current_frame_id == 1;
    frame_id = current_frame_id;
    start_frame = current_frame_id;
}

void Track::Activate(const KalmanFilterXYWH& kalman_filter, const int current_frame_id) {
    id = NextId();
    const Eigen::VectorXd measurement = xywha_;
    const auto [new_mean, new_covariance] = kalman_filter.Initiate(measurement);
    mean = new_mean;
    covariance = new_covariance;
    tracklet_len = 0;
    state = TrackState::kTracked;
    is_activated = current_frame_id == 1;
    frame_id = current_frame_id;
    start_frame = current_frame_id;
}

void Track::ReActivate(
    const Track& new_track,
    const KalmanFilterXYAH& kalman_filter,
    const int current_frame_id,
    const bool new_id
) {
    const auto [new_mean, new_covariance] = kalman_filter.Update(mean, covariance, new_track.xyah());
    mean = new_mean;
    covariance = new_covariance;
    tracklet_len = 0;
    state = TrackState::kTracked;
    is_activated = true;
    frame_id = current_frame_id;
    if (new_id) {
        id = NextId();
    }
    conf = new_track.conf;
    cls = new_track.cls;
    det_ind = new_track.det_ind;
}

void Track::ReActivate(
    const Track& new_track,
    const KalmanFilterXYWH& kalman_filter,
    const int current_frame_id,
    const bool new_id
) {
    const Eigen::VectorXd measurement = new_track.xywha();
    const auto [new_mean, new_covariance] = kalman_filter.Update(mean, covariance, measurement);
    mean = new_mean;
    covariance = new_covariance;
    tracklet_len = 0;
    state = TrackState::kTracked;
    is_activated = true;
    frame_id = current_frame_id;
    if (new_id) {
        id = NextId();
    }
    conf = new_track.conf;
    cls = new_track.cls;
    det_ind = new_track.det_ind;
}

void Track::Update(const Track& new_track, const KalmanFilterXYAH& kalman_filter, const int current_frame_id) {
    frame_id = current_frame_id;
    ++tracklet_len;
    const auto [new_mean, new_covariance] = kalman_filter.Update(mean, covariance, new_track.xyah());
    mean = new_mean;
    covariance = new_covariance;
    state = TrackState::kTracked;
    is_activated = true;
    conf = new_track.conf;
    cls = new_track.cls;
    det_ind = new_track.det_ind;
}

void Track::Update(const Track& new_track, const KalmanFilterXYWH& kalman_filter, const int current_frame_id) {
    frame_id = current_frame_id;
    ++tracklet_len;
    const Eigen::VectorXd measurement = new_track.xywha();
    const auto [new_mean, new_covariance] = kalman_filter.Update(mean, covariance, measurement);
    mean = new_mean;
    covariance = new_covariance;
    state = TrackState::kTracked;
    is_activated = true;
    conf = new_track.conf;
    cls = new_track.cls;
    det_ind = new_track.det_ind;
}

void Track::Predict(const KalmanFilterXYAH& kalman_filter) {
    if (mean.size() == 0 || covariance.size() == 0) {
        return;
    }
    KalmanFilterXYAH::Vector mean_state = mean;
    if (state != TrackState::kTracked && mean_state.size() >= 8) {
        mean_state[7] = 0.0;
    }
    const auto [predicted_mean, predicted_covariance] = kalman_filter.Predict(mean_state, covariance);
    mean = predicted_mean;
    covariance = predicted_covariance;
}

void Track::Predict(const KalmanFilterXYWH& kalman_filter) {
    if (mean.size() == 0 || covariance.size() == 0) {
        return;
    }
    KalmanFilterXYWH::Vector mean_state = mean;
    if (state != TrackState::kTracked && mean_state.size() >= 10) {
        mean_state[7] = 0.0;
        mean_state[8] = 0.0;
        mean_state[9] = 0.0;
    }
    const auto [predicted_mean, predicted_covariance] = kalman_filter.Predict(mean_state, covariance);
    mean = predicted_mean;
    covariance = predicted_covariance;
}

Eigen::Vector4d Track::xyxy() const {
    if (is_obb_) {
        return ObbToXyxy(xywha());
    }
    return XyahToXyxy(xyah());
}

Eigen::Vector4d Track::xyah() const {
    if (is_obb_) {
        const Eigen::Matrix<double, 5, 1> box = xywha();
        const double height = std::max(box[3], 1.0e-4);
        const double width = std::max(box[2], 1.0e-4);
        return Eigen::Vector4d(box[0], box[1], width / height, height);
    }
    if (mean.size() >= 4) {
        Eigen::Vector4d result = mean.head<4>();
        result[2] = std::max(result[2], 1.0e-4);
        result[3] = std::max(result[3], 1.0e-4);
        return result;
    }
    return xyah_;
}

Eigen::Matrix<double, 5, 1> Track::xywha() const {
    if (mean.size() >= 5 && is_obb_) {
        Eigen::Matrix<double, 5, 1> result = mean.head<5>();
        result[2] = std::max(result[2], 1.0e-4);
        result[3] = std::max(result[3], 1.0e-4);
        result[4] = WrapAngle(result[4]);
        return result;
    }
    if (is_obb_) {
        return xywha_;
    }
    Eigen::Vector4d xyah_box = xyah_;
    if (mean.size() >= 4) {
        xyah_box = mean.head<4>();
    }
    xyah_box[2] = std::max(xyah_box[2], 1.0e-4);
    xyah_box[3] = std::max(xyah_box[3], 1.0e-4);
    const double width = std::max(xyah_box[2] * xyah_box[3], 1.0e-4);
    Eigen::Matrix<double, 5, 1> result;
    result << xyah_box[0], xyah_box[1], width, xyah_box[3], 0.0;
    return result;
}

bool Track::UsesObb() const {
    return is_obb_;
}

}  // namespace bytetrack
