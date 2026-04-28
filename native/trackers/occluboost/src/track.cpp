#include "occluboost/track.hpp"

#include <algorithm>
#include <cmath>

namespace occluboost {

int KalmanBoxTracker::count_ = 0;

Eigen::Vector4d XyxyToZ(const Eigen::Vector4d& xyxy) {
    const double w = xyxy[2] - xyxy[0];
    const double h = xyxy[3] - xyxy[1];
    const double x = xyxy[0] + 0.5 * w;
    const double y = xyxy[1] + 0.5 * h;
    const double r = w / (h + 1.0e-6);
    Eigen::Vector4d z;
    z << x, y, h, r;
    return z;
}

Eigen::Vector4d ZToXyxy(const Eigen::Vector4d& z) {
    const double h = z[2];
    const double r = z[3];
    const double w = r <= 0.0 ? 0.0 : r * h;
    Eigen::Vector4d xyxy;
    xyxy << z[0] - 0.5 * w, z[1] - 0.5 * h, z[0] + 0.5 * w, z[1] + 0.5 * h;
    return xyxy;
}

Eigen::Vector4d XyxyToCxcywh(const Eigen::Vector4d& xyxy) {
    const double w = std::max(xyxy[2] - xyxy[0], 1.0e-6);
    const double h = std::max(xyxy[3] - xyxy[1], 1.0e-6);
    Eigen::Vector4d out;
    out << xyxy[0] + 0.5 * w, xyxy[1] + 0.5 * h, w, h;
    return out;
}

void KalmanBoxTracker::ResetCount() {
    count_ = 0;
}

int KalmanBoxTracker::NextId() {
    ++count_;
    return count_;
}

KalmanBoxTracker::KalmanBoxTracker(const Detection& detection, const int max_obs)
    : conf(detection.conf),
      cls(detection.cls),
      det_ind(detection.det_ind),
      max_obs_(std::max(max_obs, 1)) {
    id = NextId();
    KalmanFilterXYHR::Vector measurement = XyxyToZ(detection.xyxy);
    kf.Initiate(measurement);
    if (detection.has_embedding()) {
        const float norm = detection.embedding.norm();
        embedding_ = norm > 1.0e-12F ? (detection.embedding / norm).eval() : detection.embedding;
    }
}

Eigen::Vector4d KalmanBoxTracker::Predict() {
    kf.Predict();
    age += 1;
    if (time_since_update > 0) {
        hit_streak = 0;
    }
    time_since_update += 1;
    return xyxy();
}

void KalmanBoxTracker::Update(const Detection& detection) {
    UpdateWithAlpha(detection, 1.0);
}

void KalmanBoxTracker::UpdateWithAlpha(const Detection& detection, const double alpha) {
    time_since_update = 0;
    hit_streak += 1;
    // Mirror Python: history_observations gets the *pre-update* state.
    // (We don't expose history_observations because the tracker doesn't use it
    // outside plotting. Skip the bookkeeping but advance KF state correctly.)
    KalmanFilterXYHR::Vector z = XyxyToZ(detection.xyxy);
    kf.Update(z, alpha);
    conf = detection.conf;
    cls = detection.cls;
    det_ind = detection.det_ind;
}

void KalmanBoxTracker::CameraUpdate(
    const Eigen::Matrix2d& linear,
    const Eigen::Vector2d& translation
) {
    const Eigen::Vector4d box = xyxy();
    Eigen::Vector2d p1 = linear * Eigen::Vector2d(box[0], box[1]) + translation;
    Eigen::Vector2d p2 = linear * Eigen::Vector2d(box[2], box[3]) + translation;
    const double w = p2[0] - p1[0];
    const double h = p2[1] - p1[1];
    KalmanFilterXYHR::Vector& mean = kf.mutable_mean();
    mean[0] = p1[0] + 0.5 * w;
    mean[1] = p1[1] + 0.5 * h;
    mean[2] = std::max(h, 1.0e-4);
    mean[3] = h > 0.0 ? w / h : 0.0;
}

void KalmanBoxTracker::UpdateEmbedding(const Eigen::VectorXf& emb, const double alpha) {
    if (emb.size() == 0) {
        return;
    }
    if (embedding_.size() == 0) {
        const float norm = emb.norm();
        embedding_ = norm > 1.0e-12F ? (emb / norm).eval() : emb;
        return;
    }
    if (embedding_.size() != emb.size()) {
        return;
    }
    embedding_ = static_cast<float>(alpha) * embedding_ + static_cast<float>(1.0 - alpha) * emb;
    const float norm = embedding_.norm();
    if (norm > 1.0e-12F) {
        embedding_ /= norm;
    }
}

Eigen::Vector4d KalmanBoxTracker::xyxy() const {
    Eigen::Vector4d state;
    state << kf.mean()[0], kf.mean()[1], kf.mean()[2], kf.mean()[3];
    return ZToXyxy(state);
}

Eigen::Vector4d KalmanBoxTracker::xywh() const {
    return XyxyToCxcywh(xyxy());
}

double KalmanBoxTracker::GetConfidence(const double coef) const {
    constexpr int n = 7;
    if (age < n) {
        return std::pow(coef, n - age);
    }
    return std::pow(coef, std::max(0, time_since_update - 1));
}

}  // namespace occluboost
