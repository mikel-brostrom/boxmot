#include "occluboost/track.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace occluboost {

int KalmanBoxTracker::count_ = 0;

namespace {

constexpr double kPi = 3.14159265358979323846;

double WrapAngle(const double angle) {
    const double period = 2.0 * kPi;
    return std::fmod(std::fmod(angle + kPi, period) + period, period) - kPi;
}

}  // namespace

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

// Convert OBB (cx, cy, w, h, theta) to KF measurement (x, y, h, r, theta).
Eigen::Matrix<double, 5, 1> XywhaToZObb(const Eigen::Matrix<double, 5, 1>& xywha) {
    const double cx = xywha[0];
    const double cy = xywha[1];
    const double w = std::max(xywha[2], 1.0e-4);
    const double h = std::max(xywha[3], 1.0e-4);
    const double theta = xywha[4];
    Eigen::Matrix<double, 5, 1> z;
    z << cx, cy, h, w / h, theta;
    return z;
}

// Convert KF state (x, y, h, r, theta) back to OBB (cx, cy, w, h, theta).
Eigen::Matrix<double, 5, 1> ZObbToXywha(const Eigen::Matrix<double, 5, 1>& z) {
    const double cx = z[0];
    const double cy = z[1];
    const double h = z[2];
    const double r = z[3];
    const double w = h * r;
    const double theta = z[4];
    Eigen::Matrix<double, 5, 1> out;
    out << cx, cy, w, h, theta;
    return out;
}

Eigen::Vector4d XywhaToEnclosingXyxy(const Eigen::Matrix<double, 5, 1>& xywha) {
    const double cx = xywha[0];
    const double cy = xywha[1];
    const double w = xywha[2];
    const double h = xywha[3];
    const double theta = xywha[4];
    const double cos_t = std::abs(std::cos(theta));
    const double sin_t = std::abs(std::sin(theta));
    const double half_w = 0.5 * (w * cos_t + h * sin_t);
    const double half_h = 0.5 * (w * sin_t + h * cos_t);
    Eigen::Vector4d out;
    out << cx - half_w, cy - half_h, cx + half_w, cy + half_h;
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
      max_obs_(std::max(max_obs, 1)),
      is_obb_(detection.is_obb) {
    id = NextId();
    if (is_obb_) {
        const Eigen::Matrix<double, 5, 1> z = XywhaToZObb(detection.xywha);
        KalmanFilterXYHR::Vector measurement(5);
        measurement << z[0], z[1], z[2], z[3], WrapAngle(z[4]);
        kf.Initiate(measurement);
    } else {
        const Eigen::Vector4d z = XyxyToZ(detection.xyxy);
        KalmanFilterXYHR::Vector measurement(4);
        measurement << z[0], z[1], z[2], z[3];
        kf.Initiate(measurement);
    }
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
    if (is_obb_) {
        KalmanFilterXYHR::Vector z(5);
        const Eigen::Matrix<double, 5, 1> zo = XywhaToZObb(detection.xywha);
        z << zo[0], zo[1], zo[2], zo[3], zo[4];
        kf.Update(z, alpha);
    } else {
        KalmanFilterXYHR::Vector z(4);
        const Eigen::Vector4d za = XyxyToZ(detection.xyxy);
        z << za[0], za[1], za[2], za[3];
        kf.Update(z, alpha);
    }
    conf = detection.conf;
    cls = detection.cls;
    det_ind = detection.det_ind;
}

void KalmanBoxTracker::CameraUpdate(
    const Eigen::Matrix2d& linear,
    const Eigen::Vector2d& translation
) {
    KalmanFilterXYHR::Vector& mean = kf.mutable_mean();
    if (is_obb_) {
        // Approximate isotropic scale + planar rotation from the linear part,
        // mirroring boosttrack.KalmanBoxTracker.camera_update for OBB.
        const double cx = mean[0];
        const double cy = mean[1];
        const double h = mean[2];
        const double r = mean[3];
        const double theta = mean[4];
        const double w = h * r;
        const Eigen::Vector2d p = linear * Eigen::Vector2d(cx, cy) + translation;
        const double det_abs = std::abs(linear.determinant());
        const double scale = std::sqrt(std::max(det_abs, 1.0e-8));
        const double rot = std::atan2(linear(1, 0), linear(0, 0));
        const double w2 = std::max(w * scale, 1.0e-4);
        const double h2 = std::max(h * scale, 1.0e-4);
        mean[0] = p[0];
        mean[1] = p[1];
        mean[2] = h2;
        mean[3] = w2 / h2;
        mean[4] = WrapAngle(theta + rot);
        return;
    }
    const Eigen::Vector4d box = xyxy();
    Eigen::Vector2d p1 = linear * Eigen::Vector2d(box[0], box[1]) + translation;
    Eigen::Vector2d p2 = linear * Eigen::Vector2d(box[2], box[3]) + translation;
    const double w = p2[0] - p1[0];
    const double h = p2[1] - p1[1];
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

Eigen::Matrix<double, 5, 1> KalmanBoxTracker::xywha() const {
    Eigen::Matrix<double, 5, 1> state;
    if (is_obb_) {
        state << kf.mean()[0], kf.mean()[1], kf.mean()[2], kf.mean()[3], kf.mean()[4];
        Eigen::Matrix<double, 5, 1> out = ZObbToXywha(state);
        out[4] = WrapAngle(out[4]);
        return out;
    }
    const Eigen::Vector4d aabb = xyxy();
    const Eigen::Vector4d wh = XyxyToCxcywh(aabb);
    state << wh[0], wh[1], wh[2], wh[3], 0.0;
    return state;
}

Eigen::Vector4d KalmanBoxTracker::xyxy() const {
    if (is_obb_) {
        return XywhaToEnclosingXyxy(xywha());
    }
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
