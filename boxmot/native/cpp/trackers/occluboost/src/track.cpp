#include "occluboost/track.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace occluboost {

int KalmanBoxTracker::count_ = 0;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kHalfPi = kPi / 2.0;

double WrapAngle(const double angle) {
    const double period = 2.0 * kPi;
    return std::fmod(std::fmod(angle + kPi, period) + period, period) - kPi;
}

bool SimilarityParameters(
    const Eigen::Matrix2d& linear,
    double& scale,
    double& rotation
) {
    const Eigen::Matrix2d gram = linear.transpose() * linear;
    const double scale_sq = 0.5 * gram.trace();
    if (!(scale_sq > 0.0) || !(linear.determinant() > 0.0) || !std::isfinite(scale_sq)) {
        return false;
    }

    // Match numpy.allclose(linear.T @ linear, scale_sq * eye(2),
    // rtol=1e-7, atol=1e-10) used by transform_obb().
    const Eigen::Matrix2d expected = scale_sq * Eigen::Matrix2d::Identity();
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 2; ++col) {
            const double tolerance = 1.0e-10 + (1.0e-7 * std::abs(expected(row, col)));
            if (std::abs(gram(row, col) - expected(row, col)) > tolerance) {
                return false;
            }
        }
    }

    scale = std::sqrt(scale_sq);
    rotation = std::atan2(linear(1, 0), linear(0, 0));
    return std::isfinite(scale) && std::isfinite(rotation);
}

double ProperRotationAngle(const Eigen::Matrix2d& linear) {
    const Eigen::JacobiSVD<Eigen::Matrix2d> svd(
        linear, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d u = svd.matrixU();
    const Eigen::Matrix2d v = svd.matrixV();
    Eigen::Matrix2d rotation = u * v.transpose();
    if (rotation.determinant() < 0.0) {
        u.col(1) *= -1.0;
        rotation = u * v.transpose();
    }
    return std::atan2(rotation(1, 0), rotation(0, 0));
}

std::array<cv::Point2f, 4> XywhaToCorners(const Eigen::Matrix<double, 5, 1>& box) {
    const cv::RotatedRect rect(
        cv::Point2f(static_cast<float>(box[0]), static_cast<float>(box[1])),
        cv::Size2f(
            static_cast<float>(std::max(box[2], 1.0e-4)),
            static_cast<float>(std::max(box[3], 1.0e-4))
        ),
        static_cast<float>(box[4] * 180.0 / kPi)
    );
    std::array<cv::Point2f, 4> corners{};
    rect.points(corners.data());
    return corners;
}

Eigen::Matrix<double, 5, 1> AlignObbBox(
    const Eigen::Matrix<double, 5, 1>& box,
    const Eigen::Matrix<double, 5, 1>& reference
) {
    const double ref_w = std::max(reference[2], 1.0e-6);
    const double ref_h = std::max(reference[3], 1.0e-6);
    const double ref_theta = reference[4];
    const double width = std::max(box[2], 1.0e-6);
    const double height = std::max(box[3], 1.0e-6);
    const double theta = box[4];
    const std::array<std::array<double, 3>, 4> candidates = {{
        {width, height, theta},
        {width, height, theta + kPi},
        {height, width, theta + kHalfPi},
        {height, width, theta - kHalfPi},
    }};

    double best_cost = std::numeric_limits<double>::infinity();
    std::array<double, 3> best = candidates.front();
    for (const auto& candidate : candidates) {
        const double aligned_theta = ref_theta + WrapAngle(candidate[2] - ref_theta);
        const double size_cost =
            std::abs(std::log(std::max(candidate[0], 1.0e-6) / ref_w)) +
            std::abs(std::log(std::max(candidate[1], 1.0e-6) / ref_h));
        const double cost = std::abs(aligned_theta - ref_theta) + (0.05 * size_cost);
        if (cost < best_cost) {
            best_cost = cost;
            best = {candidate[0], candidate[1], aligned_theta};
        }
    }

    Eigen::Matrix<double, 5, 1> aligned = box;
    aligned[2] = best[0];
    aligned[3] = best[1];
    aligned[4] = WrapAngle(best[2]);
    return aligned;
}

Eigen::Matrix<double, 5, 1> WarpObbMeasurement(
    const Eigen::Matrix<double, 5, 1>& measurement,
    const Eigen::Matrix2d& linear,
    const Eigen::Vector2d& translation,
    const Eigen::Matrix<double, 5, 1>* alignment_reference = nullptr
) {
    const Eigen::Matrix<double, 5, 1> source_box = ZObbToXywha(measurement);
    double similarity_scale = 1.0;
    double similarity_rotation = 0.0;
    if (SimilarityParameters(linear, similarity_scale, similarity_rotation)) {
        Eigen::Matrix<double, 5, 1> warped_box = source_box;
        warped_box.head<2>() = (linear * source_box.head<2>()) + translation;
        warped_box[2] *= similarity_scale;
        warped_box[3] *= similarity_scale;
        warped_box[4] = WrapAngle(source_box[4] + similarity_rotation);
        if (alignment_reference != nullptr) {
            warped_box = AlignObbBox(warped_box, *alignment_reference);
        }
        return XywhaToZObb(warped_box);
    }

    const std::array<cv::Point2f, 4> source_corners = XywhaToCorners(source_box);
    std::array<cv::Point2f, 4> warped_corners{};
    for (std::size_t i = 0; i < source_corners.size(); ++i) {
        const Eigen::Vector2d point(source_corners[i].x, source_corners[i].y);
        const Eigen::Vector2d warped = (linear * point) + translation;
        warped_corners[i] = cv::Point2f(static_cast<float>(warped[0]), static_cast<float>(warped[1]));
    }

    const std::vector<cv::Point2f> points(warped_corners.begin(), warped_corners.end());
    const cv::RotatedRect rect = cv::minAreaRect(points);
    Eigen::Matrix<double, 5, 1> raw_box;
    raw_box << rect.center.x,
        rect.center.y,
        std::max(static_cast<double>(rect.size.width), 1.0e-4),
        std::max(static_cast<double>(rect.size.height), 1.0e-4),
        rect.angle * kPi / 180.0;

    Eigen::Matrix<double, 5, 1> reference = source_box;
    if (alignment_reference != nullptr) {
        reference = *alignment_reference;
    } else {
        reference.head<2>() = (linear * source_box.head<2>()) + translation;
        const double rot = ProperRotationAngle(linear);
        reference[4] = WrapAngle(source_box[4] + rot);
    }
    return XywhaToZObb(AlignObbBox(raw_box, reference));
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
    if ((linear.array() == Eigen::Matrix2d::Identity().array()).all()
        && (translation.array() == Eigen::Vector2d::Zero().array()).all()) {
        return;
    }

    KalmanFilterXYHR::Vector& mean = kf.mutable_mean();
    if (is_obb_) {
        const Eigen::Matrix<double, 5, 1> measurement = mean.head<5>();
        const Eigen::Matrix<double, 5, 1> warped_measurement =
            WarpObbMeasurement(measurement, linear, translation);
        const Eigen::Matrix<double, 5, 1> warped_box = ZObbToXywha(warped_measurement);

        Eigen::Matrix<double, 5, 5> jacobian;
        double similarity_scale = 1.0;
        double similarity_rotation = 0.0;
        if (SimilarityParameters(linear, similarity_scale, similarity_rotation)) {
            // Analytic Jacobian of [cx, cy, h, r, theta] under a proper
            // similarity transform. This mirrors Python's transform_obb fast
            // path and avoids differentiating through float minAreaRect.
            jacobian.setZero();
            jacobian.block<2, 2>(0, 0) = linear;
            jacobian(2, 2) = similarity_scale;
            jacobian(3, 3) = 1.0;
            jacobian(4, 4) = 1.0;
        } else {
            for (int index = 0; index < 5; ++index) {
                const double step = index == 4
                    ? 1.0e-3
                    : 1.0e-4 * std::max(std::abs(measurement[index]), 1.0);
                Eigen::Matrix<double, 5, 1> plus = measurement;
                Eigen::Matrix<double, 5, 1> minus = measurement;
                plus[index] += step;
                minus[index] -= step;
                if (index == 2 || index == 3) {
                    minus[index] = std::max(minus[index], 1.0e-6);
                }
                Eigen::Matrix<double, 5, 1> delta =
                    WarpObbMeasurement(plus, linear, translation, &warped_box) -
                    WarpObbMeasurement(minus, linear, translation, &warped_box);
                delta[4] = WrapAngle(delta[4]);
                jacobian.col(index) = delta / (plus[index] - minus[index]);
            }
        }

        const Eigen::Matrix<double, 5, 1> velocity = mean.segment<5>(5);
        mean.head<5>() = warped_measurement;
        mean.segment<5>(5) = jacobian * velocity;

        Eigen::Matrix<double, 10, 10> state_transform = Eigen::Matrix<double, 10, 10>::Zero();
        state_transform.block<5, 5>(0, 0) = jacobian;
        state_transform.block<5, 5>(5, 5) = jacobian;
        const Eigen::Matrix<double, 10, 10> transformed_covariance =
            state_transform * kf.covariance() * state_transform.transpose();
        kf.mutable_covariance() =
            (0.5 * (transformed_covariance + transformed_covariance.transpose())).eval();
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
