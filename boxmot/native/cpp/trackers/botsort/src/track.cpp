#include "botsort/track.hpp"

#include <Eigen/SVD>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>

namespace botsort {

int Track::count_ = 0;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kHalfPi = kPi / 2.0;

double WrapAngle(const double angle) {
    const double period = 2.0 * kPi;
    return std::fmod(std::fmod(angle + kPi, period) + period, period) - kPi;
}

double ProperRotationAngle(const Eigen::Matrix2d& linear) {
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(linear, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d u = svd.matrixU();
    const Eigen::Matrix2d v = svd.matrixV();
    Eigen::Matrix2d rotation = u * v.transpose();
    if (rotation.determinant() < 0.0) {
        u.col(1) *= -1.0;
        rotation = u * v.transpose();
    }
    return std::atan2(rotation(1, 0), rotation(0, 0));
}

bool IsSimilarityTransform(const Eigen::Matrix2d& linear, double& scale) {
    const double scale_squared = (linear.transpose() * linear).trace() / 2.0;
    if (scale_squared <= 0.0 || linear.determinant() <= 0.0) {
        return false;
    }

    const Eigen::Matrix2d gram = linear.transpose() * linear;
    const Eigen::Matrix2d expected = scale_squared * Eigen::Matrix2d::Identity();
    for (int row = 0; row < 2; ++row) {
        for (int column = 0; column < 2; ++column) {
            const double tolerance = 1.0e-10 + (1.0e-7 * std::abs(expected(row, column)));
            if (std::abs(gram(row, column) - expected(row, column)) > tolerance) {
                return false;
            }
        }
    }
    scale = std::sqrt(scale_squared);
    return true;
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

Eigen::Vector4d XywhToXyxy(const Eigen::Vector4d& xywh) {
    const double half_w = xywh[2] / 2.0;
    const double half_h = xywh[3] / 2.0;
    return Eigen::Vector4d(
        xywh[0] - half_w,
        xywh[1] - half_h,
        xywh[0] + half_w,
        xywh[1] + half_h
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

std::array<cv::Point2f, 4> XywhaToCorners(const Eigen::Matrix<double, 5, 1>& box) {
    std::array<cv::Point2f, 4> corners{};
    RotatedRectFromXywha(box).points(corners.data());
    return corners;
}

Eigen::Vector4d ObbToXyxy(const Eigen::Matrix<double, 5, 1>& box) {
    const std::array<cv::Point2f, 4> corners = XywhaToCorners(box);

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
        const double theta_aligned = ref_theta + WrapAngle(candidate[2] - ref_theta);
        const double angle_cost = std::abs(theta_aligned - ref_theta);
        const double size_cost =
            std::abs(std::log(std::max(candidate[0], 1.0e-6) / ref_w)) +
            std::abs(std::log(std::max(candidate[1], 1.0e-6) / ref_h));
        const double cost = angle_cost + (0.05 * size_cost);
        if (cost < best_cost) {
            best_cost = cost;
            best = {candidate[0], candidate[1], theta_aligned};
        }
    }

    Eigen::Matrix<double, 5, 1> aligned = box;
    aligned[2] = best[0];
    aligned[3] = best[1];
    aligned[4] = WrapAngle(best[2]);
    return aligned;
}

Eigen::Matrix<double, 5, 1> CornersToXywha(
    const std::array<cv::Point2f, 4>& corners,
    const Eigen::Matrix<double, 5, 1>& reference
) {
    const std::vector<cv::Point2f> points(corners.begin(), corners.end());
    const cv::RotatedRect rect = cv::minAreaRect(points);

    Eigen::Matrix<double, 5, 1> box;
    box << rect.center.x,
        rect.center.y,
        std::max(static_cast<double>(rect.size.width), 1.0e-4),
        std::max(static_cast<double>(rect.size.height), 1.0e-4),
        rect.angle * kPi / 180.0;
    return AlignObbBox(box, reference);
}

Eigen::Matrix<double, 5, 1> WarpObbMeasurement(
    const Eigen::Matrix<double, 5, 1>& measurement,
    const Eigen::Matrix2d& linear,
    const Eigen::Vector2d& translation,
    const Eigen::Matrix<double, 5, 1>* alignment_reference = nullptr
) {
    // Python transform_obb handles the SOF/ECC similarity cases analytically,
    // avoiding float32 minAreaRect noise on the benchmark-default CMC paths.
    double similarity_scale = 1.0;
    if (IsSimilarityTransform(linear, similarity_scale)) {
        Eigen::Matrix<double, 5, 1> warped = measurement;
        warped.head<2>() = (linear * measurement.head<2>()) + translation;
        warped[2] *= similarity_scale;
        warped[3] *= similarity_scale;
        warped[4] = WrapAngle(measurement[4] + std::atan2(linear(1, 0), linear(0, 0)));
        if (alignment_reference != nullptr) {
            warped = AlignObbBox(warped, *alignment_reference);
        }
        return warped;
    }

    const std::array<cv::Point2f, 4> source_corners = XywhaToCorners(measurement);
    std::array<cv::Point2f, 4> warped_corners{};
    for (std::size_t index = 0; index < source_corners.size(); ++index) {
        const Eigen::Vector2d point(source_corners[index].x, source_corners[index].y);
        const Eigen::Vector2d warped = (linear * point) + translation;
        warped_corners[index] = cv::Point2f(
            static_cast<float>(warped[0]),
            static_cast<float>(warped[1])
        );
    }

    Eigen::Matrix<double, 5, 1> reference = measurement;
    if (alignment_reference != nullptr) {
        reference = *alignment_reference;
    } else {
        reference.head<2>() = (linear * measurement.head<2>()) + translation;
        const double rotation = ProperRotationAngle(linear);
        reference[4] = WrapAngle(measurement[4] + rotation);
    }
    return CornersToXywha(warped_corners, reference);
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
    }
    UpdateClass(cls, conf);
    if (detection.has_embedding()) {
        UpdateFeatures(detection.embedding);
    }
}

void Track::ResetCount() {
    count_ = 0;
}

int Track::NextId() {
    ++count_;
    return count_;
}

Eigen::VectorXf Track::Normalize(const Eigen::VectorXf& feat) {
    if (feat.size() == 0) {
        return feat;
    }
    const float norm = feat.norm();
    if (norm <= 1.0e-12F) {
        return feat;
    }
    return feat / norm;
}

void Track::UpdateFeatures(const Eigen::VectorXf& feat) {
    if (feat.size() == 0) {
        return;
    }
    curr_feat_ = Normalize(feat);
    if (smooth_feat_.size() == 0) {
        smooth_feat_ = curr_feat_;
        return;
    }
    smooth_feat_ = (alpha_ * smooth_feat_) + ((1.0F - alpha_) * curr_feat_);
    smooth_feat_ = Normalize(smooth_feat_);
}

void Track::UpdateClass(const int cls_id, const float confidence) {
    cls_hist_[cls_id] += confidence;
    float best_score = -1.0F;
    int best_cls = cls_id;
    for (const auto& item : cls_hist_) {
        if (item.second > best_score) {
            best_score = item.second;
            best_cls = item.first;
        }
    }
    cls = best_cls;
}

Eigen::VectorXd Track::Measurement() const {
    if (is_obb_) {
        return xywha();
    }
    Eigen::VectorXd measurement(4);
    measurement = xywh();
    return measurement;
}

void Track::Activate(const KalmanFilterXYWH& kalman_filter, const int current_frame_id) {
    id = NextId();
    const auto [new_mean, new_covariance] = kalman_filter.Initiate(Measurement());
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
    const KalmanFilterXYWH& kalman_filter,
    const int current_frame_id,
    const bool new_id
) {
    const auto [new_mean, new_covariance] = kalman_filter.Update(mean, covariance, new_track.Measurement());
    mean = new_mean;
    covariance = new_covariance;
    if (new_track.curr_feat_.size() > 0) {
        UpdateFeatures(new_track.curr_feat_);
    }
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
    UpdateClass(new_track.cls, new_track.conf);
}

void Track::Update(const Track& new_track, const KalmanFilterXYWH& kalman_filter, const int current_frame_id) {
    frame_id = current_frame_id;
    ++tracklet_len;
    const auto [new_mean, new_covariance] = kalman_filter.Update(mean, covariance, new_track.Measurement());
    mean = new_mean;
    covariance = new_covariance;
    if (new_track.curr_feat_.size() > 0) {
        UpdateFeatures(new_track.curr_feat_);
    }
    state = TrackState::kTracked;
    is_activated = true;
    conf = new_track.conf;
    cls = new_track.cls;
    det_ind = new_track.det_ind;
    UpdateClass(new_track.cls, new_track.conf);
}

void Track::Predict(const KalmanFilterXYWH& kalman_filter) {
    if (mean.size() == 0 || covariance.size() == 0) {
        return;
    }
    KalmanFilterXYWH::Vector mean_state = mean;
    if (state != TrackState::kTracked) {
        if (is_obb_ && mean_state.size() >= 10) {
            mean_state[7] = 0.0;
            mean_state[8] = 0.0;
            mean_state[9] = 0.0;
        } else if (!is_obb_ && mean_state.size() >= 8) {
            mean_state[6] = 0.0;
            mean_state[7] = 0.0;
        }
    }
    const auto [predicted_mean, predicted_covariance] = kalman_filter.Predict(mean_state, covariance);
    mean = predicted_mean;
    covariance = predicted_covariance;
}

void Track::ApplyAffine(const Eigen::Matrix2d& linear, const Eigen::Vector2d& translation) {
    if (mean.size() == 0 || covariance.size() == 0) {
        return;
    }

    if (!is_obb_) {
        if (mean.size() != 8 || covariance.rows() != 8 || covariance.cols() != 8) {
            return;
        }

        // Match Python transform_aabb_kalman_state. Warping all four corners
        // and taking their enclosing AABB preserves positive width/height
        // under camera rotation or shear; applying the affine linear matrix
        // directly to (w, h) can instead shrink or flip the box.
        Eigen::Matrix4d measurement_jacobian = Eigen::Matrix4d::Zero();
        measurement_jacobian.block<2, 2>(0, 0) = linear;
        measurement_jacobian(2, 2) = std::abs(linear(0, 0));
        measurement_jacobian(2, 3) = std::abs(linear(0, 1));
        measurement_jacobian(3, 2) = std::abs(linear(1, 0));
        measurement_jacobian(3, 3) = std::abs(linear(1, 1));

        Eigen::Matrix<double, 8, 8> state_transform = Eigen::Matrix<double, 8, 8>::Zero();
        state_transform.block<4, 4>(0, 0) = measurement_jacobian;
        state_transform.block<4, 4>(4, 4) = measurement_jacobian;

        Eigen::Matrix<double, 8, 1> transformed_mean = state_transform * mean;
        transformed_mean.segment<2>(0) += translation;
        mean = transformed_mean;
        covariance = (state_transform * covariance * state_transform.transpose()).eval();
        covariance = (0.5 * (covariance + covariance.transpose())).eval();
        return;
    }

    if (mean.size() != 10 || covariance.rows() != 10 || covariance.cols() != 10) {
        return;
    }

    const Eigen::Matrix<double, 5, 1> measurement = mean.head<5>();
    const Eigen::Matrix<double, 5, 1> warped_measurement =
        WarpObbMeasurement(measurement, linear, translation);

    // Match Python transform_obb_kalman_state: the corner warp and OBB refit
    // can couple position, dimensions, and angle under rotation, anisotropic
    // scale, or shear. Carry those couplings into velocity and covariance via
    // the numerical measurement Jacobian instead of treating w/h as two
    // independent scalar scales.
    Eigen::Matrix<double, 5, 5> jacobian;
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
            WarpObbMeasurement(plus, linear, translation, &warped_measurement)
            - WarpObbMeasurement(minus, linear, translation, &warped_measurement);
        delta[4] = WrapAngle(delta[4]);
        jacobian.col(index) = delta / (plus[index] - minus[index]);
    }

    const Eigen::Matrix<double, 5, 1> velocity = mean.segment<5>(5);
    mean.head<5>() = warped_measurement;
    mean.segment<5>(5) = jacobian * velocity;

    Eigen::Matrix<double, 10, 10> state_transform = Eigen::Matrix<double, 10, 10>::Zero();
    state_transform.block<5, 5>(0, 0) = jacobian;
    state_transform.block<5, 5>(5, 5) = jacobian;
    covariance = (state_transform * covariance * state_transform.transpose()).eval();
    covariance = (0.5 * (covariance + covariance.transpose())).eval();
}

Eigen::Vector4d Track::xyxy() const {
    if (is_obb_) {
        return ObbToXyxy(xywha());
    }
    if (mean.size() >= 4) {
        return XywhToXyxy(mean.head<4>());
    }
    return XywhToXyxy(xywh_);
}

Eigen::Vector4d Track::xywh() const {
    if (is_obb_) {
        return XyxyToXywh(xyxy());
    }
    if (mean.size() >= 4) {
        Eigen::Vector4d result = mean.head<4>();
        result[2] = std::max(result[2], 1.0e-4);
        result[3] = std::max(result[3], 1.0e-4);
        return result;
    }
    return xywh_;
}

Eigen::Matrix<double, 5, 1> Track::xywha() const {
    if (is_obb_) {
        if (mean.size() >= 5) {
            Eigen::Matrix<double, 5, 1> result = mean.head<5>();
            result[2] = std::max(result[2], 1.0e-4);
            result[3] = std::max(result[3], 1.0e-4);
            result[4] = WrapAngle(result[4]);
            return result;
        }
        return xywha_;
    }

    const Eigen::Vector4d box = xywh();
    Eigen::Matrix<double, 5, 1> result;
    result << box[0], box[1], box[2], box[3], 0.0;
    return result;
}

bool Track::UsesObb() const {
    return is_obb_;
}

}  // namespace botsort
