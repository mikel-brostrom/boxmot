#include "ocsort/tracker.hpp"

#include "boxmot/trackers/base/assignment.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ocsort {

using boxmot::trackers::base::LinearAssignment;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kAssignmentThreshold = 1.0e9;

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

double AabbIoU(const Eigen::Vector4d& lhs, const Eigen::Vector4d& rhs) {
    const double x1 = std::max(lhs[0], rhs[0]);
    const double y1 = std::max(lhs[1], rhs[1]);
    const double x2 = std::min(lhs[2], rhs[2]);
    const double y2 = std::min(lhs[3], rhs[3]);
    const double inter_w = std::max(0.0, x2 - x1);
    const double inter_h = std::max(0.0, y2 - y1);
    const double inter = inter_w * inter_h;
    const double lhs_area = std::max(0.0, lhs[2] - lhs[0]) * std::max(0.0, lhs[3] - lhs[1]);
    const double rhs_area = std::max(0.0, rhs[2] - rhs[0]) * std::max(0.0, rhs[3] - rhs[1]);
    const double denom = lhs_area + rhs_area - inter;
    if (denom <= 1.0e-12) {
        return 0.0;
    }
    return inter / denom;
}

double ObbIoU(const Eigen::Matrix<double, 5, 1>& lhs, const Eigen::Matrix<double, 5, 1>& rhs) {
    const cv::RotatedRect lhs_rect = RotatedRectFromXywha(lhs);
    const cv::RotatedRect rhs_rect = RotatedRectFromXywha(rhs);

    std::vector<cv::Point2f> intersection;
    const int status = cv::rotatedRectangleIntersection(lhs_rect, rhs_rect, intersection);
    if (status == cv::INTERSECT_NONE || intersection.empty()) {
        return 0.0;
    }

    const double inter_area = std::abs(cv::contourArea(intersection));
    const double lhs_area = std::max(lhs[2], 0.0) * std::max(lhs[3], 0.0);
    const double rhs_area = std::max(rhs[2], 0.0) * std::max(rhs[3], 0.0);
    const double denom = lhs_area + rhs_area - inter_area;
    if (denom <= 1.0e-12) {
        return 0.0;
    }
    return inter_area / denom;
}

Eigen::VectorXd DetectionRow(const Detection& detection) {
    if (detection.is_obb) {
        Eigen::VectorXd row(6);
        row << detection.xywha[0], detection.xywha[1], detection.xywha[2], detection.xywha[3], detection.xywha[4], detection.conf;
        return row;
    }
    Eigen::VectorXd row(5);
    row << detection.xyxy[0], detection.xyxy[1], detection.xyxy[2], detection.xyxy[3], detection.conf;
    return row;
}

Eigen::VectorXd PredictionRow(const Eigen::VectorXd& box, const bool is_obb_mode) {
    Eigen::VectorXd row(is_obb_mode ? 6 : 5);
    row.head(box.size()) = box;
    row[box.size()] = 0.0;
    return row;
}

Eigen::Vector2d CenterOf(const Eigen::VectorXd& box, const bool is_obb_mode) {
    if (is_obb_mode) {
        return Eigen::Vector2d(box[0], box[1]);
    }
    return Eigen::Vector2d((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0);
}

Eigen::Vector2d DirectionBetween(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs, const bool is_obb_mode) {
    const Eigen::Vector2d center1 = CenterOf(lhs, is_obb_mode);
    const Eigen::Vector2d center2 = CenterOf(rhs, is_obb_mode);
    Eigen::Vector2d direction(center2[1] - center1[1], center2[0] - center1[0]);
    const double norm = std::max(direction.norm(), 1.0e-6);
    direction /= norm;
    return direction;
}

bool ObservationIsValid(const Eigen::VectorXd& box) {
    return box.size() > 0 && box[box.size() - 1] >= 0.0;
}

}  // namespace

int OCSORTTracker::KalmanBoxTracker::count = 0;

Eigen::Vector4d OCSORTTracker::KalmanBoxTracker::XyxyToXysr(const Eigen::Vector4d& bbox) {
    const double width = std::max(bbox[2] - bbox[0], 1.0e-6);
    const double height = std::max(bbox[3] - bbox[1], 1.0e-6);
    return Eigen::Vector4d(
        (bbox[0] + bbox[2]) / 2.0,
        (bbox[1] + bbox[3]) / 2.0,
        width * height,
        width / height
    );
}

Eigen::Matrix<double, 5, 1> OCSORTTracker::KalmanBoxTracker::ConvertObbToZ(const Eigen::Matrix<double, 5, 1>& obb) {
    Eigen::Matrix<double, 5, 1> z;
    const double width = std::max(obb[2], 1.0e-6);
    const double height = std::max(obb[3], 1.0e-6);
    z << obb[0], obb[1], width * height, width / height, obb[4];
    return z;
}

Eigen::VectorXd OCSORTTracker::KalmanBoxTracker::ConvertXToBbox(const KalmanFilterXYSR::Vector& state) {
    const double width = std::sqrt(std::max(state[2] * state[3], 1.0e-12));
    const double height = state[2] / std::max(width, 1.0e-6);
    Eigen::VectorXd bbox(4);
    bbox << state[0] - (width / 2.0), state[1] - (height / 2.0), state[0] + (width / 2.0), state[1] + (height / 2.0);
    return bbox;
}

Eigen::VectorXd OCSORTTracker::KalmanBoxTracker::ConvertXToObb(const KalmanFilterXYSR::Vector& state) {
    const double width = std::sqrt(std::max(state[2] * state[3], 1.0e-12));
    const double height = state[2] / std::max(width, 1.0e-6);
    Eigen::VectorXd obb(5);
    obb << state[0], state[1], width, height, state[4];
    return obb;
}

Eigen::Vector2d OCSORTTracker::KalmanBoxTracker::SpeedDirection(
    const Eigen::VectorXd& bbox1,
    const Eigen::VectorXd& bbox2,
    const bool is_obb_mode
) {
    return DirectionBetween(bbox1, bbox2, is_obb_mode);
}

Eigen::VectorXd OCSORTTracker::KalmanBoxTracker::ObservationVector(const Detection& detection) {
    return DetectionRow(detection);
}

OCSORTTracker::KalmanBoxTracker::KalmanBoxTracker(
    const Detection& detection,
    const int delta_t_value,
    const int max_obs_value,
    const double q_xy_scaling_value,
    const double q_s_scaling_value,
    const bool is_obb_mode
) :
    det_ind(detection.det_ind),
    q_xy_scaling(q_xy_scaling_value),
    q_s_scaling(q_s_scaling_value),
    q_a_scaling(q_s_scaling_value),
    is_obb(is_obb_mode),
    kf(is_obb_mode ? 9 : 7, is_obb_mode ? 5 : 4, max_obs_value),
    id(count++),
    max_obs(max_obs_value),
    conf(detection.conf),
    cls(detection.cls),
    last_observation(OCSORTTracker::PlaceholderObservation(is_obb_mode)),
    delta_t(delta_t_value) {
    if (is_obb) {
        kf.F <<
            1, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1;
        kf.H <<
            1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0;
        kf.R.block(2, 2, 3, 3) *= 10.0;
        kf.P.block(5, 5, 4, 4) *= 1000.0;
        kf.P *= 10.0;
        kf.Q.block(5, 5, 2, 2) *= q_xy_scaling;
        kf.Q(7, 7) *= q_s_scaling;
        kf.Q(8, 8) *= q_a_scaling;
        kf.x.head(5) = ConvertObbToZ(detection.xywha);
    } else {
        kf.F <<
            1, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 1;
        kf.H <<
            1, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0;
        kf.R.block(2, 2, 2, 2) *= 10.0;
        kf.P.block(4, 4, 3, 3) *= 1000.0;
        kf.P *= 10.0;
        kf.Q.block(4, 4, 2, 2) *= q_xy_scaling;
        kf.Q(6, 6) *= q_s_scaling;
        kf.x.head(4) = XyxyToXysr(detection.xyxy);
    }
}

void OCSORTTracker::KalmanBoxTracker::ResetCount() {
    count = 0;
}

Eigen::VectorXd OCSORTTracker::KalmanBoxTracker::Predict() {
    if (is_obb) {
        if ((kf.x[7] + kf.x[2]) <= 0.0) {
            kf.x[7] = 0.0;
        }
    } else if ((kf.x[6] + kf.x[2]) <= 0.0) {
        kf.x[6] = 0.0;
    }

    kf.Predict();
    ++age;
    if (time_since_update > 0) {
        hit_streak = 0;
    }
    ++time_since_update;
    return is_obb ? ConvertXToObb(kf.x) : ConvertXToBbox(kf.x);
}

void OCSORTTracker::KalmanBoxTracker::Update(const Detection* detection) {
    if (detection == nullptr) {
        kf.UpdateMissing();
        return;
    }

    det_ind = detection->det_ind;
    conf = detection->conf;
    cls = detection->cls;
    const Eigen::VectorXd bbox = ObservationVector(*detection);
    if (last_observation.sum() >= 0.0) {
        Eigen::VectorXd previous_box;
        bool found = false;
        for (int index = 0; index < delta_t; ++index) {
            const int dt = delta_t - index;
            const auto it = observations.find(age - dt);
            if (it != observations.end()) {
                previous_box = it->second;
                found = true;
                break;
            }
        }
        if (!found) {
            previous_box = last_observation;
        }
        velocity = SpeedDirection(previous_box, bbox, is_obb);
    }

    last_observation = bbox;
    observations[age] = bbox;
    time_since_update = 0;
    ++hits;
    ++hit_streak;
    if (static_cast<int>(history_observations.size()) >= max_obs) {
        history_observations.pop_front();
    }
    history_observations.push_back(bbox);

    if (is_obb) {
        kf.Update(ConvertObbToZ(detection->xywha));
    } else {
        kf.Update(XyxyToXysr(detection->xyxy));
    }
}

Eigen::VectorXd OCSORTTracker::KalmanBoxTracker::GetState() const {
    return is_obb ? ConvertXToObb(kf.x) : ConvertXToBbox(kf.x);
}

Eigen::VectorXd OCSORTTracker::KalmanBoxTracker::CurrentOutputBox() const {
    if (last_observation.sum() < 0.0) {
        return GetState();
    }
    return last_observation.head(is_obb ? 5 : 4);
}

OCSORTTracker::OCSORTTracker(Config config) : config_(std::move(config)) {
    KalmanBoxTracker::ResetCount();
}

void OCSORTTracker::Reset() {
    frame_count_ = 0;
    detection_mode_ready_ = false;
    is_obb_mode_ = false;
    KalmanBoxTracker::ResetCount();
    active_tracks_.clear();
}

Eigen::VectorXd OCSORTTracker::PlaceholderObservation(const bool is_obb_mode) {
    return Eigen::VectorXd::Constant(is_obb_mode ? 6 : 5, -1.0);
}

Eigen::VectorXd OCSORTTracker::KPreviousObs(
    const std::unordered_map<int, Eigen::VectorXd>& observations,
    const int current_age,
    const int k,
    const bool is_obb_mode
) {
    if (observations.empty()) {
        return PlaceholderObservation(is_obb_mode);
    }
    for (int index = 0; index < k; ++index) {
        const int dt = k - index;
        const auto it = observations.find(current_age - dt);
        if (it != observations.end()) {
            return it->second;
        }
    }

    int max_age = std::numeric_limits<int>::min();
    const Eigen::VectorXd* last = nullptr;
    for (const auto& entry : observations) {
        if (entry.first > max_age) {
            max_age = entry.first;
            last = &entry.second;
        }
    }
    return last != nullptr ? *last : PlaceholderObservation(is_obb_mode);
}

Eigen::MatrixXd OCSORTTracker::SimilarityMatrix(
    const std::vector<Eigen::VectorXd>& detections,
    const std::vector<Eigen::VectorXd>& tracks,
    const bool is_obb_mode
) {
    Eigen::MatrixXd similarity = Eigen::MatrixXd::Zero(
        static_cast<int>(detections.size()),
        static_cast<int>(tracks.size())
    );
    for (int det_index = 0; det_index < static_cast<int>(detections.size()); ++det_index) {
        for (int track_index = 0; track_index < static_cast<int>(tracks.size()); ++track_index) {
            if (is_obb_mode) {
                similarity(det_index, track_index) = ObbIoU(
                    detections[det_index].head<5>(),
                    tracks[track_index].head<5>()
                );
            } else {
                similarity(det_index, track_index) = AabbIoU(
                    detections[det_index].head<4>(),
                    tracks[track_index].head<4>()
                );
            }
        }
    }
    return similarity;
}

Eigen::MatrixXd OCSORTTracker::DirectionCost(
    const std::vector<Eigen::VectorXd>& detections,
    const std::vector<Eigen::VectorXd>& previous_obs,
    const std::vector<Eigen::Vector2d>& velocities,
    const std::vector<float>& scores,
    const float inertia,
    const bool is_obb_mode
) {
    Eigen::MatrixXd cost = Eigen::MatrixXd::Zero(
        static_cast<int>(detections.size()),
        static_cast<int>(previous_obs.size())
    );
    for (int track_index = 0; track_index < static_cast<int>(previous_obs.size()); ++track_index) {
        if (!ObservationIsValid(previous_obs[track_index])) {
            continue;
        }
        const Eigen::Vector2d velocity = velocities[track_index];
        const double velocity_norm = std::max(velocity.norm(), 1.0e-6);
        for (int det_index = 0; det_index < static_cast<int>(detections.size()); ++det_index) {
            const Eigen::Vector2d det_direction = DirectionBetween(
                previous_obs[track_index],
                detections[det_index],
                is_obb_mode
            );
            const double diff_angle_cos = std::clamp(
                ((velocity[1] / velocity_norm) * det_direction[1]) +
                ((velocity[0] / velocity_norm) * det_direction[0]),
                -1.0,
                1.0
            );
            const double diff_angle = std::acos(diff_angle_cos);
            const double angle_score = (kPi / 2.0 - std::abs(diff_angle)) / kPi;
            cost(det_index, track_index) = angle_score * static_cast<double>(inertia) * static_cast<double>(scores[det_index]);
        }
    }
    return cost;
}

OCSORTTracker::AssignmentResult OCSORTTracker::Associate(
    const std::vector<Eigen::VectorXd>& detections,
    const std::vector<Eigen::VectorXd>& trackers,
    const std::vector<Eigen::Vector2d>& velocities,
    const std::vector<Eigen::VectorXd>& previous_obs,
    const float iou_threshold,
    const float inertia,
    const bool is_obb_mode
) {
    AssignmentResult result;
    if (trackers.empty()) {
        result.unmatched_rows.resize(static_cast<int>(detections.size()));
        std::iota(result.unmatched_rows.begin(), result.unmatched_rows.end(), 0);
        return result;
    }

    const Eigen::MatrixXd iou_matrix = SimilarityMatrix(detections, trackers, is_obb_mode);
    std::vector<std::pair<int, int>> matched_indices;
    if (std::min(iou_matrix.rows(), iou_matrix.cols()) > 0) {
        const Eigen::ArrayXXi admissible = (iou_matrix.array() > static_cast<double>(iou_threshold)).cast<int>();
        const int max_row_sum = admissible.rows() == 0 ? 0 : admissible.rowwise().sum().maxCoeff();
        const int max_col_sum = admissible.cols() == 0 ? 0 : admissible.colwise().sum().maxCoeff();
        if (max_row_sum == 1 && max_col_sum == 1) {
            for (int row = 0; row < admissible.rows(); ++row) {
                for (int col = 0; col < admissible.cols(); ++col) {
                    if (admissible(row, col) != 0) {
                        matched_indices.emplace_back(row, col);
                    }
                }
            }
        } else {
            std::vector<float> scores;
            scores.reserve(detections.size());
            for (const auto& detection : detections) {
                scores.push_back(static_cast<float>(detection[detection.size() - 1]));
            }
            matched_indices = LinearAssignment(
                -(iou_matrix + DirectionCost(detections, previous_obs, velocities, scores, inertia, is_obb_mode)),
                kAssignmentThreshold
            ).matches;
        }
    }

    std::vector<bool> matched_det(detections.size(), false);
    std::vector<bool> matched_trk(trackers.size(), false);
    for (const auto& match : matched_indices) {
        if (iou_matrix(match.first, match.second) < static_cast<double>(iou_threshold)) {
            continue;
        }
        matched_det[match.first] = true;
        matched_trk[match.second] = true;
        result.matches.push_back(match);
    }

    for (int det_index = 0; det_index < static_cast<int>(detections.size()); ++det_index) {
        if (!matched_det[det_index]) {
            result.unmatched_rows.push_back(det_index);
        }
    }
    for (int trk_index = 0; trk_index < static_cast<int>(trackers.size()); ++trk_index) {
        if (!matched_trk[trk_index]) {
            result.unmatched_cols.push_back(trk_index);
        }
    }
    return result;
}

TrackOutput OCSORTTracker::FormatTrack(const KalmanBoxTracker& track) {
    TrackOutput output;
    output.is_obb = track.is_obb;
    output.id = track.id + 1;
    output.conf = track.conf;
    output.cls = track.cls;
    output.det_ind = track.det_ind;
    if (track.is_obb) {
        const Eigen::VectorXd box = track.CurrentOutputBox();
        output.xywha << box[0], box[1], box[2], box[3], box[4];
    } else {
        const Eigen::VectorXd box = track.CurrentOutputBox();
        output.xyxy << box[0], box[1], box[2], box[3];
    }
    return output;
}

std::vector<TrackOutput> OCSORTTracker::Update(const std::vector<Detection>& detections, const cv::Mat& image) {
    (void)image;
    if (!detections.empty()) {
        const bool det_is_obb = detections.front().is_obb;
        if (!detection_mode_ready_) {
            detection_mode_ready_ = true;
            is_obb_mode_ = det_is_obb;
        } else if (det_is_obb != is_obb_mode_) {
            throw std::runtime_error("Native OCSORT cannot switch between AABB and OBB detections after initialization.");
        }
    }

    ++frame_count_;

    std::vector<Detection> detections_second;
    std::vector<Detection> detections_first;
    detections_second.reserve(detections.size());
    detections_first.reserve(detections.size());
    for (const auto& detection : detections) {
        if (detection.conf > config_.min_conf && detection.conf < config_.det_thresh) {
            detections_second.push_back(detection);
        }
        if (detection.conf > config_.det_thresh) {
            detections_first.push_back(detection);
        }
    }

    std::vector<Eigen::VectorXd> predicted_tracks;
    predicted_tracks.reserve(active_tracks_.size());
    for (auto& track : active_tracks_) {
        predicted_tracks.push_back(PredictionRow(track.Predict(), is_obb_mode_));
    }

    std::vector<Eigen::Vector2d> velocities;
    std::vector<Eigen::VectorXd> last_boxes;
    std::vector<Eigen::VectorXd> k_observations;
    velocities.reserve(active_tracks_.size());
    last_boxes.reserve(active_tracks_.size());
    k_observations.reserve(active_tracks_.size());
    for (const auto& track : active_tracks_) {
        velocities.push_back(track.velocity.value_or(Eigen::Vector2d::Zero()));
        last_boxes.push_back(track.last_observation);
        k_observations.push_back(KPreviousObs(track.observations, track.age, config_.delta_t, is_obb_mode_));
    }

    std::vector<Eigen::VectorXd> first_rows;
    first_rows.reserve(detections_first.size());
    for (const auto& detection : detections_first) {
        first_rows.push_back(DetectionRow(detection));
    }

    const double association_threshold = static_cast<double>(config_.iou_threshold);
    AssignmentResult matched = Associate(
        first_rows,
        predicted_tracks,
        velocities,
        k_observations,
        config_.iou_threshold,
        config_.inertia,
        is_obb_mode_
    );
    for (const auto& match : matched.matches) {
        active_tracks_[match.second].Update(&detections_first[match.first]);
    }

    if (config_.use_byte && !detections_second.empty() && !matched.unmatched_cols.empty()) {
        std::vector<Eigen::VectorXd> second_rows;
        std::vector<Eigen::VectorXd> unmatched_predictions;
        second_rows.reserve(detections_second.size());
        unmatched_predictions.reserve(matched.unmatched_cols.size());
        for (const auto& detection : detections_second) {
            second_rows.push_back(DetectionRow(detection));
        }
        for (const int trk_index : matched.unmatched_cols) {
            unmatched_predictions.push_back(predicted_tracks[trk_index]);
        }

        const Eigen::MatrixXd second_similarity = SimilarityMatrix(second_rows, unmatched_predictions, is_obb_mode_);
        if (second_similarity.size() != 0 && second_similarity.maxCoeff() > association_threshold) {
            const auto second_matches = LinearAssignment(-second_similarity, kAssignmentThreshold).matches;
            std::vector<int> consumed_tracks;
            for (const auto& match_indices : second_matches) {
                if (second_similarity(match_indices.first, match_indices.second) < association_threshold) {
                    continue;
                }
                const int track_index = matched.unmatched_cols[match_indices.second];
                active_tracks_[track_index].Update(&detections_second[match_indices.first]);
                consumed_tracks.push_back(track_index);
            }
            std::vector<int> filtered_unmatched;
            filtered_unmatched.reserve(matched.unmatched_cols.size());
            for (const int track_index : matched.unmatched_cols) {
                if (std::find(consumed_tracks.begin(), consumed_tracks.end(), track_index) == consumed_tracks.end()) {
                    filtered_unmatched.push_back(track_index);
                }
            }
            matched.unmatched_cols = std::move(filtered_unmatched);
        }
    }

    if (!matched.unmatched_rows.empty() && !matched.unmatched_cols.empty()) {
        std::vector<Eigen::VectorXd> left_dets;
        std::vector<Eigen::VectorXd> left_trks;
        left_dets.reserve(matched.unmatched_rows.size());
        left_trks.reserve(matched.unmatched_cols.size());
        for (const int det_index : matched.unmatched_rows) {
            left_dets.push_back(first_rows[det_index]);
        }
        for (const int trk_index : matched.unmatched_cols) {
            left_trks.push_back(last_boxes[trk_index]);
        }

        const Eigen::MatrixXd rematch_similarity = SimilarityMatrix(left_dets, left_trks, is_obb_mode_);
        if (rematch_similarity.size() != 0 && rematch_similarity.maxCoeff() > association_threshold) {
            const auto rematched = LinearAssignment(-rematch_similarity, kAssignmentThreshold).matches;
            std::vector<int> consumed_det_indices;
            std::vector<int> consumed_trk_indices;
            for (const auto& match_indices : rematched) {
                if (rematch_similarity(match_indices.first, match_indices.second) < association_threshold) {
                    continue;
                }
                const int det_index = matched.unmatched_rows[match_indices.first];
                const int trk_index = matched.unmatched_cols[match_indices.second];
                active_tracks_[trk_index].Update(&detections_first[det_index]);
                consumed_det_indices.push_back(det_index);
                consumed_trk_indices.push_back(trk_index);
            }

            std::vector<int> next_unmatched_dets;
            std::vector<int> next_unmatched_trks;
            for (const int det_index : matched.unmatched_rows) {
                if (std::find(consumed_det_indices.begin(), consumed_det_indices.end(), det_index) == consumed_det_indices.end()) {
                    next_unmatched_dets.push_back(det_index);
                }
            }
            for (const int trk_index : matched.unmatched_cols) {
                if (std::find(consumed_trk_indices.begin(), consumed_trk_indices.end(), trk_index) == consumed_trk_indices.end()) {
                    next_unmatched_trks.push_back(trk_index);
                }
            }
            matched.unmatched_rows = std::move(next_unmatched_dets);
            matched.unmatched_cols = std::move(next_unmatched_trks);
        }
    }

    for (const int track_index : matched.unmatched_cols) {
        active_tracks_[track_index].Update(nullptr);
    }

    for (const int det_index : matched.unmatched_rows) {
        active_tracks_.emplace_back(
            detections_first[det_index],
            config_.delta_t,
            config_.max_obs,
            config_.q_xy_scaling,
            config_.q_s_scaling,
            is_obb_mode_
        );
    }

    std::vector<TrackOutput> outputs;
    for (int index = static_cast<int>(active_tracks_.size()) - 1; index >= 0; --index) {
        auto& track = active_tracks_[static_cast<std::size_t>(index)];
        if (track.time_since_update < 1 &&
            (track.hit_streak >= config_.min_hits || frame_count_ <= config_.min_hits)) {
            outputs.push_back(FormatTrack(track));
        }
        if (track.time_since_update > config_.max_age) {
            active_tracks_.erase(active_tracks_.begin() + index);
        }
    }
    return outputs;
}

}  // namespace ocsort
