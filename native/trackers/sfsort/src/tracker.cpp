#include "sfsort/tracker.hpp"

#include "boxmot/trackers/base/assignment.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace sfsort {

using boxmot::trackers::base::AssignmentResult;
using boxmot::trackers::base::LinearAssignment;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kHalfPi = kPi / 2.0;

double Clamp(const double value, const double min_value, const double max_value) {
    return std::max(min_value, std::min(value, max_value));
}

double WrapAngle(const double angle) {
    const double period = 2.0 * kPi;
    return std::fmod(std::fmod(angle + kPi, period) + period, period) - kPi;
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

bool ContainsTrackId(const std::vector<SFSORTTracker::TrackData>& tracks, const int track_id) {
    return std::any_of(tracks.begin(), tracks.end(), [track_id](const SFSORTTracker::TrackData& track) {
        return track.track_id == track_id;
    });
}

void RemoveTracksById(std::vector<SFSORTTracker::TrackData>& tracks, const std::unordered_set<int>& track_ids) {
    tracks.erase(
        std::remove_if(
            tracks.begin(),
            tracks.end(),
            [&track_ids](const SFSORTTracker::TrackData& track) { return track_ids.count(track.track_id) > 0; }
        ),
        tracks.end()
    );
}

void RemoveTrackById(std::vector<SFSORTTracker::TrackData>& tracks, const int track_id) {
    tracks.erase(
        std::remove_if(
            tracks.begin(),
            tracks.end(),
            [track_id](const SFSORTTracker::TrackData& track) { return track.track_id == track_id; }
        ),
        tracks.end()
    );
}

double TrackCenterX(const SFSORTTracker::TrackData& track) {
    if (track.is_obb) {
        return track.xywha[0];
    }
    return (track.xyxy[0] + track.xyxy[2]) / 2.0;
}

double TrackCenterY(const SFSORTTracker::TrackData& track) {
    if (track.is_obb) {
        return track.xywha[1];
    }
    return (track.xyxy[1] + track.xyxy[3]) / 2.0;
}

Eigen::Matrix<double, 5, 1> AlignObbMeasurement(
    const Eigen::Matrix<double, 5, 1>& measurement,
    const Eigen::Matrix<double, 5, 1>& reference
) {
    Eigen::Matrix<double, 5, 1> aligned = measurement;

    const double ref_w = std::max(reference[2], 1.0e-6);
    const double ref_h = std::max(reference[3], 1.0e-6);
    const double ref_theta = reference[4];
    const double w = std::max(aligned[2], 1.0e-6);
    const double h = std::max(aligned[3], 1.0e-6);
    const double theta = aligned[4];

    const std::array<std::array<double, 3>, 4> candidates = {{
        {w, h, theta},
        {w, h, theta + kPi},
        {h, w, theta + kHalfPi},
        {h, w, theta - kHalfPi},
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

    aligned[2] = best[0];
    aligned[3] = best[1];
    aligned[4] = best[2];
    return aligned;
}

double AabbIoU(const Eigen::Vector4d& lhs, const Eigen::Vector4d& rhs) {
    const double x1 = std::max(lhs[0], rhs[0]);
    const double y1 = std::max(lhs[1], rhs[1]);
    const double x2 = std::min(lhs[2], rhs[2]);
    const double y2 = std::min(lhs[3], rhs[3]);
    const double inter_w = std::max(0.0, x2 - x1);
    const double inter_h = std::max(0.0, y2 - y1);
    const double intersection = inter_w * inter_h;
    const double lhs_area = std::max(0.0, lhs[2] - lhs[0]) * std::max(0.0, lhs[3] - lhs[1]);
    const double rhs_area = std::max(0.0, rhs[2] - rhs[0]) * std::max(0.0, rhs[3] - rhs[1]);
    const double union_area = lhs_area + rhs_area - intersection;
    if (union_area <= 1.0e-12) {
        return 0.0;
    }
    return intersection / union_area;
}

double CombinedCost(
    const double iou,
    const double centerx1,
    const double centery1,
    const double centerx2,
    const double centery2,
    const Eigen::Vector4d& lhs_xyxy,
    const Eigen::Vector4d& rhs_xyxy,
    const double sw,
    const double sh
) {
    constexpr double eps = 1.0e-7;
    const double inner_diag = std::abs(centerx1 - centerx2) + std::abs(centery1 - centery2);
    const double xxc1 = std::min(lhs_xyxy[0], rhs_xyxy[0]);
    const double yyc1 = std::min(lhs_xyxy[1], rhs_xyxy[1]);
    const double xxc2 = std::max(lhs_xyxy[2], rhs_xyxy[2]);
    const double yyc2 = std::max(lhs_xyxy[3], rhs_xyxy[3]);
    const double outer_diag = std::max(std::abs(xxc2 - xxc1) + std::abs(yyc2 - yyc1), eps);
    const double diou = iou - (inner_diag / outer_diag);
    const double bbsi = diou + sh + sw;
    return 1.0 - (bbsi / 3.0);
}

Eigen::MatrixXd CalculateAabbCost(
    const std::vector<SFSORTTracker::TrackData*>& tracks,
    const std::vector<Detection>& detections,
    const bool iou_only
) {
    Eigen::MatrixXd cost(static_cast<int>(tracks.size()), static_cast<int>(detections.size()));
    for (int row = 0; row < static_cast<int>(tracks.size()); ++row) {
        const auto* track = tracks[static_cast<std::size_t>(row)];
        const Eigen::Vector4d& lhs = track->xyxy;
        const double lhs_w = lhs[2] - lhs[0];
        const double lhs_h = lhs[3] - lhs[1];
        const double lhs_cx = (lhs[0] + lhs[2]) / 2.0;
        const double lhs_cy = (lhs[1] + lhs[3]) / 2.0;
        for (int col = 0; col < static_cast<int>(detections.size()); ++col) {
            const Eigen::Vector4d& rhs = detections[static_cast<std::size_t>(col)].xyxy;
            const double rhs_w = rhs[2] - rhs[0];
            const double rhs_h = rhs[3] - rhs[1];
            const double rhs_cx = (rhs[0] + rhs[2]) / 2.0;
            const double rhs_cy = (rhs[1] + rhs[3]) / 2.0;
            const double iou = AabbIoU(lhs, rhs);
            if (iou_only) {
                cost(row, col) = 1.0 - iou;
                continue;
            }

            const double inter_w = std::max(0.0, std::min(lhs[2], rhs[2]) - std::max(lhs[0], rhs[0]));
            const double inter_h = std::max(0.0, std::min(lhs[3], rhs[3]) - std::max(lhs[1], rhs[1]));
            const double sw = inter_w / std::abs(inter_w + std::abs(rhs_w - lhs_w) + 1.0e-7);
            const double sh = inter_h / std::abs(inter_h + std::abs(rhs_h - lhs_h) + 1.0e-7);
            cost(row, col) = CombinedCost(iou, lhs_cx, lhs_cy, rhs_cx, rhs_cy, lhs, rhs, sw, sh);
        }
    }
    return cost;
}

Eigen::MatrixXd CalculateObbCost(
    const std::vector<SFSORTTracker::TrackData*>& tracks,
    const std::vector<Detection>& detections,
    const bool iou_only
) {
    Eigen::MatrixXd cost(static_cast<int>(tracks.size()), static_cast<int>(detections.size()));
    for (int row = 0; row < static_cast<int>(tracks.size()); ++row) {
        const auto* track = tracks[static_cast<std::size_t>(row)];
        const double lhs_cx = track->xywha[0];
        const double lhs_cy = track->xywha[1];
        const double lhs_w = track->xywha[2];
        const double lhs_h = track->xywha[3];
        for (int col = 0; col < static_cast<int>(detections.size()); ++col) {
            const Detection& detection = detections[static_cast<std::size_t>(col)];
            const double iou = ObbIoU(track->xywha, detection.xywha);
            if (iou_only) {
                cost(row, col) = 1.0 - iou;
                continue;
            }

            const double rhs_w = detection.xywha[2];
            const double rhs_h = detection.xywha[3];
            const double sw = std::min(lhs_w, rhs_w) / (std::max(lhs_w, rhs_w) + 1.0e-7);
            const double sh = std::min(lhs_h, rhs_h) / (std::max(lhs_h, rhs_h) + 1.0e-7);
            cost(row, col) = CombinedCost(
                iou,
                lhs_cx,
                lhs_cy,
                detection.xywha[0],
                detection.xywha[1],
                track->xyxy,
                ObbToXyxy(detection.xywha),
                sw,
                sh
            );
        }
    }
    return cost;
}

}  // namespace

void SFSORTTracker::TrackData::Update(const Detection& detection, const int frame_id, const float obb_theta_damping) {
    if (is_obb && detection.is_obb) {
        Eigen::Matrix<double, 5, 1> aligned = AlignObbMeasurement(detection.xywha, xywha);
        const double prev_theta = xywha[4];
        const double theta_delta = WrapAngle(aligned[4] - prev_theta);
        const double damping = Clamp(static_cast<double>(obb_theta_damping), 0.0, 1.0);
        theta_velocity =
            (damping * theta_velocity) +
            ((1.0 - damping) * theta_delta);
        aligned[4] = WrapAngle(prev_theta + theta_velocity);
        xywha = aligned;
        xyxy = ObbToXyxy(xywha);
    } else {
        is_obb = false;
        xyxy = detection.xyxy;
    }

    state = TrackState::kActive;
    time_since_update = 0;
    last_frame = frame_id;
    conf = detection.conf;
    cls = detection.cls;
    det_ind = detection.det_ind;
}

SFSORTTracker::SFSORTTracker(Config config) : config_(std::move(config)) {}

void SFSORTTracker::Reset() {
    frame_count_ = 0;
    id_counter_ = 0;
    margins_ready_ = false;
    detection_mode_ready_ = false;
    is_obb_mode_ = false;
    l_margin_ = 0.0;
    r_margin_ = 0.0;
    t_margin_ = 0.0;
    b_margin_ = 0.0;
    active_tracks_.clear();
    lost_tracks_.clear();
}

std::tuple<float, float, float> SFSORTTracker::DynamicThresholds(const std::vector<Detection>& detections) const {
    float hth = static_cast<float>(Clamp(config_.high_th, 0.0, 1.0));
    float nth = static_cast<float>(Clamp(config_.new_track_th, hth, 1.0));
    float mth = static_cast<float>(Clamp(config_.match_th_first, 0.0, 0.67));
    if (!config_.dynamic_tuning) {
        return {hth, nth, mth};
    }

    int count = 0;
    for (const auto& detection : detections) {
        if (detection.conf > config_.cth) {
            ++count;
        }
    }
    if (count < 1) {
        count = 1;
    }

    const double lnc = std::log10(static_cast<double>(count));
    hth = static_cast<float>(Clamp(hth - (config_.high_th_m * lnc), 0.0, 1.0));
    nth = static_cast<float>(Clamp(nth + (config_.new_track_th_m * lnc), hth, 1.0));
    mth = static_cast<float>(Clamp(mth - (config_.match_th_first_m * lnc), 0.0, 0.67));
    return {hth, nth, mth};
}

void SFSORTTracker::MaybeSetMargins(const int frame_width, const int frame_height) {
    if (frame_width <= 0 || frame_height <= 0) {
        return;
    }

    l_margin_ = static_cast<double>(Clamp(static_cast<double>(config_.horizontal_margin), 0.0, frame_width));
    r_margin_ = static_cast<double>(Clamp(static_cast<double>(frame_width - config_.horizontal_margin), 0.0, frame_width));
    t_margin_ = static_cast<double>(Clamp(static_cast<double>(config_.vertical_margin), 0.0, frame_height));
    b_margin_ = static_cast<double>(Clamp(static_cast<double>(frame_height - config_.vertical_margin), 0.0, frame_height));
    margins_ready_ = true;
}

void SFSORTTracker::PurgeStaleLostTracks() {
    lost_tracks_.erase(
        std::remove_if(
            lost_tracks_.begin(),
            lost_tracks_.end(),
            [this](const TrackData& track) {
                const int age = frame_count_ - track.last_frame;
                if (track.state == TrackState::kLostCentral) {
                    return age > config_.central_timeout;
                }
                return age > config_.marginal_timeout;
            }
        ),
        lost_tracks_.end()
    );
}

void SFSORTTracker::UpdateLostTracks(const std::vector<TrackData>& next_lost_tracks) {
    for (auto track : next_lost_tracks) {
        track.time_since_update = std::max(0, frame_count_ - track.last_frame);
        if (ContainsTrackId(lost_tracks_, track.track_id)) {
            continue;
        }

        const double u = TrackCenterX(track);
        const double v = TrackCenterY(track);
        if ((l_margin_ < u && u < r_margin_) && (t_margin_ < v && v < b_margin_)) {
            track.state = TrackState::kLostCentral;
        } else {
            track.state = TrackState::kLostMarginal;
        }
        lost_tracks_.push_back(std::move(track));
    }
}

SFSORTTracker::TrackData SFSORTTracker::NewTrack(const Detection& detection) const {
    TrackData track;
    track.is_obb = detection.is_obb;
    track.last_frame = frame_count_;
    track.track_id = id_counter_;
    track.conf = detection.conf;
    track.cls = detection.cls;
    track.det_ind = detection.det_ind;
    if (detection.is_obb) {
        track.xywha = detection.xywha;
        track.xyxy = ObbToXyxy(track.xywha);
    } else {
        track.xyxy = detection.xyxy;
    }
    return track;
}

TrackOutput SFSORTTracker::FormatTrack(const TrackData& track) {
    TrackOutput output;
    output.is_obb = track.is_obb;
    output.xyxy = track.xyxy;
    output.xywha = track.xywha;
    output.id = track.track_id;
    output.conf = track.conf;
    output.cls = track.cls;
    output.det_ind = track.det_ind;
    return output;
}

Eigen::MatrixXd SFSORTTracker::CalculateCost(
    const std::vector<TrackData*>& tracks,
    const std::vector<Detection>& detections,
    const bool iou_only
) {
    if (tracks.empty() || detections.empty()) {
        return Eigen::MatrixXd(static_cast<int>(tracks.size()), static_cast<int>(detections.size()));
    }
    if (detections.front().is_obb) {
        return CalculateObbCost(tracks, detections, iou_only);
    }
    return CalculateAabbCost(tracks, detections, iou_only);
}

std::vector<TrackOutput> SFSORTTracker::Update(const std::vector<Detection>& detections, const cv::Mat& image) {
    if (!margins_ready_) {
        const int frame_width = config_.frame_width > 0 ? config_.frame_width : image.cols;
        const int frame_height = config_.frame_height > 0 ? config_.frame_height : image.rows;
        MaybeSetMargins(frame_width, frame_height);
    }

    if (!detections.empty()) {
        const bool det_is_obb = detections.front().is_obb;
        if (!detection_mode_ready_) {
            detection_mode_ready_ = true;
            is_obb_mode_ = det_is_obb;
        } else if (det_is_obb != is_obb_mode_) {
            throw std::runtime_error("Native SFSORT cannot switch between AABB and OBB detections after initialization.");
        }
    }

    ++frame_count_;

    const auto [hth, nth, mth] = DynamicThresholds(detections);

    std::vector<Detection> definite_detections;
    std::vector<Detection> possible_detections;
    definite_detections.reserve(detections.size());
    possible_detections.reserve(detections.size());
    for (const auto& detection : detections) {
        if (detection.conf > hth) {
            definite_detections.push_back(detection);
        }
        if (detection.conf > config_.low_th && detection.conf < hth) {
            possible_detections.push_back(detection);
        }
    }

    std::vector<TrackData> next_active_tracks;
    PurgeStaleLostTracks();

    std::vector<TrackData*> track_pool;
    track_pool.reserve(active_tracks_.size() + lost_tracks_.size());
    for (auto& track : active_tracks_) {
        track_pool.push_back(&track);
    }
    for (auto& track : lost_tracks_) {
        track_pool.push_back(&track);
    }

    std::unordered_set<int> matched_lost_ids;
    std::vector<TrackData*> unmatched_track_pool;

    if (!definite_detections.empty()) {
        if (!track_pool.empty()) {
            const AssignmentResult first_matches = LinearAssignment(
                CalculateCost(track_pool, definite_detections, false),
                static_cast<double>(mth)
            );
            for (const auto& match : first_matches.matches) {
                auto* track = track_pool[static_cast<std::size_t>(match.first)];
                track->Update(definite_detections[static_cast<std::size_t>(match.second)], frame_count_, config_.obb_theta_damping);
                next_active_tracks.push_back(*track);
                if (ContainsTrackId(lost_tracks_, track->track_id)) {
                    matched_lost_ids.insert(track->track_id);
                }
            }

            for (const int unmatched_index : first_matches.unmatched_rows) {
                unmatched_track_pool.push_back(track_pool[static_cast<std::size_t>(unmatched_index)]);
            }
            for (const int unmatched_index : first_matches.unmatched_cols) {
                const Detection& detection = definite_detections[static_cast<std::size_t>(unmatched_index)];
                if (detection.conf > nth) {
                    next_active_tracks.push_back(NewTrack(detection));
                    ++id_counter_;
                }
            }
        } else {
            for (const auto& detection : definite_detections) {
                if (detection.conf > nth) {
                    next_active_tracks.push_back(NewTrack(detection));
                    ++id_counter_;
                }
            }
        }
    }

    std::vector<TrackData> next_lost_tracks;
    next_lost_tracks.reserve(unmatched_track_pool.size());
    for (auto* track : unmatched_track_pool) {
        next_lost_tracks.push_back(*track);
    }

    if (!possible_detections.empty() && !unmatched_track_pool.empty()) {
        const AssignmentResult second_matches = LinearAssignment(
            CalculateCost(unmatched_track_pool, possible_detections, true),
            static_cast<double>(config_.match_th_second)
        );
        for (const auto& match : second_matches.matches) {
            auto* track = unmatched_track_pool[static_cast<std::size_t>(match.first)];
            track->Update(possible_detections[static_cast<std::size_t>(match.second)], frame_count_, config_.obb_theta_damping);
            next_active_tracks.push_back(*track);
            if (ContainsTrackId(lost_tracks_, track->track_id)) {
                matched_lost_ids.insert(track->track_id);
            }
            RemoveTrackById(next_lost_tracks, track->track_id);
        }
    }

    if (definite_detections.empty() && possible_detections.empty()) {
        next_lost_tracks.clear();
        next_lost_tracks.reserve(track_pool.size());
        for (auto* track : track_pool) {
            next_lost_tracks.push_back(*track);
        }
    }

    if (!matched_lost_ids.empty()) {
        RemoveTracksById(lost_tracks_, matched_lost_ids);
    }
    UpdateLostTracks(next_lost_tracks);
    active_tracks_ = next_active_tracks;

    std::vector<TrackOutput> outputs;
    outputs.reserve(next_active_tracks.size());
    for (const auto& track : next_active_tracks) {
        outputs.push_back(FormatTrack(track));
    }
    return outputs;
}

}  // namespace sfsort
