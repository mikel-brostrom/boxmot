#include "botsort/tracker.hpp"

#include "boxmot/trackers/base/assignment.hpp"

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace botsort {

using boxmot::trackers::base::AssignmentResult;
using boxmot::trackers::base::LinearAssignment;

namespace {

constexpr double kPi = 3.14159265358979323846;

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

double PairwiseIoU(const Track::Ptr& lhs, const Track::Ptr& rhs) {
    if (lhs->UsesObb() || rhs->UsesObb()) {
        return ObbIoU(lhs->xywha(), rhs->xywha());
    }
    return AabbIoU(lhs->xyxy(), rhs->xyxy());
}

Eigen::MatrixXd IouDistance(const std::vector<Track::Ptr>& tracks, const std::vector<Track::Ptr>& detections) {
    Eigen::MatrixXd cost(static_cast<int>(tracks.size()), static_cast<int>(detections.size()));
    if (tracks.empty() || detections.empty()) {
        return cost;
    }
    for (int row = 0; row < static_cast<int>(tracks.size()); ++row) {
        for (int col = 0; col < static_cast<int>(detections.size()); ++col) {
            cost(row, col) = 1.0 - PairwiseIoU(tracks[row], detections[col]);
        }
    }
    return cost;
}

Eigen::MatrixXd EmbeddingDistance(const std::vector<Track::Ptr>& tracks, const std::vector<Track::Ptr>& detections) {
    Eigen::MatrixXd cost = Eigen::MatrixXd::Ones(static_cast<int>(tracks.size()), static_cast<int>(detections.size()));
    if (tracks.empty() || detections.empty()) {
        return cost;
    }
    for (int row = 0; row < static_cast<int>(tracks.size()); ++row) {
        if (!tracks[row]->HasSmoothFeature()) {
            continue;
        }
        for (int col = 0; col < static_cast<int>(detections.size()); ++col) {
            if (!detections[col]->curr_feat().size()) {
                continue;
            }
            if (tracks[row]->smooth_feat().size() != detections[col]->curr_feat().size()) {
                continue;
            }
            const double similarity = static_cast<double>(tracks[row]->smooth_feat().dot(detections[col]->curr_feat()));
            cost(row, col) = std::max(0.0, 1.0 - similarity);
        }
    }
    return cost;
}

Eigen::MatrixXd FuseScore(Eigen::MatrixXd cost_matrix, const std::vector<Track::Ptr>& detections) {
    if (cost_matrix.size() == 0) {
        return cost_matrix;
    }
    for (int col = 0; col < static_cast<int>(detections.size()); ++col) {
        const double conf = static_cast<double>(detections[col]->conf);
        for (int row = 0; row < cost_matrix.rows(); ++row) {
            const double iou_similarity = 1.0 - cost_matrix(row, col);
            cost_matrix(row, col) = 1.0 - (iou_similarity * conf);
        }
    }
    return cost_matrix;
}

std::vector<Track::Ptr> JointTracks(const std::vector<Track::Ptr>& lhs, const std::vector<Track::Ptr>& rhs) {
    std::vector<Track::Ptr> result;
    result.reserve(lhs.size() + rhs.size());
    std::unordered_set<int> seen;
    for (const auto& track : lhs) {
        seen.insert(track->id);
        result.push_back(track);
    }
    for (const auto& track : rhs) {
        if (seen.insert(track->id).second) {
            result.push_back(track);
        }
    }
    return result;
}

std::vector<Track::Ptr> SubTracks(const std::vector<Track::Ptr>& lhs, const std::vector<Track::Ptr>& rhs) {
    std::unordered_set<int> remove_ids;
    for (const auto& track : rhs) {
        remove_ids.insert(track->id);
    }
    std::vector<Track::Ptr> result;
    result.reserve(lhs.size());
    for (const auto& track : lhs) {
        if (remove_ids.count(track->id) == 0) {
            result.push_back(track);
        }
    }
    return result;
}

std::pair<std::vector<Track::Ptr>, std::vector<Track::Ptr>> RemoveDuplicateTracks(
    const std::vector<Track::Ptr>& lhs,
    const std::vector<Track::Ptr>& rhs
) {
    const Eigen::MatrixXd distances = IouDistance(lhs, rhs);
    std::unordered_set<int> dup_lhs;
    std::unordered_set<int> dup_rhs;
    for (int row = 0; row < distances.rows(); ++row) {
        for (int col = 0; col < distances.cols(); ++col) {
            if (distances(row, col) < 0.15) {
                const int lhs_time = lhs[row]->frame_id - lhs[row]->start_frame;
                const int rhs_time = rhs[col]->frame_id - rhs[col]->start_frame;
                if (lhs_time > rhs_time) {
                    dup_rhs.insert(col);
                } else {
                    dup_lhs.insert(row);
                }
            }
        }
    }

    std::vector<Track::Ptr> filtered_lhs;
    std::vector<Track::Ptr> filtered_rhs;
    for (int index = 0; index < static_cast<int>(lhs.size()); ++index) {
        if (dup_lhs.count(index) == 0) {
            filtered_lhs.push_back(lhs[index]);
        }
    }
    for (int index = 0; index < static_cast<int>(rhs.size()); ++index) {
        if (dup_rhs.count(index) == 0) {
            filtered_rhs.push_back(rhs[index]);
        }
    }
    return {filtered_lhs, filtered_rhs};
}

}  // namespace

BotSortTracker::BotSortTracker(Config config)
    : config_(std::move(config)),
      max_time_lost_(static_cast<int>((static_cast<double>(config_.frame_rate) / 30.0) * static_cast<double>(config_.track_buffer))),
      cmc_(CreateCameraMotionCompensator(config_.cmc_method)) {
    Track::ResetCount();
    if (max_time_lost_ <= 0) {
        max_time_lost_ = config_.track_buffer;
    }
    if (config_.with_reid && !config_.reid_model_path.empty()) {
        reid_model_ = MaybeCreateOnnxReIdModel(config_.reid_model_path, config_.reid_preprocess);
    }
}

void BotSortTracker::Reset() {
    frame_count_ = 0;
    max_time_lost_ = static_cast<int>((static_cast<double>(config_.frame_rate) / 30.0) * static_cast<double>(config_.track_buffer));
    if (max_time_lost_ <= 0) {
        max_time_lost_ = config_.track_buffer;
    }
    detection_mode_ready_ = false;
    is_obb_mode_ = false;
    kalman_filter_ = KalmanFilterXYWH(4);
    Track::ResetCount();
    active_tracks_.clear();
    lost_tracks_.clear();
    removed_tracks_.clear();
    cmc_ = CreateCameraMotionCompensator(config_.cmc_method);
    reid_model_.reset();
    if (config_.with_reid && !config_.reid_model_path.empty()) {
        reid_model_ = MaybeCreateOnnxReIdModel(config_.reid_model_path, config_.reid_preprocess);
    }
}

std::vector<Track::Ptr> BotSortTracker::CreateDetectionTracks(const std::vector<Detection>& detections) const {
    std::vector<Track::Ptr> result;
    result.reserve(detections.size());
    for (const auto& detection : detections) {
        result.push_back(std::make_shared<Track>(detection));
    }
    return result;
}

std::pair<std::vector<Track::Ptr>, std::vector<Track::Ptr>> BotSortTracker::SeparateTracks() const {
    std::vector<Track::Ptr> unconfirmed;
    std::vector<Track::Ptr> active;
    for (const auto& track : active_tracks_) {
        if (track->is_activated) {
            active.push_back(track);
        } else {
            unconfirmed.push_back(track);
        }
    }
    return {unconfirmed, active};
}

void BotSortTracker::ApplyCameraMotionCompensation(
    const cv::Mat& image,
    const std::vector<Detection>& detections,
    std::vector<Track::Ptr>& strack_pool,
    std::vector<Track::Ptr>& unconfirmed
) {
    if (!cmc_) {
        return;
    }
    const cv::Mat warp = cmc_->Apply(image, detections);
    if (warp.empty() || warp.rows != 2 || warp.cols != 3) {
        return;
    }

    Eigen::Matrix2d linear;
    linear << warp.at<float>(0, 0), warp.at<float>(0, 1),
        warp.at<float>(1, 0), warp.at<float>(1, 1);
    Eigen::Vector2d translation(warp.at<float>(0, 2), warp.at<float>(1, 2));

    for (auto& track : strack_pool) {
        track->ApplyAffine(linear, translation);
    }
    for (auto& track : unconfirmed) {
        track->ApplyAffine(linear, translation);
    }
}

void BotSortTracker::UpdateTrackStates(std::vector<Track::Ptr>& removed_tracks) {
    for (const auto& track : lost_tracks_) {
        if ((frame_count_ - track->frame_id) > max_time_lost_) {
            track->MarkRemoved();
            removed_tracks.push_back(track);
        }
    }
}

std::vector<TrackOutput> BotSortTracker::PrepareOutput(
    const std::vector<Track::Ptr>& activated_tracks,
    const std::vector<Track::Ptr>& refind_tracks,
    const std::vector<Track::Ptr>& lost_tracks,
    const std::vector<Track::Ptr>& removed_tracks
) {
    std::vector<Track::Ptr> tracked_only;
    for (const auto& track : active_tracks_) {
        if (track->state == TrackState::kTracked) {
            tracked_only.push_back(track);
        }
    }
    active_tracks_ = JointTracks(tracked_only, activated_tracks);
    active_tracks_ = JointTracks(active_tracks_, refind_tracks);
    lost_tracks_ = SubTracks(lost_tracks_, active_tracks_);

    lost_tracks_.insert(lost_tracks_.end(), lost_tracks.begin(), lost_tracks.end());
    lost_tracks_ = SubTracks(lost_tracks_, removed_tracks_);
    removed_tracks_.insert(removed_tracks_.end(), removed_tracks.begin(), removed_tracks.end());

    auto [dedup_active, dedup_lost] = RemoveDuplicateTracks(active_tracks_, lost_tracks_);
    active_tracks_ = std::move(dedup_active);
    lost_tracks_ = std::move(dedup_lost);

    std::vector<TrackOutput> outputs;
    for (const auto& track : active_tracks_) {
        if (!track->is_activated) {
            continue;
        }
        TrackOutput output;
        output.is_obb = track->UsesObb();
        output.id = track->id;
        output.xyxy = track->xyxy();
        output.xywha = track->xywha();
        output.conf = track->conf;
        output.cls = track->cls;
        output.det_ind = track->det_ind;
        outputs.push_back(output);
    }
    return outputs;
}

std::vector<TrackOutput> BotSortTracker::Update(const std::vector<Detection>& detections, const cv::Mat& image) {
    if (!detections.empty()) {
        const bool det_is_obb = detections.front().is_obb;
        if (!detection_mode_ready_) {
            detection_mode_ready_ = true;
            is_obb_mode_ = det_is_obb;
            kalman_filter_ = KalmanFilterXYWH(det_is_obb ? 5 : 4);
        } else if (det_is_obb != is_obb_mode_) {
            throw std::runtime_error("Native BoTSORT cannot switch between AABB and OBB detections after initialization.");
        }
    }

    ++frame_count_;
    last_reid_time_ms_ = 0.0;

    std::vector<Detection> working_detections = detections;
    if (config_.with_reid && reid_model_.has_value()) {
        bool needs_embeddings = false;
        for (const auto& detection : working_detections) {
            if (!detection.has_embedding()) {
                needs_embeddings = true;
                break;
            }
        }
        if (needs_embeddings) {
            const auto reid_started = std::chrono::steady_clock::now();
            const std::vector<Eigen::VectorXf> features = reid_model_->GetFeatures(working_detections, image);
            const auto reid_finished = std::chrono::steady_clock::now();
            last_reid_time_ms_ = std::chrono::duration<double, std::milli>(reid_finished - reid_started).count();
            if (features.size() != working_detections.size()) {
                throw std::runtime_error("Native ReID returned a different number of embeddings than detections.");
            }
            for (std::size_t index = 0; index < working_detections.size(); ++index) {
                working_detections[index].embedding = features[index];
            }
        }
    }

    std::vector<Detection> detections_first_raw;
    std::vector<Detection> detections_second_raw;
    detections_first_raw.reserve(working_detections.size());
    detections_second_raw.reserve(working_detections.size());
    for (const auto& detection : working_detections) {
        if (detection.conf > config_.track_high_thresh) {
            detections_first_raw.push_back(detection);
        }
        if (detection.conf > config_.track_low_thresh && detection.conf < config_.track_high_thresh) {
            detections_second_raw.push_back(detection);
        }
    }

    const std::vector<Track::Ptr> detections_first = CreateDetectionTracks(detections_first_raw);
    const std::vector<Track::Ptr> detections_second = CreateDetectionTracks(detections_second_raw);

    auto [unconfirmed, active_tracks] = SeparateTracks();
    std::vector<Track::Ptr> activated_tracks;
    std::vector<Track::Ptr> refind_tracks;
    std::vector<Track::Ptr> lost_tracks;
    std::vector<Track::Ptr> removed_tracks;

    std::vector<Track::Ptr> strack_pool = JointTracks(active_tracks, lost_tracks_);
    for (auto& track : strack_pool) {
        track->Predict(kalman_filter_);
    }
    ApplyCameraMotionCompensation(image, working_detections, strack_pool, unconfirmed);

    const Eigen::MatrixXd iou_first = IouDistance(strack_pool, detections_first);
    Eigen::MatrixXd dist_first = config_.fuse_first_associate ? FuseScore(iou_first, detections_first) : iou_first;

    if (config_.with_reid && dist_first.size() > 0) {
        Eigen::MatrixXd emb_first = EmbeddingDistance(strack_pool, detections_first);
        for (int row = 0; row < emb_first.rows(); ++row) {
            for (int col = 0; col < emb_first.cols(); ++col) {
                if (emb_first(row, col) > config_.appearance_thresh || iou_first(row, col) > config_.proximity_thresh) {
                    emb_first(row, col) = 1.0;
                }
                dist_first(row, col) = std::min(dist_first(row, col), emb_first(row, col));
            }
        }
    }

    const AssignmentResult first_matches = LinearAssignment(dist_first, config_.match_thresh);
    for (const auto& match : first_matches.matches) {
        const auto& track = strack_pool[match.first];
        const auto& detection = detections_first[match.second];
        if (track->state == TrackState::kTracked) {
            track->Update(*detection, kalman_filter_, frame_count_);
            activated_tracks.push_back(track);
        } else {
            track->ReActivate(*detection, kalman_filter_, frame_count_, false);
            refind_tracks.push_back(track);
        }
    }

    std::vector<Track::Ptr> remaining_tracked;
    for (const int index : first_matches.unmatched_rows) {
        if (strack_pool[index]->state == TrackState::kTracked) {
            remaining_tracked.push_back(strack_pool[index]);
        }
    }

    const Eigen::MatrixXd second_dist = IouDistance(remaining_tracked, detections_second);
    const AssignmentResult second_matches = LinearAssignment(second_dist, 0.5F);
    for (const auto& match : second_matches.matches) {
        const auto& track = remaining_tracked[match.first];
        const auto& detection = detections_second[match.second];
        if (track->state == TrackState::kTracked) {
            track->Update(*detection, kalman_filter_, frame_count_);
            activated_tracks.push_back(track);
        } else {
            track->ReActivate(*detection, kalman_filter_, frame_count_, false);
            refind_tracks.push_back(track);
        }
    }
    for (const int index : second_matches.unmatched_rows) {
        const auto& track = remaining_tracked[index];
        if (track->state != TrackState::kLost) {
            track->MarkLost();
            lost_tracks.push_back(track);
        }
    }

    std::vector<Track::Ptr> remaining_high;
    remaining_high.reserve(first_matches.unmatched_cols.size());
    for (const int index : first_matches.unmatched_cols) {
        remaining_high.push_back(detections_first[index]);
    }

    const Eigen::MatrixXd iou_unc = IouDistance(unconfirmed, remaining_high);
    Eigen::MatrixXd dist_unc = FuseScore(iou_unc, remaining_high);
    if (config_.with_reid && dist_unc.size() > 0) {
        Eigen::MatrixXd emb_unc = EmbeddingDistance(unconfirmed, remaining_high);
        emb_unc /= 2.0;
        for (int row = 0; row < emb_unc.rows(); ++row) {
            for (int col = 0; col < emb_unc.cols(); ++col) {
                if (emb_unc(row, col) > config_.appearance_thresh || iou_unc(row, col) > config_.proximity_thresh) {
                    emb_unc(row, col) = 1.0;
                }
                dist_unc(row, col) = std::min(dist_unc(row, col), emb_unc(row, col));
            }
        }
    }

    const AssignmentResult unconfirmed_matches = LinearAssignment(dist_unc, 0.7F);
    for (const auto& match : unconfirmed_matches.matches) {
        const auto& track = unconfirmed[match.first];
        track->Update(*remaining_high[match.second], kalman_filter_, frame_count_);
        activated_tracks.push_back(track);
    }
    for (const int index : unconfirmed_matches.unmatched_rows) {
        const auto& track = unconfirmed[index];
        track->MarkRemoved();
        removed_tracks.push_back(track);
    }
    for (const int index : unconfirmed_matches.unmatched_cols) {
        const auto& track = remaining_high[index];
        if (track->conf < config_.new_track_thresh) {
            continue;
        }
        track->Activate(kalman_filter_, frame_count_);
        activated_tracks.push_back(track);
    }

    UpdateTrackStates(removed_tracks);
    return PrepareOutput(activated_tracks, refind_tracks, lost_tracks, removed_tracks);
}

}  // namespace botsort
