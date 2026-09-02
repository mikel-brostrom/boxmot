#include "bytetrack/tracker.hpp"

#include "boxmot/trackers/base/assignment.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace bytetrack {

using boxmot::trackers::base::AssignmentResult;
using boxmot::trackers::base::LinearAssignment;

namespace {

double PairwiseSimilarity(
    const Track::Ptr& lhs,
    const Track::Ptr& rhs,
    const boxmot::trackers::base::AssociationMode mode,
    const int frame_width = 0,
    const int frame_height = 0
) {
    if (lhs->UsesObb() || rhs->UsesObb()) {
        return boxmot::trackers::base::ObbAssociationSimilarity(
            lhs->xywha(), rhs->xywha(), mode, frame_width, frame_height
        );
    }
    return boxmot::trackers::base::AabbAssociationSimilarity(
        lhs->xyxy(), rhs->xyxy(), mode, frame_width, frame_height
    );
}

Eigen::MatrixXd AssociationDistance(
    const std::vector<Track::Ptr>& tracks,
    const std::vector<Track::Ptr>& detections,
    const boxmot::trackers::base::AssociationMode mode,
    const int frame_width = 0,
    const int frame_height = 0
) {
    Eigen::MatrixXd cost(static_cast<int>(tracks.size()), static_cast<int>(detections.size()));
    if (tracks.empty() || detections.empty()) {
        return cost;
    }
    for (int row = 0; row < static_cast<int>(tracks.size()); ++row) {
        for (int col = 0; col < static_cast<int>(detections.size()); ++col) {
            cost(row, col) = 1.0 - PairwiseSimilarity(
                tracks[row], detections[col], mode, frame_width, frame_height
            );
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
    const Eigen::MatrixXd distances = AssociationDistance(
        lhs,
        rhs,
        boxmot::trackers::base::AssociationMode::kIou
    );
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

ByteTrackTracker::ByteTrackTracker(Config config)
    : config_(std::move(config)),
      association_mode_(boxmot::trackers::base::ParseAssociationMode(config_.asso_func)),
      max_time_lost_(static_cast<int>((static_cast<double>(config_.frame_rate) / 30.0) * static_cast<double>(config_.track_buffer))) {
    Track::ResetCount();
    if (max_time_lost_ <= 0) {
        max_time_lost_ = config_.track_buffer;
    }
}

void ByteTrackTracker::Reset() {
    frame_count_ = 0;
    max_time_lost_ = static_cast<int>((static_cast<double>(config_.frame_rate) / 30.0) * static_cast<double>(config_.track_buffer));
    if (max_time_lost_ <= 0) {
        max_time_lost_ = config_.track_buffer;
    }
    detection_mode_ready_ = false;
    is_obb_mode_ = false;
    association_frame_width_ = 0;
    association_frame_height_ = 0;
    Track::ResetCount();
    active_tracks_.clear();
    lost_tracks_.clear();
    removed_tracks_.clear();
}

std::vector<Track::Ptr> ByteTrackTracker::CreateDetectionTracks(const std::vector<Detection>& detections) const {
    std::vector<Track::Ptr> result;
    result.reserve(detections.size());
    for (const auto& detection : detections) {
        result.push_back(std::make_shared<Track>(detection));
    }
    return result;
}

std::pair<std::vector<Track::Ptr>, std::vector<Track::Ptr>> ByteTrackTracker::SeparateTracks() const {
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

void ByteTrackTracker::UpdateTrackStates(std::vector<Track::Ptr>& removed_tracks) {
    for (const auto& track : lost_tracks_) {
        if ((frame_count_ - track->frame_id) > max_time_lost_) {
            track->state = TrackState::kRemoved;
            removed_tracks.push_back(track);
        }
    }
}

std::vector<TrackOutput> ByteTrackTracker::PrepareOutput(
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

std::vector<TrackOutput> ByteTrackTracker::Update(const std::vector<Detection>& detections, const cv::Mat& image) {
    if (boxmot::trackers::base::AssociationModeRequiresFrameDimensions(association_mode_)
        && (association_frame_width_ <= 0 || association_frame_height_ <= 0)) {
        if (image.empty()) {
            throw std::runtime_error("Native ByteTrack requires an image to initialize centroid association.");
        }
        association_frame_width_ = image.cols;
        association_frame_height_ = image.rows;
    }
    if (!detections.empty()) {
        const bool det_is_obb = detections.front().is_obb;
        if (!detection_mode_ready_) {
            detection_mode_ready_ = true;
            is_obb_mode_ = det_is_obb;
            boxmot::trackers::base::ValidateAssociationModeForDetections(association_mode_, det_is_obb);
        } else if (det_is_obb != is_obb_mode_) {
            throw std::runtime_error("Native ByteTrack cannot switch between AABB and OBB detections after initialization.");
        }
    }

    ++frame_count_;

    std::vector<Detection> detections_first_raw;
    std::vector<Detection> detections_second_raw;
    detections_first_raw.reserve(detections.size());
    detections_second_raw.reserve(detections.size());
    for (const auto& detection : detections) {
        if (detection.conf > config_.track_thresh) {
            detections_first_raw.push_back(detection);
        }
        if (detection.conf > config_.min_conf && detection.conf < config_.track_thresh) {
            detections_second_raw.push_back(detection);
        }
    }

    std::vector<Track::Ptr> detections_first = CreateDetectionTracks(detections_first_raw);
    std::vector<Track::Ptr> detections_second = CreateDetectionTracks(detections_second_raw);

    auto [unconfirmed, tracked_tracks] = SeparateTracks();
    std::vector<Track::Ptr> activated_tracks;
    std::vector<Track::Ptr> refind_tracks;
    std::vector<Track::Ptr> lost_tracks;
    std::vector<Track::Ptr> removed_tracks;

    std::vector<Track::Ptr> strack_pool = JointTracks(tracked_tracks, lost_tracks_);
    for (auto& track : strack_pool) {
        if (is_obb_mode_) {
            track->Predict(kalman_filter_obb_);
        } else {
            track->Predict(kalman_filter_);
        }
    }

    Eigen::MatrixXd dist_first = FuseScore(
        AssociationDistance(
            strack_pool,
            detections_first,
            association_mode_,
            association_frame_width_,
            association_frame_height_
        ),
        detections_first
    );
    const AssignmentResult first_matches = LinearAssignment(dist_first, config_.match_thresh);
    for (const auto& match : first_matches.matches) {
        const auto& track = strack_pool[match.first];
        const auto& detection = detections_first[match.second];
        if (track->state == TrackState::kTracked) {
            if (is_obb_mode_) {
                track->Update(*detection, kalman_filter_obb_, frame_count_);
            } else {
                track->Update(*detection, kalman_filter_, frame_count_);
            }
            activated_tracks.push_back(track);
        } else {
            if (is_obb_mode_) {
                track->ReActivate(*detection, kalman_filter_obb_, frame_count_, false);
            } else {
                track->ReActivate(*detection, kalman_filter_, frame_count_, false);
            }
            refind_tracks.push_back(track);
        }
    }

    std::vector<Track::Ptr> remaining_tracked;
    for (const int index : first_matches.unmatched_rows) {
        if (strack_pool[index]->state == TrackState::kTracked) {
            remaining_tracked.push_back(strack_pool[index]);
        }
    }

    const AssignmentResult second_matches = LinearAssignment(
        AssociationDistance(
            remaining_tracked,
            detections_second,
            association_mode_,
            association_frame_width_,
            association_frame_height_
        ),
        0.5F
    );
    for (const auto& match : second_matches.matches) {
        const auto& track = remaining_tracked[match.first];
        const auto& detection = detections_second[match.second];
        if (track->state == TrackState::kTracked) {
            if (is_obb_mode_) {
                track->Update(*detection, kalman_filter_obb_, frame_count_);
            } else {
                track->Update(*detection, kalman_filter_, frame_count_);
            }
            activated_tracks.push_back(track);
        } else {
            if (is_obb_mode_) {
                track->ReActivate(*detection, kalman_filter_obb_, frame_count_, false);
            } else {
                track->ReActivate(*detection, kalman_filter_, frame_count_, false);
            }
            refind_tracks.push_back(track);
        }
    }
    for (const int index : second_matches.unmatched_rows) {
        const auto& track = remaining_tracked[index];
        if (track->state != TrackState::kLost) {
            track->state = TrackState::kLost;
            lost_tracks.push_back(track);
        }
    }

    std::vector<Track::Ptr> remaining_high;
    remaining_high.reserve(first_matches.unmatched_cols.size());
    for (const int index : first_matches.unmatched_cols) {
        remaining_high.push_back(detections_first[index]);
    }

    const AssignmentResult unconfirmed_matches = LinearAssignment(
        FuseScore(
            AssociationDistance(
                unconfirmed,
                remaining_high,
                association_mode_,
                association_frame_width_,
                association_frame_height_
            ),
            remaining_high
        ),
        0.7F
    );
    for (const auto& match : unconfirmed_matches.matches) {
        const auto& track = unconfirmed[match.first];
        if (is_obb_mode_) {
            track->Update(*remaining_high[match.second], kalman_filter_obb_, frame_count_);
        } else {
            track->Update(*remaining_high[match.second], kalman_filter_, frame_count_);
        }
        activated_tracks.push_back(track);
    }
    for (const int index : unconfirmed_matches.unmatched_rows) {
        const auto& track = unconfirmed[index];
        track->state = TrackState::kRemoved;
        removed_tracks.push_back(track);
    }
    for (const int index : unconfirmed_matches.unmatched_cols) {
        const auto& track = remaining_high[index];
        if (track->conf < config_.track_thresh) {
            continue;
        }
        if (is_obb_mode_) {
            track->Activate(kalman_filter_obb_, frame_count_);
        } else {
            track->Activate(kalman_filter_, frame_count_);
        }
        activated_tracks.push_back(track);
    }

    UpdateTrackStates(removed_tracks);
    return PrepareOutput(activated_tracks, refind_tracks, lost_tracks, removed_tracks);
}

}  // namespace bytetrack
