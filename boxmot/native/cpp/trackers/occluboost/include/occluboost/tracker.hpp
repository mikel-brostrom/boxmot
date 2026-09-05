#pragma once

#include "boxmot/trackers/base/association.hpp"
#include "boxmot/trackers/base/base_tracker.hpp"
#include "occluboost/cmc.hpp"
#include "occluboost/track.hpp"
#include "occluboost/types.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace occluboost {

class OccluBoostTracker final : public boxmot::trackers::base::TrackerBase<Detection, TrackOutput> {
public:
    explicit OccluBoostTracker(Config config);

    std::vector<TrackOutput> Update(const std::vector<Detection>& detections,
                                    const cv::Mat& image) override;
    void Reset() override;
    [[nodiscard]] bool SupportsObb() const noexcept override { return true; }
    [[nodiscard]] bool SupportsEmbeddings() const noexcept override { return true; }

private:
    // OBB-only update branch (mirrors Python OccluBoost._update_obb).
    std::vector<TrackOutput> UpdateObb(const std::vector<Detection>& detections,
                                       const cv::Mat& image);

    // Confidence boosting (use_dlo_boost / use_duo_boost branches).
    void DloConfidenceBoost(std::vector<Detection>& detections) const;
    void DuoConfidenceBoost(std::vector<Detection>& detections) const;
    void DloConfidenceBoostObb(std::vector<Detection>& detections) const;
    void DuoConfidenceBoostObb(std::vector<Detection>& detections) const;

    // Mahalanobis distance matrix (Nd x Nt).
    Eigen::MatrixXd GetMhDistMatrix(const std::vector<Detection>& detections) const;
    Eigen::MatrixXd GetMhDistMatrixObb(const std::vector<Detection>& detections) const;

    // OccluTrack abnormal-motion suppression coefficient.
    double ComputeAmsAlpha(KalmanBoxTracker& trk, const Eigen::Vector4d& det_xyxy) const;
    void AmsUpdate(KalmanBoxTracker& trk, const Detection& det);
    void MaybeActivate(KalmanBoxTracker& trk) const;

    // Drop younger of any pair of duplicate emissions sharing IoU >= threshold.
    void SuppressDuplicateEmissions(
        std::vector<std::pair<KalmanBoxTracker::Ptr, Eigen::Vector4d>>& emitted);

    // Output filter: aspect/area gating.
    bool PassesFilter(const Eigen::Vector4d& xyxy) const;
    bool PassesObbFilter(const Eigen::Matrix<double, 5, 1>& xywha) const;

    Config config_;
    boxmot::trackers::base::AssociationMode association_mode_;
    int association_frame_width_ = 0;
    int association_frame_height_ = 0;
    int frame_count_ = 0;
    std::int64_t next_track_id_ = 1;
    std::vector<KalmanBoxTracker::Ptr> trackers_;
    std::unique_ptr<CameraMotionCompensator> cmc_;
    // Detection-mode latch (AABB vs OBB) determined from first non-empty frame.
    bool detection_mode_ready_ = false;
    bool is_obb_mode_ = false;
};

}  // namespace occluboost
