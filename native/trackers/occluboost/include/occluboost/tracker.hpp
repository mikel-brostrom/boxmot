#pragma once

#include "boxmot/trackers/base/base_tracker.hpp"
#include "occluboost/cmc.hpp"
#include "occluboost/reid_onnx.hpp"
#include "occluboost/track.hpp"
#include "occluboost/types.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace occluboost {

class OccluBoostTracker final
    : public boxmot::trackers::base::TrackerBase<Detection, TrackOutput> {
public:
    explicit OccluBoostTracker(Config config);

    std::vector<TrackOutput> Update(const std::vector<Detection>& detections, const cv::Mat& image) override;
    void Reset() override;
    [[nodiscard]] double LastReIdTimeMs() const noexcept { return last_reid_time_ms_; }

    [[nodiscard]] bool SupportsObb() const noexcept override { return false; }
    [[nodiscard]] bool SupportsReId() const noexcept override { return true; }

private:
    // Detection prep
    std::vector<Detection> EnsureEmbeddings(std::vector<Detection> detections, const cv::Mat& image);

    // Confidence boosting (use_dlo_boost / use_duo_boost branches).
    void DloConfidenceBoost(std::vector<Detection>& detections) const;
    void DuoConfidenceBoost(std::vector<Detection>& detections) const;

    // Mahalanobis distance matrix (Nd x Nt).
    Eigen::MatrixXd GetMhDistMatrix(const std::vector<Detection>& detections) const;

    // OccluTrack abnormal-motion suppression coefficient.
    double ComputeAmsAlpha(KalmanBoxTracker& trk, const Eigen::Vector4d& det_xyxy) const;
    void AmsUpdate(KalmanBoxTracker& trk, const Detection& det);
    void MaybeActivate(KalmanBoxTracker& trk) const;

    // Drop younger of any pair of duplicate emissions sharing IoU >= threshold.
    void SuppressDuplicateEmissions(
        std::vector<std::pair<KalmanBoxTracker::Ptr, Eigen::Vector4d>>& emitted
    );

    // Output filter: aspect/area gating.
    bool PassesFilter(const Eigen::Vector4d& xyxy) const;

    Config config_;
    int frame_count_ = 0;
    std::vector<KalmanBoxTracker::Ptr> trackers_;
    std::unique_ptr<CameraMotionCompensator> cmc_;
    std::optional<OnnxReIdModel> reid_model_;
    double last_reid_time_ms_ = 0.0;
};

}  // namespace occluboost
