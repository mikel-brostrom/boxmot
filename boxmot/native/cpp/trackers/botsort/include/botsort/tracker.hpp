#pragma once

#include "boxmot/trackers/base/base_tracker.hpp"
#include "botsort/cmc.hpp"
#include "botsort/reid_onnx.hpp"
#include "botsort/track.hpp"
#include "botsort/types.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace botsort {

class BotSortTracker final : public boxmot::trackers::base::TrackerBase<Detection, TrackOutput> {
public:
    explicit BotSortTracker(Config config);

    std::vector<TrackOutput> Update(const std::vector<Detection>& detections, const cv::Mat& image) override;
    void Reset() override;
    [[nodiscard]] double LastReIdTimeMs() const noexcept { return last_reid_time_ms_; }
    [[nodiscard]] double LastReIdPreprocessTimeMs() const noexcept { return last_reid_preprocess_time_ms_; }
    [[nodiscard]] double LastReIdProcessTimeMs() const noexcept { return last_reid_process_time_ms_; }
    [[nodiscard]] double LastReIdPostprocessTimeMs() const noexcept { return last_reid_postprocess_time_ms_; }

    [[nodiscard]] bool SupportsObb() const noexcept override { return true; }
    [[nodiscard]] bool SupportsReId() const noexcept override { return true; }

private:
    std::vector<Track::Ptr> CreateDetectionTracks(const std::vector<Detection>& detections) const;
    std::pair<std::vector<Track::Ptr>, std::vector<Track::Ptr>> SeparateTracks() const;
    void ApplyCameraMotionCompensation(
        const cv::Mat& image,
        const std::vector<Detection>& detections,
        std::vector<Track::Ptr>& strack_pool,
        std::vector<Track::Ptr>& unconfirmed
    );
    void UpdateTrackStates(std::vector<Track::Ptr>& removed_tracks);
    std::vector<TrackOutput> PrepareOutput(
        const std::vector<Track::Ptr>& activated_tracks,
        const std::vector<Track::Ptr>& refind_tracks,
        const std::vector<Track::Ptr>& lost_tracks,
        const std::vector<Track::Ptr>& removed_tracks
    );

    Config config_;
    int frame_count_ = 0;
    int max_time_lost_ = 30;
    KalmanFilterXYWH kalman_filter_{4};
    std::unique_ptr<CameraMotionCompensator> cmc_;
    std::optional<OnnxReIdModel> reid_model_;
    std::vector<Track::Ptr> active_tracks_;
    std::vector<Track::Ptr> lost_tracks_;
    std::vector<Track::Ptr> removed_tracks_;
    double last_reid_time_ms_ = 0.0;
    double last_reid_preprocess_time_ms_ = 0.0;
    double last_reid_process_time_ms_ = 0.0;
    double last_reid_postprocess_time_ms_ = 0.0;
    bool detection_mode_ready_ = false;
    bool is_obb_mode_ = false;
};

}  // namespace botsort
