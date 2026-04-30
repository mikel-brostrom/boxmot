#pragma once

#include "boxmot/trackers/base/base_tracker.hpp"
#include "bytetrack/track.hpp"
#include "bytetrack/types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace bytetrack {

class ByteTrackTracker final : public boxmot::trackers::base::TrackerBase<Detection, TrackOutput> {
public:
    explicit ByteTrackTracker(Config config);

    std::vector<TrackOutput> Update(const std::vector<Detection>& detections, const cv::Mat& image) override;
    void Reset() override;

    [[nodiscard]] bool SupportsObb() const noexcept override { return true; }
    [[nodiscard]] bool SupportsReId() const noexcept override { return false; }

private:
    std::vector<Track::Ptr> CreateDetectionTracks(const std::vector<Detection>& detections) const;
    std::pair<std::vector<Track::Ptr>, std::vector<Track::Ptr>> SeparateTracks() const;
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
    KalmanFilterXYAH kalman_filter_;
    KalmanFilterXYWH kalman_filter_obb_{5};
    bool detection_mode_ready_ = false;
    bool is_obb_mode_ = false;
    std::vector<Track::Ptr> active_tracks_;
    std::vector<Track::Ptr> lost_tracks_;
    std::vector<Track::Ptr> removed_tracks_;
};

}  // namespace bytetrack
