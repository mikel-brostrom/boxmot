#pragma once

#include "boxmot/trackers/base/base_tracker.hpp"
#include "sfsort/types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace sfsort {

class SFSORTTracker final : public boxmot::trackers::base::TrackerBase<Detection, TrackOutput> {
public:
    struct TrackData {
        bool is_obb = false;
        Eigen::Vector4d xyxy = Eigen::Vector4d::Zero();
        Eigen::Matrix<double, 5, 1> xywha = Eigen::Matrix<double, 5, 1>::Zero();
        int last_frame = 0;
        int track_id = -1;
        float conf = 0.0F;
        int cls = 0;
        int det_ind = -1;
        TrackState state = TrackState::kActive;
        int time_since_update = 0;
        double theta_velocity = 0.0;

        void Update(const Detection& detection, int frame_id, float obb_theta_damping);
    };

    explicit SFSORTTracker(Config config);

    std::vector<TrackOutput> Update(const std::vector<Detection>& detections, const cv::Mat& image) override;
    void Reset() override;

    [[nodiscard]] bool SupportsObb() const noexcept override { return true; }
    [[nodiscard]] bool SupportsReId() const noexcept override { return false; }

private:
    [[nodiscard]] std::tuple<float, float, float> DynamicThresholds(const std::vector<Detection>& detections) const;
    void MaybeSetMargins(int frame_width, int frame_height);
    void PurgeStaleLostTracks();
    void UpdateLostTracks(const std::vector<TrackData>& next_lost_tracks);
    [[nodiscard]] TrackData NewTrack(const Detection& detection) const;
    [[nodiscard]] static TrackOutput FormatTrack(const TrackData& track);
    [[nodiscard]] static Eigen::MatrixXd CalculateCost(
        const std::vector<TrackData*>& tracks,
        const std::vector<Detection>& detections,
        bool iou_only
    );

    Config config_;
    int frame_count_ = 0;
    int id_counter_ = 0;
    bool margins_ready_ = false;
    bool is_obb_mode_ = false;
    bool detection_mode_ready_ = false;
    double l_margin_ = 0.0;
    double r_margin_ = 0.0;
    double t_margin_ = 0.0;
    double b_margin_ = 0.0;
    std::vector<TrackData> active_tracks_;
    std::vector<TrackData> lost_tracks_;
};

}  // namespace sfsort
