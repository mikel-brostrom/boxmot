#pragma once
// BoxMOT DeepStream adapter types and configuration.

#include "deepstream/reid_tensorrt.hpp"

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace boxmot::deepstream {

// Which BoxMOT tracker algorithm to use
enum class TrackerAlgorithm {
    kBotSort = 0,
    kByteTrack = 1,
    kOcSort = 2,
    kSfSort = 3,
    kOccluBoost = 4,
};

// Configuration parsed from the DeepStream low-level tracker YAML config file
struct BoxMOTTrackerConfig {
    // Tracker selection
    TrackerAlgorithm algorithm = TrackerAlgorithm::kBotSort;

    // Common tracker parameters
    float track_high_thresh = 0.6f;
    float track_low_thresh = 0.1f;
    float new_track_thresh = 0.7f;
    int track_buffer = 30;
    float match_thresh = 0.8f;
    int frame_rate = 30;
    int max_targets_per_stream = 150;

    // BoTSORT-specific
    float proximity_thresh = 0.5f;
    float appearance_thresh = 0.25f;
    std::string cmc_method = "none";
    bool fuse_first_associate = false;
    bool with_reid = true;
    int max_obs = 50;

    // ByteTrack-specific
    // (uses common params)

    // OCSORT-specific
    float delta_t = 3.0f;
    float iou_thresh = 0.3f;
    float vel_dir_weight = 0.2f;
    float Q_xy_scaling = 1.0f / 20.0f;
    float Q_s_scaling = 1.0f / 160.0f;

    // SFSORT-specific
    float scene_point_thresh = 0.5f;
    float high_score_thresh = 0.6f;
    float low_score_thresh = 0.1f;

    // ReID configuration (TensorRT)
    TensorRTReIdConfig reid;
    bool enable_reid = true;

    // Target management (matching DeepStream's conventions)
    int probation_age = 3;
    int max_shadow_tracking_age = 30;
    int early_termination_age = 1;
    bool output_terminated_tracks = false;
    bool output_shadow_tracks = false;
    bool support_past_frame = true;

    // Data association
    int association_matcher_type = 0;  // 0=GREEDY, 1=CASCADED
    bool check_class_match = false;
    float min_matching_score_overall = 0.0f;
    float min_matching_score_iou = 0.0f;
    float matching_score_weight_iou = 1.0f;
    float matching_score_weight_reid = 0.0f;
    float min_iou_diff_new_target = 0.5f;
};

// Tracked object data matching DeepStream's output format
struct TrackedObject {
    int64_t track_id = -1;
    int class_id = 0;
    float bbox_left = 0.0f;
    float bbox_top = 0.0f;
    float bbox_width = 0.0f;
    float bbox_height = 0.0f;
    float confidence = 1.0f;
    float tracker_confidence = 1.0f;
    int age = 0;
    bool is_shadow = false;
    int associated_det_index = -1;  // Index into input detection list, -1 if none
};

// Past-frame object data for miscellaneous data output
struct PastFrameObject {
    int64_t track_id = -1;
    int class_id = 0;
    int frame_num = 0;
    float bbox_left = 0.0f;
    float bbox_top = 0.0f;
    float bbox_width = 0.0f;
    float bbox_height = 0.0f;
    float confidence = 1.0f;
};

// Per-stream tracker context
struct StreamContext {
    int64_t next_id = 0;
    int frame_count = 0;
    std::vector<TrackedObject> last_output;
    std::vector<std::vector<PastFrameObject>> past_frame_buffer;
    std::vector<TrackedObject> terminated_tracks;
    std::vector<TrackedObject> shadow_tracks;
};

// Parse a YAML config file into BoxMOTTrackerConfig
BoxMOTTrackerConfig ParseConfig(const std::string& config_path);

}  // namespace boxmot::deepstream
