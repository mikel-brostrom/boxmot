#pragma once

#include <Eigen/Dense>

#include <filesystem>
#include <string>
#include <vector>

namespace botsort {

namespace fs = std::filesystem;

enum class TrackState {
    kNew = 0,
    kTracked = 1,
    kLost = 2,
    kRemoved = 3,
};

struct Config {
    float track_high_thresh = 0.6F;
    float track_low_thresh = 0.1F;
    float new_track_thresh = 0.7F;
    int track_buffer = 30;
    float match_thresh = 0.8F;
    float proximity_thresh = 0.5F;
    float appearance_thresh = 0.25F;
    std::string cmc_method = "ecc";
    int frame_rate = 30;
    bool fuse_first_associate = false;
    bool with_reid = true;
    int max_obs = 50;
    std::string reid_model_path;
    std::string reid_preprocess = "resize_pad";
};

struct Detection {
    bool is_obb = false;
    Eigen::Vector4d xyxy = Eigen::Vector4d::Zero();
    Eigen::Matrix<double, 5, 1> xywha = Eigen::Matrix<double, 5, 1>::Zero();
    float conf = 0.0F;
    int cls = 0;
    int det_ind = -1;
    Eigen::VectorXf embedding;

    bool has_embedding() const { return embedding.size() > 0; }
};

struct TrackOutput {
    bool is_obb = false;
    int id = -1;
    Eigen::Vector4d xyxy = Eigen::Vector4d::Zero();
    Eigen::Matrix<double, 5, 1> xywha = Eigen::Matrix<double, 5, 1>::Zero();
    float conf = 0.0F;
    int cls = 0;
    int det_ind = -1;
};

struct ReplayOptions {
    fs::path mot_root;
    fs::path det_emb_root;
    std::string detector_name;
    std::string reid_name;
    fs::path reid_model_path;
    std::string reid_preprocess = "resize";
    std::string sequence;
    fs::path output_path;
    float conf_threshold = 0.0F;
    int target_fps = 0;
    Config tracker;
};

struct ReplaySummary {
    std::string sequence;
    int num_frames = 0;
    double track_time_ms = 0.0;
    std::vector<int> kept_frame_ids;
};

}  // namespace botsort
