#pragma once

#include <Eigen/Dense>

#include <filesystem>
#include <string>
#include <vector>

namespace ocsort {

namespace fs = std::filesystem;

struct Config {
    float min_conf = 0.1F;
    float det_thresh = 0.6F;
    float iou_threshold = 0.3F;
    int max_age = 30;
    int min_hits = 3;
    int delta_t = 3;
    bool use_byte = false;
    float inertia = 0.1F;
    float q_xy_scaling = 0.01F;
    float q_s_scaling = 0.0001F;
    int max_obs = 50;
};

struct Detection {
    bool is_obb = false;
    Eigen::Vector4d xyxy = Eigen::Vector4d::Zero();
    Eigen::Matrix<double, 5, 1> xywha = Eigen::Matrix<double, 5, 1>::Zero();
    float conf = 0.0F;
    int cls = 0;
    int det_ind = -1;
};

struct TrackOutput {
    bool is_obb = false;
    Eigen::Vector4d xyxy = Eigen::Vector4d::Zero();
    Eigen::Matrix<double, 5, 1> xywha = Eigen::Matrix<double, 5, 1>::Zero();
    int id = -1;
    float conf = 0.0F;
    int cls = 0;
    int det_ind = -1;
};

struct ReplayOptions {
    fs::path mot_root;
    fs::path det_emb_root;
    std::string detector_name;
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

}  // namespace ocsort
