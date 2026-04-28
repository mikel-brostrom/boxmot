#pragma once

#include <Eigen/Dense>

#include <filesystem>
#include <string>
#include <vector>

namespace bytetrack {

namespace fs = std::filesystem;

enum class TrackState {
    kNew = 0,
    kTracked = 1,
    kLost = 2,
    kRemoved = 3,
};

struct Config {
    float min_conf = 0.1F;
    float track_thresh = 0.6F;
    float match_thresh = 0.9F;
    int track_buffer = 30;
    int frame_rate = 30;
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

}  // namespace bytetrack
