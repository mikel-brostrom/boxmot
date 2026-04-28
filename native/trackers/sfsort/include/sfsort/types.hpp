#pragma once

#include <Eigen/Dense>

#include <filesystem>
#include <string>
#include <vector>

namespace sfsort {

namespace fs = std::filesystem;

enum class TrackState {
    kActive = 0,
    kLostCentral = 1,
    kLostMarginal = 2,
};

struct Config {
    float high_th = 0.6F;
    float match_th_first = 0.67F;
    float new_track_th = 0.7F;
    float low_th = 0.1F;
    float match_th_second = 0.3F;
    bool dynamic_tuning = false;
    float cth = 0.5F;
    float high_th_m = 0.0F;
    float new_track_th_m = 0.0F;
    float match_th_first_m = 0.0F;
    float obb_theta_damping = 0.8F;
    int marginal_timeout = 0;
    int central_timeout = 0;
    int frame_width = 0;
    int frame_height = 0;
    int horizontal_margin = 0;
    int vertical_margin = 0;
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

}  // namespace sfsort
