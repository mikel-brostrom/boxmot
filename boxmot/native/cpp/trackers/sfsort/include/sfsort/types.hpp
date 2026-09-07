#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <string>

namespace sfsort {

enum class TrackState {
    kActive = 0,
    kLostCentral = 1,
    kLostMarginal = 2,
};

struct Config {
    std::string asso_func = "iou";
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
    std::int64_t cls = 0;
    std::int64_t det_ind = -1;
};

struct TrackOutput {
    bool is_obb = false;
    Eigen::Vector4d xyxy = Eigen::Vector4d::Zero();
    Eigen::Matrix<double, 5, 1> xywha = Eigen::Matrix<double, 5, 1>::Zero();
    std::int64_t id = -1;
    float conf = 0.0F;
    std::int64_t cls = 0;
    std::int64_t det_ind = -1;
};

}  // namespace sfsort
