#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <string>

namespace bytetrack {

enum class TrackState {
    kNew = 0,
    kTracked = 1,
    kLost = 2,
    kRemoved = 3,
};

struct Config {
    std::string asso_func = "iou";
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
    std::int64_t cls = 0;
    std::int64_t det_ind = -1;
};

struct TrackOutput {
    bool is_obb = false;
    std::int64_t id = -1;
    Eigen::Vector4d xyxy = Eigen::Vector4d::Zero();
    Eigen::Matrix<double, 5, 1> xywha = Eigen::Matrix<double, 5, 1>::Zero();
    float conf = 0.0F;
    std::int64_t cls = 0;
    std::int64_t det_ind = -1;
};

}  // namespace bytetrack
