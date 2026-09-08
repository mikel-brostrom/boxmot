#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <string>

namespace ocsort {

struct Config {
    std::string asso_func = "iou";
    float min_conf = 0.1F;
    float det_thresh = 0.6F;
    float iou_threshold = 0.3F;
    int max_age = 30;
    int min_hits = 3;
    int delta_t = 3;
    bool use_byte = false;
    float inertia = 0.1F;
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

}  // namespace ocsort
