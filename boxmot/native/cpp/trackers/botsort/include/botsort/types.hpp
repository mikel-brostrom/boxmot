#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <string>

namespace botsort {

enum class TrackState {
    kNew = 0,
    kTracked = 1,
    kLost = 2,
    kRemoved = 3,
};

struct Config {
    std::string asso_func = "iou";
    float track_high_thresh = 0.6F;
    float track_low_thresh = 0.1F;
    float new_track_thresh = 0.7F;
    int track_buffer = 30;
    float match_thresh = 0.8F;
    float proximity_thresh = 0.5F;
    float appearance_thresh = 0.25F;
    float second_match_thresh = 0.5F;
    float unconfirmed_match_thresh = 0.7F;
    float unconfirmed_emb_scale = 2.0F;
    std::string cmc_method = "ecc";
    int frame_rate = 30;
    bool fuse_first_associate = false;
    bool use_embeddings = true;
    int max_obs = 50;
};

struct Detection {
    bool is_obb = false;
    Eigen::Vector4d xyxy = Eigen::Vector4d::Zero();
    Eigen::Matrix<double, 5, 1> xywha = Eigen::Matrix<double, 5, 1>::Zero();
    float conf = 0.0F;
    std::int64_t cls = 0;
    std::int64_t det_ind = -1;
    Eigen::VectorXf embedding;

    bool has_embedding() const { return embedding.size() > 0; }
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

}  // namespace botsort
