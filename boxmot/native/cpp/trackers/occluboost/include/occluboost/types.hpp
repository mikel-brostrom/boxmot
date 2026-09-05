#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <string>

namespace occluboost {

struct Config {
    std::string asso_func = "iou";
    // BoostTrack inherited
    int max_age = 120;
    int min_hits = 1;
    float det_thresh = 0.6F;
    float iou_threshold = 0.2F;
    int min_box_area = 10;
    float aspect_ratio_thresh = 1.6F;
    float lambda_iou = 0.3F;
    float lambda_mhd = 0.3F;
    float lambda_shape = 0.5F;
    bool use_dlo_boost = true;
    bool use_duo_boost = false;
    float dlo_boost_coef = 0.65F;
    bool s_sim_corr = false;
    bool use_rich_s = false;
    bool use_sb = true;
    bool use_vt = false;
    bool use_embeddings = true;
    std::string cmc_method = "ecc";
    int max_obs = 50;

    // OccluBoost specific
    float recovery_appearance_thresh = 0.4F;
    float recovery_iou_thresh = 0.2F;
    int recovery_max_age = 70;
    float feat_alpha = 0.95F;
    float track_low_thresh = 0.04F;
    float second_iou_thresh = 0.5F;
    float second_appearance_thresh = 0.6F;
    int second_pass_max_age = 5;
    int second_pass_min_hits = 3;
    bool use_second_pass = true;
    float new_track_thresh = 0.6F;
    int confirm_hits = 4;
    float instant_confirm_thresh = 0.77F;
    int tentative_max_age = 1;
    float duplicate_iou_thresh = 0.95F;
    bool ams_enabled = true;
    float ams_alpha0 = 0.4F;
    float ams_threshold = 0.5F;
    int ams_buffer_size = 30;
    float ams_shrink_ratio = 0.75F;
    float lambda_emb_multiplier = 1.5F;

    // Multi-class OBB detections need a less restrictive operating point than
    // the MOT-tuned AABB defaults. These mirror the Python tracker settings.
    float obb_det_thresh = 0.2F;
    float obb_iou_threshold = 0.15F;
    float obb_new_track_thresh = 0.3F;
    float obb_instant_confirm_thresh = 0.5F;
    int obb_max_age = 30;
    int obb_recovery_max_age = 15;
    float obb_second_iou_thresh = 0.3F;
};

struct Detection {
    bool is_obb = false;  // True for OBB detections (xywha set); false for AABB (xyxy set).
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

}  // namespace occluboost
