#pragma once

#include "boxmot/trackers/base/association.hpp"

#include <Eigen/Dense>

#include <utility>
#include <vector>

namespace occluboost {

struct AssociationResult {
    std::vector<std::pair<int, int>> matches;  // (det_index, trk_index)
    std::vector<int> unmatched_dets;
    std::vector<int> unmatched_trks;
    Eigen::MatrixXd geometry_matrix;
};

Eigen::MatrixXd SoftBiouBatch(const Eigen::MatrixXd& dets, const Eigen::MatrixXd& trks);
Eigen::MatrixXd ShapeSimilarityV1(const Eigen::MatrixXd& dets, const Eigen::MatrixXd& trks);
Eigen::MatrixXd MhDistSimilarity(const Eigen::MatrixXd& mh_dist);

// BoostTrack-style multi-cue association.
//   detections: (Nd x >=5) with [x1,y1,x2,y2,conf,...]
//   trackers:   (Nt x 5)   with [x1,y1,x2,y2,track_conf]
//   mh_dist:    (Nd x Nt) precomputed Mahalanobis squared distances; can be empty.
//   emb_cost:   (Nd x Nt) precomputed cosine similarity matrix (Nd x Nt); can be empty.
AssociationResult Associate(const Eigen::MatrixXd& detections,
                            const Eigen::MatrixXd& trackers,
                            double similarity_threshold,
                            const Eigen::MatrixXd& mh_dist,
                            const Eigen::VectorXd& detection_confidence,
                            const Eigen::VectorXd& track_confidence,
                            const Eigen::MatrixXd& emb_cost,
                            double lambda_iou,
                            double lambda_mhd,
                            double lambda_shape,
                            double lambda_emb_multiplier,
                            boxmot::trackers::base::AssociationMode association_mode,
                            int frame_width,
                            int frame_height);

}  // namespace occluboost
