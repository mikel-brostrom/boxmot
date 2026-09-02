#include "occluboost/tracker.hpp"

#include "occluboost/association.hpp"
#include "boxmot/trackers/base/assignment.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace occluboost {

using boxmot::trackers::base::AssignmentResult;
using boxmot::trackers::base::LinearAssignment;

namespace {

Eigen::MatrixXd PredictedTracksMatrix(const std::vector<KalmanBoxTracker::Ptr>& trks) {
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(static_cast<int>(trks.size()), 5);
    for (std::size_t i = 0; i < trks.size(); ++i) {
        const Eigen::Vector4d xyxy = trks[i]->xyxy();
        out(static_cast<int>(i), 0) = xyxy[0];
        out(static_cast<int>(i), 1) = xyxy[1];
        out(static_cast<int>(i), 2) = xyxy[2];
        out(static_cast<int>(i), 3) = xyxy[3];
        out(static_cast<int>(i), 4) = trks[i]->GetConfidence();
    }
    return out;
}

Eigen::MatrixXd DetectionsMatrix(const std::vector<Detection>& dets) {
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(static_cast<int>(dets.size()), 5);
    for (std::size_t i = 0; i < dets.size(); ++i) {
        out(static_cast<int>(i), 0) = dets[i].xyxy[0];
        out(static_cast<int>(i), 1) = dets[i].xyxy[1];
        out(static_cast<int>(i), 2) = dets[i].xyxy[2];
        out(static_cast<int>(i), 3) = dets[i].xyxy[3];
        out(static_cast<int>(i), 4) = static_cast<double>(dets[i].conf);
    }
    return out;
}

}  // namespace

namespace {

boxmot::trackers::base::ReIdDevice ParseReIdDevice(const std::string& s) {
    if (s.empty() || s == "auto") return boxmot::trackers::base::ReIdDevice::kAuto;
    if (s == "cpu") return boxmot::trackers::base::ReIdDevice::kCpu;
    if (s == "cuda" || s == "gpu" || s.rfind("cuda:", 0) == 0) return boxmot::trackers::base::ReIdDevice::kCuda;
    if (s == "coreml" || s == "mps" || s == "metal") return boxmot::trackers::base::ReIdDevice::kCoreMl;
    // Bare numeric strings like "0" or "1" refer to CUDA devices.
    if (!s.empty() && std::all_of(s.begin(), s.end(), ::isdigit)) return boxmot::trackers::base::ReIdDevice::kCuda;
    return boxmot::trackers::base::ReIdDevice::kAuto;
}

}  // namespace

OccluBoostTracker::OccluBoostTracker(Config config)
    : config_(std::move(config)),
      association_mode_(boxmot::trackers::base::ParseAssociationMode(config_.asso_func)),
      cmc_(CreateCameraMotionCompensator(config_.cmc_method)) {
    config_.obb_det_thresh = std::max(config_.obb_det_thresh, 0.0F);
    config_.obb_iou_threshold = std::clamp(config_.obb_iou_threshold, 0.0F, 1.0F);
    config_.obb_new_track_thresh = std::max(config_.obb_new_track_thresh, config_.obb_det_thresh);
    config_.obb_instant_confirm_thresh =
        std::max(config_.obb_instant_confirm_thresh, config_.obb_new_track_thresh);
    config_.obb_max_age = std::max(config_.obb_max_age, 0);
    config_.obb_recovery_max_age = std::max(config_.obb_recovery_max_age, 0);
    config_.obb_second_iou_thresh = std::clamp(config_.obb_second_iou_thresh, 0.0F, 1.0F);
    KalmanBoxTracker::ResetCount();
    if (config_.with_reid && !config_.reid_model_path.empty()) {
        reid_model_ = MaybeCreateOnnxReIdModel(
            config_.reid_model_path, config_.reid_preprocess,
            boxmot::trackers::base::ReIdBackend::kAuto,
            ParseReIdDevice(config_.reid_device));
    }
}

void OccluBoostTracker::Reset() {
    frame_count_ = 0;
    trackers_.clear();
    cmc_ = CreateCameraMotionCompensator(config_.cmc_method);
    KalmanBoxTracker::ResetCount();
    reid_model_.reset();
    if (config_.with_reid && !config_.reid_model_path.empty()) {
        reid_model_ = MaybeCreateOnnxReIdModel(
            config_.reid_model_path, config_.reid_preprocess,
            boxmot::trackers::base::ReIdBackend::kAuto,
            ParseReIdDevice(config_.reid_device));
    }
    last_reid_time_ms_ = 0.0;
    last_reid_preprocess_time_ms_ = 0.0;
    last_reid_process_time_ms_ = 0.0;
    last_reid_postprocess_time_ms_ = 0.0;
    detection_mode_ready_ = false;
    is_obb_mode_ = false;
    association_frame_width_ = 0;
    association_frame_height_ = 0;
}

std::vector<Detection> OccluBoostTracker::EnsureEmbeddings(
    std::vector<Detection> detections,
    const cv::Mat& image
) {
    last_reid_time_ms_ = 0.0;
    last_reid_preprocess_time_ms_ = 0.0;
    last_reid_process_time_ms_ = 0.0;
    last_reid_postprocess_time_ms_ = 0.0;
    if (!config_.with_reid || !reid_model_.has_value()) {
        return detections;
    }
    bool needs_embeddings = false;
    for (const auto& det : detections) {
        if (!det.has_embedding()) {
            needs_embeddings = true;
            break;
        }
    }
    if (!needs_embeddings) {
        return detections;
    }
    const TimedReIdFeatures timed = GetReIdFeaturesTimed(*reid_model_, detections, image);
    last_reid_preprocess_time_ms_ = timed.preprocess_ms;
    last_reid_process_time_ms_ = timed.process_ms;
    last_reid_postprocess_time_ms_ = timed.postprocess_ms;
    last_reid_time_ms_ = timed.preprocess_ms + timed.process_ms + timed.postprocess_ms;
    if (timed.features.size() != detections.size()) {
        throw std::runtime_error("Native OccluBoost ReID returned a different number of embeddings than detections.");
    }
    for (std::size_t i = 0; i < detections.size(); ++i) {
        detections[i].embedding = timed.features[i];
    }
    return detections;
}

void OccluBoostTracker::DloConfidenceBoost(std::vector<Detection>& detections) const {
    if (detections.empty() || trackers_.empty()) {
        return;
    }
    Eigen::MatrixXd dets_mat = DetectionsMatrix(detections);
    Eigen::MatrixXd trks_mat = PredictedTracksMatrix(trackers_);

    // Python computes sbiou = soft_biou_batch(detections, trackers) but only
    // uses it when (use_sb || use_vt) AND use_rich_s — locked off for OccluBoost.
    Eigen::MatrixXd S = IouBatch(dets_mat, trks_mat);
    if (S.rows() == 0 || S.cols() == 0) {
        return;
    }

    Eigen::VectorXd max_s = S.rowwise().maxCoeff();

    if (!config_.use_sb && !config_.use_vt) {
        for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
            const double boosted = max_s(i) * config_.dlo_boost_coef;
            if (boosted > detections[i].conf) {
                detections[i].conf = static_cast<float>(boosted);
            }
        }
        return;
    }

    if (config_.use_sb) {
        constexpr double alpha = 0.65;
        for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
            const double boosted = alpha * detections[i].conf + (1.0 - alpha) * std::pow(max_s(i), 1.5);
            if (boosted > detections[i].conf) {
                detections[i].conf = static_cast<float>(boosted);
            }
        }
    }
    if (config_.use_vt) {
        constexpr double threshold_s = 0.95;
        constexpr double threshold_e = 0.8;
        for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
            bool any = false;
            for (std::size_t j = 0; j < trackers_.size(); ++j) {
                const double thr = std::max(threshold_s - (trackers_[j]->time_since_update - 1), threshold_e);
                if (S(i, static_cast<int>(j)) > thr) {
                    any = true;
                    break;
                }
            }
            if (any && detections[i].conf < config_.det_thresh + 1.0e-5F) {
                detections[i].conf = config_.det_thresh + 1.0e-5F;
            }
        }
    }
}

void OccluBoostTracker::DuoConfidenceBoost(std::vector<Detection>& detections) const {
    // OccluBoost defaults disable DUO; provided for completeness.
    if (!config_.use_duo_boost || detections.empty() || trackers_.empty()) {
        return;
    }
    constexpr double limit = 13.2767;
    Eigen::MatrixXd mh = GetMhDistMatrix(detections);
    if (mh.size() == 0) {
        return;
    }
    Eigen::VectorXd min_dists = mh.rowwise().minCoeff();
    std::vector<int> boost_inds;
    for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
        if (min_dists(i) > limit && detections[i].conf < config_.det_thresh) {
            boost_inds.push_back(i);
        }
    }
    if (boost_inds.empty()) {
        return;
    }
    constexpr double iou_limit = 0.3;

    Eigen::MatrixXd boost_dets(static_cast<int>(boost_inds.size()), 5);
    for (std::size_t k = 0; k < boost_inds.size(); ++k) {
        const auto& d = detections[boost_inds[k]];
        boost_dets(static_cast<int>(k), 0) = d.xyxy[0];
        boost_dets(static_cast<int>(k), 1) = d.xyxy[1];
        boost_dets(static_cast<int>(k), 2) = d.xyxy[2];
        boost_dets(static_cast<int>(k), 3) = d.xyxy[3];
        boost_dets(static_cast<int>(k), 4) = d.conf;
    }
    Eigen::MatrixXd bdiou = IouBatch(boost_dets, boost_dets);
    for (int i = 0; i < bdiou.rows(); ++i) {
        bdiou(i, i) = 0.0;
    }

    std::unordered_set<int> remaining;
    for (int k = 0; k < static_cast<int>(boost_inds.size()); ++k) {
        const double row_max = bdiou.row(k).maxCoeff();
        if (row_max <= iou_limit) {
            remaining.insert(boost_inds[k]);
        } else {
            // Same conf-tiebreaker as Python: keep the highest-confidence detection in the cluster.
            float best_conf = detections[boost_inds[k]].conf;
            for (int j = 0; j < bdiou.cols(); ++j) {
                if (bdiou(k, j) > iou_limit) {
                    best_conf = std::max(best_conf, detections[boost_inds[j]].conf);
                }
            }
            if (detections[boost_inds[k]].conf == best_conf) {
                remaining.insert(boost_inds[k]);
            }
        }
    }
    for (int idx : remaining) {
        detections[idx].conf = std::max(detections[idx].conf, config_.det_thresh + 1.0e-4F);
    }
}

void OccluBoostTracker::DloConfidenceBoostObb(std::vector<Detection>& detections) const {
    if (detections.empty() || trackers_.empty()) {
        return;
    }

    Eigen::MatrixXd det_boxes(static_cast<int>(detections.size()), 5);
    for (std::size_t i = 0; i < detections.size(); ++i) {
        det_boxes.row(static_cast<int>(i)) = detections[i].xywha.transpose();
    }
    Eigen::MatrixXd trk_boxes(static_cast<int>(trackers_.size()), 5);
    Eigen::VectorXd trk_conf(static_cast<int>(trackers_.size()));
    for (std::size_t j = 0; j < trackers_.size(); ++j) {
        trk_boxes.row(static_cast<int>(j)) = trackers_[j]->xywha().transpose();
        trk_conf(static_cast<int>(j)) = std::clamp(
            static_cast<double>(trackers_[j]->GetConfidence()), 0.0, 1.0);
    }

    Eigen::MatrixXd similarity = IouBatchObb(det_boxes, trk_boxes);
    if (config_.use_rich_s) {
        const Eigen::MatrixXd mhd_similarity = MhDistSimilarity(GetMhDistMatrixObb(detections));

        Eigen::MatrixXd shape_similarity(det_boxes.rows(), trk_boxes.rows());
        for (int i = 0; i < det_boxes.rows(); ++i) {
            for (int j = 0; j < trk_boxes.rows(); ++j) {
                const auto relative_delta = [](const double lhs, const double rhs) {
                    return std::abs(lhs - rhs) / std::max({lhs, rhs, 1.0e-6});
                };
                const double direct = relative_delta(det_boxes(i, 2), trk_boxes(j, 2))
                    + relative_delta(det_boxes(i, 3), trk_boxes(j, 3));
                const double swapped = relative_delta(det_boxes(i, 2), trk_boxes(j, 3))
                    + relative_delta(det_boxes(i, 3), trk_boxes(j, 2));
                shape_similarity(i, j) = std::exp(-std::min(direct, swapped));
            }
        }

        Eigen::MatrixXd buffered_dets = det_boxes;
        Eigen::MatrixXd buffered_trks = trk_boxes;
        const double det_scale = 1.0 + (1.0 - trk_conf.maxCoeff()) * 0.5;
        buffered_dets.col(2) *= det_scale;
        buffered_dets.col(3) *= det_scale;
        for (int j = 0; j < buffered_trks.rows(); ++j) {
            const double scale = 1.0 + (1.0 - trk_conf(j));
            buffered_trks(j, 2) *= scale;
            buffered_trks(j, 3) *= scale;
        }
        const Eigen::MatrixXd soft_iou = IouBatchObb(buffered_dets, buffered_trks);
        similarity = (mhd_similarity + shape_similarity + soft_iou) / 3.0;
    }

    const Eigen::VectorXd max_similarity = similarity.rowwise().maxCoeff();
    if (!config_.use_sb && !config_.use_vt) {
        for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
            detections[static_cast<std::size_t>(i)].conf = std::max(
                detections[static_cast<std::size_t>(i)].conf,
                static_cast<float>(max_similarity(i) * config_.dlo_boost_coef));
        }
        return;
    }

    if (config_.use_sb) {
        constexpr double alpha = 0.65;
        for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
            const float original = detections[static_cast<std::size_t>(i)].conf;
            const double boosted = alpha * original + (1.0 - alpha) * std::pow(max_similarity(i), 1.5);
            detections[static_cast<std::size_t>(i)].conf = std::max(original, static_cast<float>(boosted));
        }
    }
    if (config_.use_vt) {
        for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
            bool visible = false;
            for (int j = 0; j < static_cast<int>(trackers_.size()); ++j) {
                const double threshold = std::max(
                    0.95 - static_cast<double>(trackers_[static_cast<std::size_t>(j)]->time_since_update - 1),
                    0.8);
                if (similarity(i, j) > threshold) {
                    visible = true;
                    break;
                }
            }
            if (visible) {
                detections[static_cast<std::size_t>(i)].conf = std::max(
                    detections[static_cast<std::size_t>(i)].conf,
                    config_.obb_det_thresh + 1.0e-5F);
            }
        }
    }
}

void OccluBoostTracker::DuoConfidenceBoostObb(std::vector<Detection>& detections) const {
    if (!config_.use_duo_boost || detections.empty() || trackers_.empty()) {
        return;
    }
    constexpr double mhd_limit = 13.2767;
    const Eigen::MatrixXd mh = GetMhDistMatrixObb(detections);
    if (mh.size() == 0) {
        return;
    }

    const Eigen::VectorXd min_dists = mh.rowwise().minCoeff();
    std::vector<int> boost_indices;
    for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
        if (min_dists(i) > mhd_limit && detections[static_cast<std::size_t>(i)].conf < config_.obb_det_thresh) {
            boost_indices.push_back(i);
        }
    }
    if (boost_indices.empty()) {
        return;
    }

    Eigen::MatrixXd candidate_boxes(static_cast<int>(boost_indices.size()), 5);
    for (std::size_t i = 0; i < boost_indices.size(); ++i) {
        candidate_boxes.row(static_cast<int>(i)) = detections[boost_indices[i]].xywha.transpose();
    }
    Eigen::MatrixXd pairwise_iou = IouBatchObb(candidate_boxes, candidate_boxes);
    for (int i = 0; i < pairwise_iou.rows(); ++i) {
        pairwise_iou(i, i) = 0.0;
    }

    constexpr double iou_limit = 0.3;
    std::unordered_set<int> remaining;
    for (int i = 0; i < static_cast<int>(boost_indices.size()); ++i) {
        if (pairwise_iou.row(i).maxCoeff() <= iou_limit) {
            remaining.insert(boost_indices[static_cast<std::size_t>(i)]);
            continue;
        }
        int best = boost_indices[static_cast<std::size_t>(i)];
        float best_conf = detections[static_cast<std::size_t>(best)].conf;
        for (int j = 0; j < pairwise_iou.cols(); ++j) {
            if (pairwise_iou(i, j) > iou_limit) {
                const int candidate = boost_indices[static_cast<std::size_t>(j)];
                if (detections[static_cast<std::size_t>(candidate)].conf > best_conf) {
                    best = candidate;
                    best_conf = detections[static_cast<std::size_t>(candidate)].conf;
                }
            }
        }
        remaining.insert(best);
    }
    for (const int index : remaining) {
        detections[static_cast<std::size_t>(index)].conf = config_.obb_det_thresh + 1.0e-4F;
    }
}

Eigen::MatrixXd OccluBoostTracker::GetMhDistMatrix(const std::vector<Detection>& detections) const {
    const int n = static_cast<int>(detections.size());
    const int m = static_cast<int>(trackers_.size());
    if (n == 0 || m == 0) {
        return Eigen::MatrixXd::Zero(n, m);
    }
    constexpr int dim = 4;
    Eigen::MatrixXd z(n, dim);
    Eigen::MatrixXd x(m, dim);
    Eigen::MatrixXd sigma_inv(m, dim);
    for (int i = 0; i < n; ++i) {
        z.row(i) = XyxyToZ(detections[i].xyxy).transpose();
    }
    for (int j = 0; j < m; ++j) {
        const auto& tr = trackers_[j];
        for (int d = 0; d < dim; ++d) {
            x(j, d) = tr->kf.mean()(d);
            const double diag = tr->kf.covariance()(d, d);
            sigma_inv(j, d) = diag != 0.0 ? 1.0 / diag : 0.0;
        }
    }
    Eigen::MatrixXd out(n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            double acc = 0.0;
            for (int d = 0; d < dim; ++d) {
                const double diff = z(i, d) - x(j, d);
                acc += diff * diff * sigma_inv(j, d);
            }
            out(i, j) = acc;
        }
    }
    return out;
}

Eigen::MatrixXd OccluBoostTracker::GetMhDistMatrixObb(const std::vector<Detection>& detections) const {
    const int n = static_cast<int>(detections.size());
    const int m = static_cast<int>(trackers_.size());
    if (n == 0 || m == 0) {
        return Eigen::MatrixXd::Zero(n, m);
    }
    constexpr int dim = 5;
    Eigen::MatrixXd z(n, dim);
    Eigen::MatrixXd x(m, dim);
    Eigen::MatrixXd sigma_inv(m, dim);
    for (int i = 0; i < n; ++i) {
        z.row(i) = XywhaToZObb(detections[static_cast<std::size_t>(i)].xywha).transpose();
    }
    for (int j = 0; j < m; ++j) {
        const auto& tracker = trackers_[static_cast<std::size_t>(j)];
        if (!tracker->kf.is_obb() || tracker->kf.mean().size() < dim) {
            return Eigen::MatrixXd::Constant(n, m, std::numeric_limits<double>::infinity());
        }
        for (int d = 0; d < dim; ++d) {
            x(j, d) = tracker->kf.mean()(d);
            const double variance = tracker->kf.covariance()(d, d);
            sigma_inv(j, d) = variance > 0.0 ? 1.0 / variance : 0.0;
        }
    }
    Eigen::MatrixXd out(n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            out(i, j) = ((z.row(i) - x.row(j)).array().square() * sigma_inv.row(j).array()).sum();
        }
    }
    return out;
}

double OccluBoostTracker::ComputeAmsAlpha(KalmanBoxTracker& trk, const Eigen::Vector4d& det_xyxy) const {
    if (!config_.ams_enabled || config_.ams_alpha0 >= 1.0F) {
        return 1.0;
    }
    const Eigen::Vector4d cur = XyxyToCxcywh(det_xyxy);
    auto& buf = trk.ams_buffer();
    if (buf.size() < 2) {
        if (static_cast<int>(buf.size()) >= config_.ams_buffer_size) {
            buf.pop_front();
        }
        buf.push_back(cur);
        return 1.0;
    }
    const Eigen::Vector4d prev = buf.back();
    const Eigen::Vector4d cur_v = cur - prev;

    Eigen::Vector4d sum_diff = Eigen::Vector4d::Zero();
    int n_diffs = 0;
    auto it = buf.begin();
    auto next = std::next(it);
    while (next != buf.end()) {
        sum_diff += (*next - *it);
        ++it;
        ++next;
        ++n_diffs;
    }
    Eigen::Vector4d mean_v = n_diffs > 0 ? (sum_diff / static_cast<double>(n_diffs)).eval() : Eigen::Vector4d::Zero();

    constexpr double eps = 1.0e-6;
    const double cur_c = std::hypot(cur_v[0], cur_v[1]);
    const double mean_c = std::hypot(mean_v[0], mean_v[1]);
    const double cur_a = std::hypot(cur_v[2], cur_v[3]);
    const double mean_a = std::hypot(mean_v[2], mean_v[3]);

    const double d_c = std::max(0.0, cur_c - mean_c) / std::max(mean_c, eps);
    const double d_a = std::max(0.0, cur_a - mean_a) / std::max(mean_a, eps);

    const double alpha_c = d_c <= config_.ams_threshold ? 1.0 : config_.ams_alpha0;
    const double alpha_a = d_a <= config_.ams_threshold ? 1.0 : config_.ams_alpha0;
    double alpha = 0.5 * (alpha_c + alpha_a);

    const double cur_area = cur[2] * cur[3];
    double mean_area = 0.0;
    for (const auto& v : buf) {
        mean_area += v[2] * v[3];
    }
    mean_area /= static_cast<double>(buf.size());
    if (cur_area >= mean_area * config_.ams_shrink_ratio) {
        alpha = 1.0;
    }

    if (static_cast<int>(buf.size()) >= config_.ams_buffer_size) {
        buf.pop_front();
    }
    buf.push_back(cur);
    return alpha;
}

void OccluBoostTracker::AmsUpdate(KalmanBoxTracker& trk, const Detection& det) {
    if (trk.is_obb()) {
        // OBB tracks: AMS speed-spike heuristic assumes a rectangular box; skip.
        trk.UpdateWithAlpha(det, 1.0);
        return;
    }
    const double alpha = ComputeAmsAlpha(trk, det.xyxy);
    trk.UpdateWithAlpha(det, alpha);
}

void OccluBoostTracker::MaybeActivate(KalmanBoxTracker& trk) const {
    if (!trk.is_activated && trk.hit_streak >= config_.confirm_hits) {
        trk.is_activated = true;
    }
}

bool OccluBoostTracker::PassesFilter(const Eigen::Vector4d& xyxy) const {
    const double w = xyxy[2] - xyxy[0];
    const double h = xyxy[3] - xyxy[1];
    if (h <= 0.0) {
        return false;
    }
    if (w / h > config_.aspect_ratio_thresh) {
        return false;
    }
    if (w * h <= config_.min_box_area) {
        return false;
    }
    return true;
}

bool OccluBoostTracker::PassesObbFilter(const Eigen::Matrix<double, 5, 1>& xywha) const {
    // Mirror Python BoostTrack.filter_outputs_by_geometry in OBB mode:
    // use the enclosing image-axis box for the aspect-ratio gate, while the
    // area gate uses the oriented rectangle's native width and height.
    const Eigen::Vector4d xyxy = XywhaToEnclosingXyxy(xywha);
    const double width = xyxy[2] - xyxy[0];
    const double height = xyxy[3] - xyxy[1];
    if (height <= 0.0) {
        return false;
    }
    if (width / height > config_.aspect_ratio_thresh) {
        return false;
    }
    if (xywha[2] * xywha[3] <= config_.min_box_area) {
        return false;
    }
    return true;
}

void OccluBoostTracker::SuppressDuplicateEmissions(
    std::vector<std::pair<KalmanBoxTracker::Ptr, Eigen::Vector4d>>& emitted
) {
    const int n = static_cast<int>(emitted.size());
    if (n <= 1 || config_.duplicate_iou_thresh <= 0.0F || config_.duplicate_iou_thresh >= 1.0F) {
        return;
    }
    Eigen::MatrixXd ious;
    if (is_obb_mode_) {
        // ``emitted[i].second`` holds enclosing AABB; pull oriented box from the
        // tracker for true OBB IoU (mirrors Python OBB suppression branch).
        Eigen::MatrixXd boxes(n, 5);
        for (int i = 0; i < n; ++i) {
            boxes.row(i) = emitted[i].first->xywha().transpose();
        }
        ious = IouBatchObb(boxes, boxes);
    } else {
        Eigen::MatrixXd boxes(n, 4);
        for (int i = 0; i < n; ++i) {
            boxes.row(i) = emitted[i].second.transpose();
        }
        ious = IouBatch(boxes, boxes);
    }
    for (int i = 0; i < n; ++i) {
        ious(i, i) = 0.0;
    }
    std::unordered_set<int> drop;
    for (int i = 0; i < n; ++i) {
        if (drop.count(i)) {
            continue;
        }
        for (int j = i + 1; j < n; ++j) {
            if (drop.count(j)) {
                continue;
            }
            if (ious(i, j) >= config_.duplicate_iou_thresh) {
                const int age_i = emitted[i].first->age;
                const int age_j = emitted[j].first->age;
                drop.insert(age_i >= age_j ? j : i);
            }
        }
    }
    if (drop.empty()) {
        return;
    }
    std::unordered_set<int> drop_ids;
    for (const int k : drop) {
        drop_ids.insert(emitted[k].first->id);
    }
    trackers_.erase(
        std::remove_if(trackers_.begin(), trackers_.end(),
            [&](const KalmanBoxTracker::Ptr& trk) { return drop_ids.count(trk->id) > 0; }),
        trackers_.end()
    );
    std::vector<std::pair<KalmanBoxTracker::Ptr, Eigen::Vector4d>> kept;
    kept.reserve(emitted.size() - drop.size());
    for (int k = 0; k < n; ++k) {
        if (!drop.count(k)) {
            kept.push_back(emitted[k]);
        }
    }
    emitted = std::move(kept);
}

std::vector<TrackOutput> OccluBoostTracker::Update(
    const std::vector<Detection>& detections,
    const cv::Mat& image
) {
    if (boxmot::trackers::base::AssociationModeRequiresFrameDimensions(association_mode_)
        && (association_frame_width_ <= 0 || association_frame_height_ <= 0)) {
        if (image.empty()) {
            throw std::runtime_error("Native OccluBoost requires an image to initialize centroid association.");
        }
        association_frame_width_ = image.cols;
        association_frame_height_ = image.rows;
    }
    // Latch detection mode (AABB vs OBB) from first non-empty frame.
    if (!detections.empty()) {
        const bool det_is_obb = detections.front().is_obb;
        if (!detection_mode_ready_) {
            detection_mode_ready_ = true;
            is_obb_mode_ = det_is_obb;
            boxmot::trackers::base::ValidateAssociationModeForDetections(association_mode_, det_is_obb);
        } else if (det_is_obb != is_obb_mode_) {
            throw std::runtime_error(
                "Native OccluBoost cannot switch between AABB and OBB detections after initialization."
            );
        }
        for (const auto& d : detections) {
            if (d.is_obb != is_obb_mode_) {
                throw std::runtime_error(
                    "Native OccluBoost cannot mix AABB and OBB detections in the same frame."
                );
            }
        }
    }

    if (is_obb_mode_) {
        return UpdateObb(detections, image);
    }

    ++frame_count_;
    last_reid_time_ms_ = 0.0;
    last_reid_preprocess_time_ms_ = 0.0;
    last_reid_process_time_ms_ = 0.0;
    last_reid_postprocess_time_ms_ = 0.0;

    // Camera-motion compensation applied before predict (Python: cmc.apply→camera_update→predict).
    if (cmc_) {
        const cv::Mat warp = cmc_->Apply(image, detections);
        if (!warp.empty() && warp.rows == 2 && warp.cols == 3) {
            Eigen::Matrix2d linear;
            linear << warp.at<float>(0, 0), warp.at<float>(0, 1),
                warp.at<float>(1, 0), warp.at<float>(1, 1);
            const Eigen::Vector2d translation(warp.at<float>(0, 2), warp.at<float>(1, 2));
            for (auto& trk : trackers_) {
                trk->CameraUpdate(linear, translation);
            }
        }
    }

    // Predict + capture predicted boxes / track confidences.
    Eigen::MatrixXd trks_np = Eigen::MatrixXd::Zero(static_cast<int>(trackers_.size()), 5);
    Eigen::VectorXd track_conf = Eigen::VectorXd::Zero(static_cast<int>(trackers_.size()));
    for (std::size_t i = 0; i < trackers_.size(); ++i) {
        const Eigen::Vector4d xyxy = trackers_[i]->Predict();
        const double tc = trackers_[i]->GetConfidence();
        trks_np(static_cast<int>(i), 0) = xyxy[0];
        trks_np(static_cast<int>(i), 1) = xyxy[1];
        trks_np(static_cast<int>(i), 2) = xyxy[2];
        trks_np(static_cast<int>(i), 3) = xyxy[3];
        trks_np(static_cast<int>(i), 4) = tc;
        track_conf(static_cast<int>(i)) = tc;
    }

    // Capture original confidences before any boosting (used for second-pass split).
    std::vector<float> orig_confs;
    orig_confs.reserve(detections.size());
    for (const auto& d : detections) {
        orig_confs.push_back(d.conf);
    }

    // ReID embeddings (or noop if cached/disabled).
    std::vector<Detection> working = EnsureEmbeddings(detections, image);

    // DLO + DUO boosting (DUO is off by default).
    if (config_.use_dlo_boost) {
        DloConfidenceBoost(working);
    }
    if (config_.use_duo_boost) {
        DuoConfidenceBoost(working);
    }

    // Split into first-pass (>= det_thresh) and second-pass low-conf (>= track_low_thresh, < det_thresh).
    std::vector<Detection> dets_first;
    std::vector<Detection> dets_second;
    dets_first.reserve(working.size());
    dets_second.reserve(working.size());
    for (std::size_t i = 0; i < working.size(); ++i) {
        const bool keep = working[i].conf >= config_.det_thresh;
        if (keep) {
            dets_first.push_back(working[i]);
        } else if (config_.use_second_pass
                   && orig_confs[i] >= config_.track_low_thresh
                   && orig_confs[i] < config_.det_thresh) {
            dets_second.push_back(working[i]);
        }
    }

    // Assemble matrices for the first-pass association.
    Eigen::MatrixXd dets_mat = DetectionsMatrix(dets_first);
    Eigen::VectorXd det_conf = Eigen::VectorXd::Zero(static_cast<int>(dets_first.size()));
    for (std::size_t i = 0; i < dets_first.size(); ++i) {
        det_conf(static_cast<int>(i)) = dets_first[i].conf;
    }

    // emb_cost = dets_embs @ tracker_embs.T (cosine similarity for normalised embs).
    Eigen::MatrixXd emb_cost(0, 0);
    if (config_.with_reid && !trackers_.empty() && !dets_first.empty()) {
        const int feat_dim = static_cast<int>(dets_first.front().embedding.size());
        bool ok = feat_dim > 0;
        for (const auto& d : dets_first) {
            if (d.embedding.size() != feat_dim) {
                ok = false;
                break;
            }
        }
        for (const auto& trk : trackers_) {
            if (!trk->HasEmbedding() || trk->embedding().size() != feat_dim) {
                ok = false;
                break;
            }
        }
        if (ok) {
            Eigen::MatrixXf det_emb_mat(static_cast<int>(dets_first.size()), feat_dim);
            for (std::size_t i = 0; i < dets_first.size(); ++i) {
                det_emb_mat.row(static_cast<int>(i)) = dets_first[i].embedding.transpose();
            }
            Eigen::MatrixXf trk_emb_mat(static_cast<int>(trackers_.size()), feat_dim);
            for (std::size_t j = 0; j < trackers_.size(); ++j) {
                trk_emb_mat.row(static_cast<int>(j)) = trackers_[j]->embedding().transpose();
            }
            emb_cost = (det_emb_mat * trk_emb_mat.transpose()).cast<double>();
        }
    }

    Eigen::MatrixXd mh_dist = GetMhDistMatrix(dets_first);

    AssociationResult assoc = Associate(
        dets_mat,
        trks_np,
        config_.iou_threshold,
        mh_dist,
        det_conf,
        track_conf,
        emb_cost,
        config_.lambda_iou,
        config_.lambda_mhd,
        config_.lambda_shape,
        config_.lambda_emb_multiplier,
        association_mode_,
        association_frame_width_,
        association_frame_height_
    );

    // dets_alpha for ReID EMA on matched pairs.
    Eigen::VectorXd dets_alpha = Eigen::VectorXd::Zero(static_cast<int>(dets_first.size()));
    constexpr double af = 0.95;
    for (std::size_t i = 0; i < dets_first.size(); ++i) {
        const double trust = (dets_first[i].conf - config_.det_thresh) / std::max(1.0 - config_.det_thresh, 1.0e-6);
        dets_alpha(static_cast<int>(i)) = af + (1.0 - af) * (1.0 - trust);
    }

    // Apply matches.
    for (const auto& [d, t] : assoc.matches) {
        AmsUpdate(*trackers_[t], dets_first[d]);
        if (config_.with_reid && dets_first[d].has_embedding()) {
            trackers_[t]->UpdateEmbedding(dets_first[d].embedding, dets_alpha(d));
        }
        MaybeActivate(*trackers_[t]);
    }

    std::vector<int> unmatched_dets = assoc.unmatched_dets;
    std::vector<int> unmatched_trks = assoc.unmatched_trks;

    // ----- ReID-only recovery pass -----
    if (config_.with_reid && !unmatched_trks.empty() && !unmatched_dets.empty()) {
        std::vector<int> elig;
        for (int t : unmatched_trks) {
            if (trackers_[t]->time_since_update <= config_.recovery_max_age && trackers_[t]->HasEmbedding()) {
                elig.push_back(t);
            }
        }
        if (!elig.empty()) {
            const int feat_dim = static_cast<int>(trackers_[elig.front()]->embedding().size());
            Eigen::MatrixXf trk_emb_mat(static_cast<int>(elig.size()), feat_dim);
            for (std::size_t j = 0; j < elig.size(); ++j) {
                trk_emb_mat.row(static_cast<int>(j)) = trackers_[elig[j]]->embedding().transpose();
            }
            Eigen::MatrixXf det_emb_mat(static_cast<int>(unmatched_dets.size()), feat_dim);
            std::vector<int> u_det_idx;
            u_det_idx.reserve(unmatched_dets.size());
            int row_count = 0;
            for (int d : unmatched_dets) {
                if (dets_first[d].embedding.size() != feat_dim) {
                    continue;
                }
                det_emb_mat.row(row_count) = dets_first[d].embedding.transpose();
                u_det_idx.push_back(d);
                ++row_count;
            }
            if (row_count > 0) {
                det_emb_mat.conservativeResize(row_count, feat_dim);
                Eigen::MatrixXd sim = (det_emb_mat * trk_emb_mat.transpose()).cast<double>();

                Eigen::MatrixXd det_box_mat(row_count, 5);
                for (int i = 0; i < row_count; ++i) {
                    const auto& d = dets_first[u_det_idx[i]];
                    det_box_mat(i, 0) = d.xyxy[0];
                    det_box_mat(i, 1) = d.xyxy[1];
                    det_box_mat(i, 2) = d.xyxy[2];
                    det_box_mat(i, 3) = d.xyxy[3];
                    det_box_mat(i, 4) = d.conf;
                }
                Eigen::MatrixXd trk_box_mat(static_cast<int>(elig.size()), 5);
                for (std::size_t j = 0; j < elig.size(); ++j) {
                    const Eigen::Vector4d xyxy = trackers_[elig[j]]->xyxy();
                    trk_box_mat(static_cast<int>(j), 0) = xyxy[0];
                    trk_box_mat(static_cast<int>(j), 1) = xyxy[1];
                    trk_box_mat(static_cast<int>(j), 2) = xyxy[2];
                    trk_box_mat(static_cast<int>(j), 3) = xyxy[3];
                    trk_box_mat(static_cast<int>(j), 4) = trackers_[elig[j]]->GetConfidence();
                }
                Eigen::MatrixXd ious = boxmot::trackers::base::AabbAssociationMatrix(
                    det_box_mat,
                    trk_box_mat,
                    association_mode_,
                    association_frame_width_,
                    association_frame_height_
                );

                Eigen::MatrixXd gated = sim;
                bool any_pos = false;
                for (int i = 0; i < gated.rows(); ++i) {
                    for (int j = 0; j < gated.cols(); ++j) {
                        if (ious(i, j) < config_.recovery_iou_thresh
                            || sim(i, j) < config_.recovery_appearance_thresh) {
                            gated(i, j) = -1.0;
                        } else if (gated(i, j) > 0.0) {
                            any_pos = true;
                        }
                    }
                }
                if (any_pos) {
                    Eigen::MatrixXd cost = -gated;  // maximise gated similarity.
                    AssignmentResult hung = LinearAssignment(cost, std::numeric_limits<double>::infinity());
                    std::unordered_set<int> matched_dets_set;
                    std::unordered_set<int> matched_tracks_set;
                    for (const auto& [r, c] : hung.matches) {
                        if (gated(r, c) <= 0.0) {
                            continue;
                        }
                        const int det_global = u_det_idx[r];
                        const int trk_global = elig[c];
                        matched_dets_set.insert(det_global);
                        matched_tracks_set.insert(trk_global);
                        AmsUpdate(*trackers_[trk_global], dets_first[det_global]);
                        if (dets_first[det_global].has_embedding()) {
                            trackers_[trk_global]->UpdateEmbedding(dets_first[det_global].embedding, config_.feat_alpha);
                        }
                        MaybeActivate(*trackers_[trk_global]);
                    }
                    if (!matched_dets_set.empty()) {
                        std::vector<int> remaining;
                        remaining.reserve(unmatched_dets.size());
                        for (int d : unmatched_dets) {
                            if (!matched_dets_set.count(d)) {
                                remaining.push_back(d);
                            }
                        }
                        unmatched_dets = std::move(remaining);
                        std::vector<int> remaining_tracks;
                        remaining_tracks.reserve(unmatched_trks.size());
                        for (int track : unmatched_trks) {
                            if (!matched_tracks_set.count(track)) {
                                remaining_tracks.push_back(track);
                            }
                        }
                        unmatched_trks = std::move(remaining_tracks);
                    }
                }
            }
        }
    }

    // ----- Second pass on low-confidence detections (appearance-gated IoU) -----
    if (config_.use_second_pass && !unmatched_trks.empty() && !dets_second.empty()) {
        std::vector<int> elig;
        for (int t : unmatched_trks) {
            const auto& trk = trackers_[t];
            if (trk->time_since_update <= config_.second_pass_max_age
                && trk->hit_streak >= config_.second_pass_min_hits
                && trk->is_activated) {
                elig.push_back(t);
            }
        }
        if (!elig.empty()) {
            Eigen::MatrixXd det_box_mat = DetectionsMatrix(dets_second);
            Eigen::MatrixXd trk_box_mat(static_cast<int>(elig.size()), 5);
            for (std::size_t j = 0; j < elig.size(); ++j) {
                const Eigen::Vector4d xyxy = trackers_[elig[j]]->xyxy();
                trk_box_mat(static_cast<int>(j), 0) = xyxy[0];
                trk_box_mat(static_cast<int>(j), 1) = xyxy[1];
                trk_box_mat(static_cast<int>(j), 2) = xyxy[2];
                trk_box_mat(static_cast<int>(j), 3) = xyxy[3];
                trk_box_mat(static_cast<int>(j), 4) = trackers_[elig[j]]->GetConfidence();
            }
            Eigen::MatrixXd ious = boxmot::trackers::base::AabbAssociationMatrix(
                det_box_mat,
                trk_box_mat,
                association_mode_,
                association_frame_width_,
                association_frame_height_
            );
            Eigen::MatrixXd cost = Eigen::MatrixXd::Constant(ious.rows(), ious.cols(), 1.0);
            for (int i = 0; i < ious.rows(); ++i) {
                for (int j = 0; j < ious.cols(); ++j) {
                    if (ious(i, j) >= config_.second_iou_thresh) {
                        cost(i, j) = 1.0 - ious(i, j);
                    }
                }
            }

            if (config_.with_reid && trackers_[elig.front()]->HasEmbedding()) {
                const int feat_dim = static_cast<int>(trackers_[elig.front()]->embedding().size());
                bool any_det_emb = false;
                for (const auto& d : dets_second) {
                    if (d.embedding.size() == feat_dim) {
                        any_det_emb = true;
                        break;
                    }
                }
                if (any_det_emb) {
                    Eigen::MatrixXf trk_emb_mat(static_cast<int>(elig.size()), feat_dim);
                    for (std::size_t j = 0; j < elig.size(); ++j) {
                        trk_emb_mat.row(static_cast<int>(j)) = trackers_[elig[j]]->embedding().transpose();
                    }
                    Eigen::MatrixXf det_emb_mat(static_cast<int>(dets_second.size()), feat_dim);
                    for (std::size_t i = 0; i < dets_second.size(); ++i) {
                        if (dets_second[i].embedding.size() == feat_dim) {
                            det_emb_mat.row(static_cast<int>(i)) = dets_second[i].embedding.transpose();
                        } else {
                            det_emb_mat.row(static_cast<int>(i)).setZero();
                        }
                    }
                    Eigen::MatrixXd sim2 = (det_emb_mat * trk_emb_mat.transpose()).cast<double>();
                    for (int i = 0; i < cost.rows(); ++i) {
                        for (int j = 0; j < cost.cols(); ++j) {
                            if (sim2(i, j) < config_.second_appearance_thresh) {
                                cost(i, j) = 1.0;
                            }
                        }
                    }
                }
            }

            bool any_match = false;
            for (int i = 0; i < cost.rows() && !any_match; ++i) {
                for (int j = 0; j < cost.cols(); ++j) {
                    if (cost(i, j) < 1.0) {
                        any_match = true;
                        break;
                    }
                }
            }
            if (any_match) {
                AssignmentResult hung = LinearAssignment(cost, std::numeric_limits<double>::infinity());
                std::unordered_set<int> used;
                for (const auto& [r, c] : hung.matches) {
                    if (cost(r, c) >= 1.0) {
                        continue;
                    }
                    const int trk_global = elig[c];
                    if (!used.insert(trk_global).second) {
                        continue;
                    }
                    AmsUpdate(*trackers_[trk_global], dets_second[r]);
                    if (config_.with_reid && dets_second[r].has_embedding()
                        && trackers_[trk_global]->embedding().size() == dets_second[r].embedding.size()) {
                        trackers_[trk_global]->UpdateEmbedding(dets_second[r].embedding, config_.feat_alpha);
                    }
                    MaybeActivate(*trackers_[trk_global]);
                }
            }
        }
    }

    // ----- Spawn new tracks from unmatched first-pass detections -----
    for (int i : unmatched_dets) {
        if (dets_first[i].conf < config_.new_track_thresh) {
            continue;
        }
        auto trk = std::make_shared<KalmanBoxTracker>(dets_first[i], config_.max_obs);
        trk->is_activated = (dets_first[i].conf >= config_.instant_confirm_thresh) || (config_.confirm_hits <= 1);
        trackers_.push_back(trk);
    }

    // ----- Build emit list (active tracks gating) -----
    std::vector<std::pair<KalmanBoxTracker::Ptr, Eigen::Vector4d>> emitted;
    emitted.reserve(trackers_.size());
    for (const auto& trk : trackers_) {
        const Eigen::Vector4d xyxy = trk->xyxy();
        const bool warmup = frame_count_ <= config_.min_hits;
        if (trk->time_since_update < 1
            && trk->is_activated
            && (trk->hit_streak >= config_.min_hits || warmup)) {
            emitted.emplace_back(trk, xyxy);
        }
    }

    // Duplicate suppression on emissions.
    if (emitted.size() > 1) {
        SuppressDuplicateEmissions(emitted);
    }

    std::vector<TrackOutput> outputs;
    outputs.reserve(emitted.size());
    for (auto& [trk, xyxy] : emitted) {
        if (!PassesFilter(xyxy)) {
            continue;
        }
        TrackOutput out;
        out.is_obb = false;
        out.id = trk->id;
        out.xyxy = xyxy;
        out.conf = trk->conf;
        out.cls = trk->cls;
        out.det_ind = trk->det_ind;
        outputs.push_back(out);
    }

    // Lifecycle filter: keep alive within max_age; tentative within tentative_max_age.
    std::vector<KalmanBoxTracker::Ptr> kept;
    kept.reserve(trackers_.size());
    for (auto& trk : trackers_) {
        if (trk->time_since_update > config_.max_age) {
            continue;
        }
        if (!trk->is_activated && trk->time_since_update > config_.tentative_max_age) {
            continue;
        }
        kept.push_back(trk);
    }
    trackers_ = std::move(kept);

    return outputs;
}

// ---------------------------------------------------------------------------
// OBB code path (mirrors Python OccluBoost._update_obb).
// ---------------------------------------------------------------------------
std::vector<TrackOutput> OccluBoostTracker::UpdateObb(
    const std::vector<Detection>& detections,
    const cv::Mat& image
) {
    ++frame_count_;
    last_reid_time_ms_ = 0.0;
    last_reid_preprocess_time_ms_ = 0.0;
    last_reid_process_time_ms_ = 0.0;
    last_reid_postprocess_time_ms_ = 0.0;

    if (cmc_) {
        const cv::Mat warp = cmc_->Apply(image, detections);
        if (!warp.empty() && warp.rows == 2 && warp.cols == 3) {
            Eigen::Matrix2d linear;
            linear << warp.at<float>(0, 0), warp.at<float>(0, 1),
                warp.at<float>(1, 0), warp.at<float>(1, 1);
            const Eigen::Vector2d translation(warp.at<float>(0, 2), warp.at<float>(1, 2));
            for (auto& tracker : trackers_) {
                tracker->CameraUpdate(linear, translation);
            }
        }
    }

    // Predict trackers and capture xywha + track confidences.
    Eigen::MatrixXd trks_xywha = Eigen::MatrixXd::Zero(static_cast<int>(trackers_.size()), 5);
    for (std::size_t i = 0; i < trackers_.size(); ++i) {
        trackers_[i]->Predict();
        trks_xywha.row(static_cast<int>(i)) = trackers_[i]->xywha().transpose();
    }

    // Capture original confidences before any changes.
    std::vector<float> orig_confs;
    orig_confs.reserve(detections.size());
    for (const auto& d : detections) {
        orig_confs.push_back(d.conf);
    }

    // ReID embeddings (or noop if cached/disabled).
    std::vector<Detection> working = EnsureEmbeddings(detections, image);

    if (config_.use_dlo_boost) {
        DloConfidenceBoostObb(working);
    }
    if (config_.use_duo_boost) {
        DuoConfidenceBoostObb(working);
    }

    // Split at the dedicated OBB operating point. Preserve original scores so
    // confidence promotion cannot make a detection eligible for both passes.
    std::vector<Detection> dets_first;
    std::vector<Detection> dets_second;
    dets_first.reserve(working.size());
    dets_second.reserve(working.size());
    for (std::size_t i = 0; i < working.size(); ++i) {
        const bool keep = working[i].conf >= config_.obb_det_thresh;
        if (keep) {
            dets_first.push_back(working[i]);
        } else if (config_.use_second_pass
                   && orig_confs[i] >= config_.track_low_thresh
                   && orig_confs[i] < config_.obb_det_thresh) {
            dets_second.push_back(working[i]);
        }
    }

    // Detection xywha matrix.
    Eigen::MatrixXd dets_xywha = Eigen::MatrixXd::Zero(static_cast<int>(dets_first.size()), 5);
    for (std::size_t i = 0; i < dets_first.size(); ++i) {
        dets_xywha.row(static_cast<int>(i)) = dets_first[i].xywha.transpose();
    }

    // First-pass association: oriented IoU (+ optional ReID fusion).
    Eigen::MatrixXd emb_sim(0, 0);
    if (config_.with_reid && !trackers_.empty() && !dets_first.empty()) {
        const int feat_dim = static_cast<int>(dets_first.front().embedding.size());
        bool ok = feat_dim > 0;
        for (const auto& d : dets_first) {
            if (d.embedding.size() != feat_dim) { ok = false; break; }
        }
        for (const auto& trk : trackers_) {
            if (!trk->HasEmbedding() || trk->embedding().size() != feat_dim) { ok = false; break; }
        }
        if (ok) {
            Eigen::MatrixXf det_emb_mat(static_cast<int>(dets_first.size()), feat_dim);
            for (std::size_t i = 0; i < dets_first.size(); ++i) {
                det_emb_mat.row(static_cast<int>(i)) = dets_first[i].embedding.transpose();
            }
            Eigen::MatrixXf trk_emb_mat(static_cast<int>(trackers_.size()), feat_dim);
            for (std::size_t j = 0; j < trackers_.size(); ++j) {
                trk_emb_mat.row(static_cast<int>(j)) = trackers_[j]->embedding().transpose();
            }
            emb_sim = (det_emb_mat * trk_emb_mat.transpose()).cast<double>();
        }
    }

    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_dets;
    std::vector<int> unmatched_trks;
    const int n_dets = static_cast<int>(dets_first.size());
    const int n_trks = static_cast<int>(trackers_.size());
    if (n_dets == 0 || n_trks == 0) {
        for (int i = 0; i < n_dets; ++i) unmatched_dets.push_back(i);
        for (int j = 0; j < n_trks; ++j) unmatched_trks.push_back(j);
    } else {
        const Eigen::MatrixXd iou = boxmot::trackers::base::ObbAssociationMatrix(
            dets_xywha,
            trks_xywha,
            association_mode_,
            association_frame_width_,
            association_frame_height_
        );
        Eigen::MatrixXd cost = Eigen::MatrixXd::Constant(n_dets, n_trks, 1.0);
        for (int i = 0; i < n_dets; ++i) {
            for (int j = 0; j < n_trks; ++j) {
                cost(i, j) = 1.0 - iou(i, j);
                if (iou(i, j) < config_.obb_iou_threshold) {
                    cost(i, j) = 1.0e6;
                }
            }
        }
        if (emb_sim.size() > 0) {
            const double lambda_emb = static_cast<double>(config_.lambda_iou) + 0.5;
            for (int i = 0; i < n_dets; ++i) {
                for (int j = 0; j < n_trks; ++j) {
                    cost(i, j) -= lambda_emb * emb_sim(i, j);
                    if (iou(i, j) < config_.obb_iou_threshold) {
                        cost(i, j) = 1.0e6;
                    }
                }
            }
        }
        AssignmentResult hung = LinearAssignment(cost, std::numeric_limits<double>::infinity());
        std::unordered_set<int> matched_d, matched_t;
        for (const auto& [r, c] : hung.matches) {
            if (cost(r, c) >= 1.0e5) continue;
            matches.emplace_back(r, c);
            matched_d.insert(r);
            matched_t.insert(c);
        }
        for (int i = 0; i < n_dets; ++i) if (!matched_d.count(i)) unmatched_dets.push_back(i);
        for (int j = 0; j < n_trks; ++j) if (!matched_t.count(j)) unmatched_trks.push_back(j);
    }

    // Apply matched updates.
    for (const auto& [d, t] : matches) {
        AmsUpdate(*trackers_[t], dets_first[d]);
        if (config_.with_reid && dets_first[d].has_embedding()) {
            const double trust = (dets_first[d].conf - config_.obb_det_thresh) /
                                 std::max(1.0 - config_.obb_det_thresh, 1.0e-6);
            constexpr double af = 0.95;
            const double alpha_emb = af + (1.0 - af) * (1.0 - trust);
            trackers_[t]->UpdateEmbedding(dets_first[d].embedding, alpha_emb);
        }
        MaybeActivate(*trackers_[t]);
    }

    // ----- ReID-only recovery pass -----
    if (config_.with_reid && !unmatched_trks.empty() && !unmatched_dets.empty()) {
        std::vector<int> elig;
        for (int t : unmatched_trks) {
            if (trackers_[t]->time_since_update <= config_.obb_recovery_max_age && trackers_[t]->HasEmbedding()) {
                elig.push_back(t);
            }
        }
        if (!elig.empty()) {
            const int feat_dim = static_cast<int>(trackers_[elig.front()]->embedding().size());
            std::vector<int> u_det_idx;
            u_det_idx.reserve(unmatched_dets.size());
            for (int d : unmatched_dets) {
                if (dets_first[d].embedding.size() == feat_dim) {
                    u_det_idx.push_back(d);
                }
            }
            if (!u_det_idx.empty()) {
                Eigen::MatrixXf trk_emb_mat(static_cast<int>(elig.size()), feat_dim);
                for (std::size_t j = 0; j < elig.size(); ++j) {
                    trk_emb_mat.row(static_cast<int>(j)) = trackers_[elig[j]]->embedding().transpose();
                }
                Eigen::MatrixXf det_emb_mat(static_cast<int>(u_det_idx.size()), feat_dim);
                for (std::size_t i = 0; i < u_det_idx.size(); ++i) {
                    det_emb_mat.row(static_cast<int>(i)) = dets_first[u_det_idx[i]].embedding.transpose();
                }
                Eigen::MatrixXd sim = (det_emb_mat * trk_emb_mat.transpose()).cast<double>();

                Eigen::MatrixXd det_xywha_mat(static_cast<int>(u_det_idx.size()), 5);
                for (std::size_t i = 0; i < u_det_idx.size(); ++i) {
                    det_xywha_mat.row(static_cast<int>(i)) = dets_first[u_det_idx[i]].xywha.transpose();
                }
                Eigen::MatrixXd trk_xywha_mat(static_cast<int>(elig.size()), 5);
                for (std::size_t j = 0; j < elig.size(); ++j) {
                    trk_xywha_mat.row(static_cast<int>(j)) = trackers_[elig[j]]->xywha().transpose();
                }
                const Eigen::MatrixXd ious = boxmot::trackers::base::ObbAssociationMatrix(
                    det_xywha_mat,
                    trk_xywha_mat,
                    association_mode_,
                    association_frame_width_,
                    association_frame_height_
                );

                Eigen::MatrixXd gated = sim;
                bool any_pos = false;
                for (int i = 0; i < gated.rows(); ++i) {
                    for (int j = 0; j < gated.cols(); ++j) {
                        if (ious(i, j) < config_.recovery_iou_thresh
                            || sim(i, j) < config_.recovery_appearance_thresh) {
                            gated(i, j) = -1.0;
                        } else if (gated(i, j) > 0.0) {
                            any_pos = true;
                        }
                    }
                }
                if (any_pos) {
                    Eigen::MatrixXd cost = -gated;
                    AssignmentResult hung = LinearAssignment(cost, std::numeric_limits<double>::infinity());
                    std::unordered_set<int> matched_dets_set;
                    std::unordered_set<int> matched_tracks_set;
                    for (const auto& [r, c] : hung.matches) {
                        if (gated(r, c) <= 0.0) continue;
                        const int det_global = u_det_idx[r];
                        const int trk_global = elig[c];
                        matched_dets_set.insert(det_global);
                        matched_tracks_set.insert(trk_global);
                        AmsUpdate(*trackers_[trk_global], dets_first[det_global]);
                        if (dets_first[det_global].has_embedding()) {
                            trackers_[trk_global]->UpdateEmbedding(dets_first[det_global].embedding, config_.feat_alpha);
                        }
                        MaybeActivate(*trackers_[trk_global]);
                    }
                    if (!matched_dets_set.empty()) {
                        std::vector<int> remaining;
                        for (int d : unmatched_dets) {
                            if (!matched_dets_set.count(d)) remaining.push_back(d);
                        }
                        unmatched_dets = std::move(remaining);
                        std::vector<int> remaining_tracks;
                        remaining_tracks.reserve(unmatched_trks.size());
                        for (int track : unmatched_trks) {
                            if (!matched_tracks_set.count(track)) {
                                remaining_tracks.push_back(track);
                            }
                        }
                        unmatched_trks = std::move(remaining_tracks);
                    }
                }
            }
        }
    }

    // ----- Second pass on low-confidence detections (oriented IoU + appearance gate) -----
    if (config_.use_second_pass && !unmatched_trks.empty() && !dets_second.empty()) {
        std::vector<int> elig;
        for (int t : unmatched_trks) {
            const auto& trk = trackers_[t];
            if (trk->time_since_update <= config_.second_pass_max_age
                && trk->hit_streak >= config_.second_pass_min_hits
                && trk->is_activated) {
                elig.push_back(t);
            }
        }
        if (!elig.empty()) {
            Eigen::MatrixXd det_xywha_mat(static_cast<int>(dets_second.size()), 5);
            for (std::size_t i = 0; i < dets_second.size(); ++i) {
                det_xywha_mat.row(static_cast<int>(i)) = dets_second[i].xywha.transpose();
            }
            Eigen::MatrixXd trk_xywha_mat(static_cast<int>(elig.size()), 5);
            for (std::size_t j = 0; j < elig.size(); ++j) {
                trk_xywha_mat.row(static_cast<int>(j)) = trackers_[elig[j]]->xywha().transpose();
            }
            const Eigen::MatrixXd ious = boxmot::trackers::base::ObbAssociationMatrix(
                det_xywha_mat,
                trk_xywha_mat,
                association_mode_,
                association_frame_width_,
                association_frame_height_
            );
            Eigen::MatrixXd cost = Eigen::MatrixXd::Constant(ious.rows(), ious.cols(), 1.0);
            for (int i = 0; i < ious.rows(); ++i) {
                for (int j = 0; j < ious.cols(); ++j) {
                    if (ious(i, j) >= config_.obb_second_iou_thresh) {
                        cost(i, j) = 1.0 - ious(i, j);
                    }
                }
            }
            if (config_.with_reid && trackers_[elig.front()]->HasEmbedding()) {
                const int feat_dim = static_cast<int>(trackers_[elig.front()]->embedding().size());
                bool any_det_emb = false;
                for (const auto& d : dets_second) {
                    if (d.embedding.size() == feat_dim) { any_det_emb = true; break; }
                }
                if (any_det_emb) {
                    Eigen::MatrixXf trk_emb_mat(static_cast<int>(elig.size()), feat_dim);
                    for (std::size_t j = 0; j < elig.size(); ++j) {
                        trk_emb_mat.row(static_cast<int>(j)) = trackers_[elig[j]]->embedding().transpose();
                    }
                    Eigen::MatrixXf det_emb_mat(static_cast<int>(dets_second.size()), feat_dim);
                    for (std::size_t i = 0; i < dets_second.size(); ++i) {
                        if (dets_second[i].embedding.size() == feat_dim) {
                            det_emb_mat.row(static_cast<int>(i)) = dets_second[i].embedding.transpose();
                        } else {
                            det_emb_mat.row(static_cast<int>(i)).setZero();
                        }
                    }
                    Eigen::MatrixXd sim2 = (det_emb_mat * trk_emb_mat.transpose()).cast<double>();
                    for (int i = 0; i < cost.rows(); ++i) {
                        for (int j = 0; j < cost.cols(); ++j) {
                            if (sim2(i, j) < config_.second_appearance_thresh) {
                                cost(i, j) = 1.0;
                            }
                        }
                    }
                }
            }
            bool any_match = false;
            for (int i = 0; i < cost.rows() && !any_match; ++i) {
                for (int j = 0; j < cost.cols(); ++j) {
                    if (cost(i, j) < 1.0) { any_match = true; break; }
                }
            }
            if (any_match) {
                AssignmentResult hung = LinearAssignment(cost, std::numeric_limits<double>::infinity());
                std::unordered_set<int> used;
                for (const auto& [r, c] : hung.matches) {
                    if (cost(r, c) >= 1.0) continue;
                    const int trk_global = elig[c];
                    if (!used.insert(trk_global).second) continue;
                    AmsUpdate(*trackers_[trk_global], dets_second[r]);
                    if (config_.with_reid && dets_second[r].has_embedding()
                        && trackers_[trk_global]->embedding().size() == dets_second[r].embedding.size()) {
                        trackers_[trk_global]->UpdateEmbedding(dets_second[r].embedding, config_.feat_alpha);
                    }
                    MaybeActivate(*trackers_[trk_global]);
                }
            }
        }
    }

    // ----- Spawn new tracks from unmatched first-pass detections -----
    for (int i : unmatched_dets) {
        if (dets_first[i].conf < config_.obb_new_track_thresh) continue;
        auto trk = std::make_shared<KalmanBoxTracker>(dets_first[i], config_.max_obs);
        trk->is_activated =
            (dets_first[i].conf >= config_.obb_instant_confirm_thresh) || (config_.confirm_hits <= 1);
        trackers_.push_back(trk);
    }

    // ----- Build emit list -----
    std::vector<std::pair<KalmanBoxTracker::Ptr, Eigen::Vector4d>> emitted;
    emitted.reserve(trackers_.size());
    for (const auto& trk : trackers_) {
        const Eigen::Vector4d aabb = trk->xyxy();
        const bool warmup = frame_count_ <= config_.min_hits;
        if (trk->time_since_update < 1
            && trk->is_activated
            && (trk->hit_streak >= config_.min_hits || warmup)) {
            emitted.emplace_back(trk, aabb);
        }
    }
    if (emitted.size() > 1) {
        SuppressDuplicateEmissions(emitted);
    }

    std::vector<TrackOutput> outputs;
    outputs.reserve(emitted.size());
    for (auto& [trk, aabb] : emitted) {
        const Eigen::Matrix<double, 5, 1> xywha = trk->xywha();
        if (!PassesObbFilter(xywha)) {
            continue;
        }
        TrackOutput out;
        out.is_obb = true;
        out.id = trk->id;
        out.xyxy = aabb;
        out.xywha = xywha;
        out.conf = trk->conf;
        out.cls = trk->cls;
        out.det_ind = trk->det_ind;
        outputs.push_back(out);
    }

    // Lifecycle filter.
    std::vector<KalmanBoxTracker::Ptr> kept;
    kept.reserve(trackers_.size());
    for (auto& trk : trackers_) {
        if (trk->time_since_update > config_.obb_max_age) continue;
        if (!trk->is_activated && trk->time_since_update > config_.tentative_max_age) continue;
        kept.push_back(trk);
    }
    trackers_ = std::move(kept);

    return outputs;
}

}  // namespace occluboost
