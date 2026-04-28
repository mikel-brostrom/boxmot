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

OccluBoostTracker::OccluBoostTracker(Config config)
    : config_(std::move(config)),
      cmc_(CreateCameraMotionCompensator(config_.cmc_method)) {
    KalmanBoxTracker::ResetCount();
    if (config_.with_reid && !config_.reid_model_path.empty()) {
        reid_model_ = MaybeCreateOnnxReIdModel(config_.reid_model_path, config_.reid_preprocess);
    }
}

void OccluBoostTracker::Reset() {
    frame_count_ = 0;
    trackers_.clear();
    cmc_ = CreateCameraMotionCompensator(config_.cmc_method);
    KalmanBoxTracker::ResetCount();
    reid_model_.reset();
    if (config_.with_reid && !config_.reid_model_path.empty()) {
        reid_model_ = MaybeCreateOnnxReIdModel(config_.reid_model_path, config_.reid_preprocess);
    }
    last_reid_time_ms_ = 0.0;
}

std::vector<Detection> OccluBoostTracker::EnsureEmbeddings(
    std::vector<Detection> detections,
    const cv::Mat& image
) {
    last_reid_time_ms_ = 0.0;
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
    const auto t0 = std::chrono::steady_clock::now();
    const auto features = reid_model_->GetFeatures(detections, image);
    const auto t1 = std::chrono::steady_clock::now();
    last_reid_time_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (features.size() != detections.size()) {
        throw std::runtime_error("Native OccluBoost ReID returned a different number of embeddings than detections.");
    }
    for (std::size_t i = 0; i < detections.size(); ++i) {
        detections[i].embedding = features[i];
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

void OccluBoostTracker::SuppressDuplicateEmissions(
    std::vector<std::pair<KalmanBoxTracker::Ptr, Eigen::Vector4d>>& emitted
) {
    const int n = static_cast<int>(emitted.size());
    if (n <= 1 || config_.duplicate_iou_thresh <= 0.0F || config_.duplicate_iou_thresh >= 1.0F) {
        return;
    }
    Eigen::MatrixXd boxes(n, 4);
    for (int i = 0; i < n; ++i) {
        boxes.row(i) = emitted[i].second.transpose();
    }
    Eigen::MatrixXd ious = IouBatch(boxes, boxes);
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
    ++frame_count_;
    last_reid_time_ms_ = 0.0;

    // Reject any OBB inputs (OccluBoost native is AABB-only).
    for (const auto& d : detections) {
        if (d.is_obb) {
            throw std::runtime_error("Native OccluBoost only supports AABB detections.");
        }
    }

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
        config_.lambda_shape
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
                Eigen::MatrixXd ious = IouBatch(det_box_mat, trk_box_mat);

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
                    for (const auto& [r, c] : hung.matches) {
                        if (gated(r, c) <= 0.0) {
                            continue;
                        }
                        const int det_global = u_det_idx[r];
                        const int trk_global = elig[c];
                        matched_dets_set.insert(det_global);
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
            Eigen::MatrixXd ious = IouBatch(det_box_mat, trk_box_mat);
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

}  // namespace occluboost
