#include "occluboost/association.hpp"

#include "boxmot/trackers/base/assignment.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace occluboost {

namespace {

constexpr double kMhdLimit = 13.2767;  // chi^2 99% interval, dof=4.
constexpr double kPi = 3.14159265358979323846;

cv::RotatedRect RotatedRectFromXywha(const Eigen::Matrix<double, 5, 1>& box) {
    return cv::RotatedRect(
        cv::Point2f(static_cast<float>(box[0]), static_cast<float>(box[1])),
        cv::Size2f(
            static_cast<float>(std::max(box[2], 1.0e-4)),
            static_cast<float>(std::max(box[3], 1.0e-4))
        ),
        static_cast<float>(box[4] * 180.0 / kPi)
    );
}

double ObbIoU(const Eigen::Matrix<double, 5, 1>& lhs, const Eigen::Matrix<double, 5, 1>& rhs) {
    const cv::RotatedRect lhs_rect = RotatedRectFromXywha(lhs);
    const cv::RotatedRect rhs_rect = RotatedRectFromXywha(rhs);

    std::vector<cv::Point2f> intersection;
    const int status = cv::rotatedRectangleIntersection(lhs_rect, rhs_rect, intersection);
    if (status == cv::INTERSECT_NONE || intersection.empty()) {
        return 0.0;
    }

    const double inter_area = std::abs(cv::contourArea(intersection));
    const double lhs_area = std::max(lhs[2], 0.0) * std::max(lhs[3], 0.0);
    const double rhs_area = std::max(rhs[2], 0.0) * std::max(rhs[3], 0.0);
    const double denom = lhs_area + rhs_area - inter_area;
    if (denom <= 1.0e-12) {
        return 0.0;
    }
    return inter_area / denom;
}

}  // namespace

Eigen::MatrixXd IouBatch(const Eigen::MatrixXd& dets, const Eigen::MatrixXd& trks) {
    const int n = static_cast<int>(dets.rows());
    const int m = static_cast<int>(trks.rows());
    Eigen::MatrixXd iou = Eigen::MatrixXd::Zero(n, m);
    if (n == 0 || m == 0) {
        return iou;
    }
    for (int i = 0; i < n; ++i) {
        const double dx1 = dets(i, 0), dy1 = dets(i, 1), dx2 = dets(i, 2), dy2 = dets(i, 3);
        const double da = std::max(0.0, dx2 - dx1) * std::max(0.0, dy2 - dy1);
        for (int j = 0; j < m; ++j) {
            const double tx1 = trks(j, 0), ty1 = trks(j, 1), tx2 = trks(j, 2), ty2 = trks(j, 3);
            const double xx1 = std::max(dx1, tx1);
            const double yy1 = std::max(dy1, ty1);
            const double xx2 = std::min(dx2, tx2);
            const double yy2 = std::min(dy2, ty2);
            const double w = std::max(0.0, xx2 - xx1);
            const double h = std::max(0.0, yy2 - yy1);
            const double inter = w * h;
            const double ta = std::max(0.0, tx2 - tx1) * std::max(0.0, ty2 - ty1);
            const double denom = da + ta - inter;
            iou(i, j) = denom > 0.0 ? inter / denom : 0.0;
        }
    }
    return iou;
}

Eigen::MatrixXd IouBatchObb(const Eigen::MatrixXd& dets_xywha, const Eigen::MatrixXd& trks_xywha) {
    const int n = static_cast<int>(dets_xywha.rows());
    const int m = static_cast<int>(trks_xywha.rows());
    Eigen::MatrixXd iou = Eigen::MatrixXd::Zero(n, m);
    if (n == 0 || m == 0 || dets_xywha.cols() < 5 || trks_xywha.cols() < 5) {
        return iou;
    }
    for (int i = 0; i < n; ++i) {
        Eigen::Matrix<double, 5, 1> a;
        a << dets_xywha(i, 0), dets_xywha(i, 1), dets_xywha(i, 2), dets_xywha(i, 3), dets_xywha(i, 4);
        for (int j = 0; j < m; ++j) {
            Eigen::Matrix<double, 5, 1> b;
            b << trks_xywha(j, 0), trks_xywha(j, 1), trks_xywha(j, 2), trks_xywha(j, 3), trks_xywha(j, 4);
            iou(i, j) = ObbIoU(a, b);
        }
    }
    return iou;
}

Eigen::MatrixXd SoftBiouBatch(const Eigen::MatrixXd& dets, const Eigen::MatrixXd& trks) {
    // bboxes2 (trks) supplies the confidence in column 4.
    const int n = static_cast<int>(dets.rows());
    const int m = static_cast<int>(trks.rows());
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n, m);
    if (n == 0 || m == 0 || trks.cols() < 5) {
        return out;
    }
    constexpr double k1 = 0.25;
    constexpr double k2 = 0.5;
    for (int i = 0; i < n; ++i) {
        const double bx1 = dets(i, 0), by1 = dets(i, 1), bx2 = dets(i, 2), by2 = dets(i, 3);
        const double bw = bx2 - bx1, bh = by2 - by1;
        for (int j = 0; j < m; ++j) {
            const double conf = trks(j, 4);
            const double bias = 1.0 - conf;
            const double b1x1 = bx1 - bw * bias * k1;
            const double b1y1 = by1 - bh * bias * k1;
            const double b1x2 = bx2 + bw * bias * k1;
            const double b1y2 = by2 + bh * bias * k1;

            const double tx1 = trks(j, 0), ty1 = trks(j, 1), tx2 = trks(j, 2), ty2 = trks(j, 3);
            const double tw = tx2 - tx1, th = ty2 - ty1;
            const double b2x1 = tx1 - tw * bias * k2;
            const double b2y1 = ty1 - th * bias * k2;
            const double b2x2 = tx2 + tw * bias * k2;
            const double b2y2 = ty2 + th * bias * k2;

            const double xx1 = std::max(b1x1, b2x1);
            const double yy1 = std::max(b1y1, b2y1);
            const double xx2 = std::min(b1x2, b2x2);
            const double yy2 = std::min(b1y2, b2y2);
            const double w = std::max(0.0, xx2 - xx1);
            const double h = std::max(0.0, yy2 - yy1);
            const double inter = w * h;
            const double da = (b1x2 - b1x1) * (b1y2 - b1y1);
            const double ta = (b2x2 - b2x1) * (b2y2 - b2y1);
            const double denom = da + ta - inter;
            out(i, j) = denom > 0.0 ? inter / denom : 0.0;
        }
    }
    return out;
}

Eigen::MatrixXd ShapeSimilarityV1(const Eigen::MatrixXd& dets, const Eigen::MatrixXd& trks) {
    const int n = static_cast<int>(dets.rows());
    const int m = static_cast<int>(trks.rows());
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n, m);
    if (n == 0 || m == 0) {
        return out;
    }
    for (int i = 0; i < n; ++i) {
        const double dw = dets(i, 2) - dets(i, 0);
        const double dh = dets(i, 3) - dets(i, 1);
        for (int j = 0; j < m; ++j) {
            const double tw = trks(j, 2) - trks(j, 0);
            const double th = trks(j, 3) - trks(j, 1);
            const double max_w = std::max(dw, tw);
            // Note: shape_similarity_v1 reuses max(dw, tw) for the height term.
            const double term = std::abs(dw - tw) / std::max(max_w, 1.0e-12)
                              + std::abs(dh - th) / std::max(max_w, 1.0e-12);
            out(i, j) = std::exp(-term);
        }
    }
    return out;
}

Eigen::MatrixXd MhDistSimilarity(const Eigen::MatrixXd& mh_dist) {
    const int n = static_cast<int>(mh_dist.rows());
    const int m = static_cast<int>(mh_dist.cols());
    if (n == 0 || m == 0) {
        return Eigen::MatrixXd::Zero(n, m);
    }
    Eigen::MatrixXd clamped = mh_dist;
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> mask(n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            mask(i, j) = clamped(i, j) > kMhdLimit;
            if (mask(i, j)) {
                clamped(i, j) = kMhdLimit;
            }
            clamped(i, j) = kMhdLimit - clamped(i, j);
        }
    }
    // exp(d) / sum_over_axis_0(exp(d))  -- per-column softmax (axis=0 in numpy).
    Eigen::MatrixXd exped(n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            exped(i, j) = std::exp(clamped(i, j));
        }
    }
    Eigen::RowVectorXd col_sum = exped.colwise().sum();
    for (int j = 0; j < m; ++j) {
        const double denom = col_sum(j) > 0.0 ? col_sum(j) : 1.0e-12;
        for (int i = 0; i < n; ++i) {
            exped(i, j) /= denom;
            if (mask(i, j)) {
                exped(i, j) = 0.0;
            }
        }
    }
    return exped;
}

AssociationResult Associate(
    const Eigen::MatrixXd& detections,
    const Eigen::MatrixXd& trackers,
    const double iou_threshold,
    const Eigen::MatrixXd& mh_dist,
    const Eigen::VectorXd& detection_confidence,
    const Eigen::VectorXd& track_confidence,
    const Eigen::MatrixXd& emb_cost,
    const double lambda_iou,
    const double lambda_mhd,
    const double lambda_shape
) {
    AssociationResult result;
    const int nd = static_cast<int>(detections.rows());
    const int nt = static_cast<int>(trackers.rows());
    if (nt == 0) {
        result.unmatched_dets.resize(nd);
        for (int i = 0; i < nd; ++i) {
            result.unmatched_dets[i] = i;
        }
        result.iou_matrix = Eigen::MatrixXd::Zero(nd, 0);
        return result;
    }
    if (nd == 0) {
        result.unmatched_trks.resize(nt);
        for (int j = 0; j < nt; ++j) {
            result.unmatched_trks[j] = j;
        }
        result.iou_matrix = Eigen::MatrixXd::Zero(0, nt);
        return result;
    }

    Eigen::MatrixXd iou_matrix = IouBatch(detections, trackers);
    Eigen::MatrixXd cost_matrix = iou_matrix;

    // Confidence-weighted IoU contribution.
    Eigen::MatrixXd conf = Eigen::MatrixXd::Zero(nd, nt);
    const bool has_conf = detection_confidence.size() == nd && track_confidence.size() == nt;
    if (has_conf) {
        for (int i = 0; i < nd; ++i) {
            for (int j = 0; j < nt; ++j) {
                conf(i, j) = detection_confidence(i) * track_confidence(j);
                if (iou_matrix(i, j) < iou_threshold) {
                    conf(i, j) = 0.0;
                }
            }
        }
        cost_matrix += lambda_iou * conf.cwiseProduct(IouBatch(detections, trackers));
    }

    if (mh_dist.size() > 0) {
        const Eigen::MatrixXd mh_sim = MhDistSimilarity(mh_dist);
        cost_matrix += lambda_mhd * mh_sim;
        if (has_conf) {
            const Eigen::MatrixXd shape_sim = ShapeSimilarityV1(detections, trackers);
            cost_matrix += lambda_shape * conf.cwiseProduct(shape_sim);
        }
    }

    if (emb_cost.size() > 0) {
        const double lambda_emb = (1.0 + lambda_iou + lambda_shape + lambda_mhd) * 1.5;
        cost_matrix += lambda_emb * emb_cost;
    }

    // Hungarian min-cost assignment over -cost_matrix; threshold = +inf so we
    // accept all assignments and rely on the post-filter for validity. The
    // base solver clamps an infinite threshold internally so the rectangular
    // padding stays finite.
    Eigen::MatrixXd hungarian_cost = -cost_matrix;
    boxmot::trackers::base::AssignmentResult hungarian =
        boxmot::trackers::base::LinearAssignment(hungarian_cost, std::numeric_limits<double>::infinity());

    std::vector<bool> det_matched(nd, false);
    std::vector<bool> trk_matched(nt, false);
    for (const auto& match : hungarian.matches) {
        const int d = match.first;
        const int t = match.second;
        const bool iou_ok = iou_matrix(d, t) >= iou_threshold;
        bool emb_ok = false;
        if (emb_cost.size() > 0
            && iou_matrix(d, t) >= 0.5 * iou_threshold
            && emb_cost(d, t) >= 0.75) {
            emb_ok = true;
        }
        if (iou_ok || emb_ok) {
            result.matches.emplace_back(d, t);
            det_matched[d] = true;
            trk_matched[t] = true;
        }
    }

    for (int i = 0; i < nd; ++i) {
        if (!det_matched[i]) {
            result.unmatched_dets.push_back(i);
        }
    }
    for (int j = 0; j < nt; ++j) {
        if (!trk_matched[j]) {
            result.unmatched_trks.push_back(j);
        }
    }
    result.iou_matrix = std::move(iou_matrix);
    return result;
}

}  // namespace occluboost
