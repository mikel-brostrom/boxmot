#include "boxmot/trackers/base/association.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace boxmot::trackers::base {

namespace {

constexpr double kEpsilon = 1.0e-12;
constexpr double kEquivalentObbAngleTolerance = 1.0e-6;
constexpr double kEquivalentObbLinearTolerance = 1.0e-9;
constexpr double kMinimumObbSide = 1.0e-4;
constexpr double kPi = 3.14159265358979323846;

using ObbGeometry = Eigen::Matrix<double, 5, 1>;

std::string NormalizeName(const std::string_view value) {
    std::string normalized(value);
    const auto first =
        std::find_if_not(normalized.begin(), normalized.end(), [](const unsigned char ch) {
            return std::isspace(ch) != 0;
        });
    const auto last =
        std::find_if_not(normalized.rbegin(), normalized.rend(), [](const unsigned char ch) {
            return std::isspace(ch) != 0;
        }).base();
    normalized = first < last ? std::string(first, last) : std::string();
    std::transform(
        normalized.begin(), normalized.end(), normalized.begin(), [](const unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
    return normalized;
}

struct AabbTerms {
    double intersection = 0.0;
    double union_area = 0.0;
    double iou = 0.0;
    double enclosing_width = 0.0;
    double enclosing_height = 0.0;
    double center_distance_squared = 0.0;
};

AabbTerms ComputeAabbTerms(const Eigen::Vector4d& lhs, const Eigen::Vector4d& rhs) {
    const double intersection_width =
        std::max(0.0, std::min(lhs[2], rhs[2]) - std::max(lhs[0], rhs[0]));
    const double intersection_height =
        std::max(0.0, std::min(lhs[3], rhs[3]) - std::max(lhs[1], rhs[1]));
    const double lhs_width = std::max(0.0, lhs[2] - lhs[0]);
    const double lhs_height = std::max(0.0, lhs[3] - lhs[1]);
    const double rhs_width = std::max(0.0, rhs[2] - rhs[0]);
    const double rhs_height = std::max(0.0, rhs[3] - rhs[1]);

    AabbTerms terms;
    terms.intersection = intersection_width * intersection_height;
    terms.union_area = lhs_width * lhs_height + rhs_width * rhs_height - terms.intersection;
    terms.iou = terms.union_area > kEpsilon ? terms.intersection / terms.union_area : 0.0;
    terms.enclosing_width = std::max(lhs[2], rhs[2]) - std::min(lhs[0], rhs[0]);
    terms.enclosing_height = std::max(lhs[3], rhs[3]) - std::min(lhs[1], rhs[1]);
    const double lhs_center_x = (lhs[0] + lhs[2]) / 2.0;
    const double lhs_center_y = (lhs[1] + lhs[3]) / 2.0;
    const double rhs_center_x = (rhs[0] + rhs[2]) / 2.0;
    const double rhs_center_y = (rhs[1] + rhs[3]) / 2.0;
    terms.center_distance_squared =
        std::pow(lhs_center_x - rhs_center_x, 2.0) + std::pow(lhs_center_y - rhs_center_y, 2.0);
    return terms;
}

double CentroidSimilarity(const double lhs_x,
                          const double lhs_y,
                          const double rhs_x,
                          const double rhs_y,
                          const int frame_width,
                          const int frame_height) {
    if (frame_width <= 0 || frame_height <= 0) {
        throw std::runtime_error("Centroid association requires positive frame dimensions.");
    }
    const double frame_diagonal =
        std::hypot(static_cast<double>(frame_width), static_cast<double>(frame_height));
    return 1.0 - (std::hypot(lhs_x - rhs_x, lhs_y - rhs_y) / frame_diagonal);
}

ObbGeometry NormalizeObbGeometry(const ObbGeometry& box) {
    ObbGeometry normalized = box;
    normalized[2] = std::max(normalized[2], kMinimumObbSide);
    normalized[3] = std::max(normalized[3], kMinimumObbSide);
    return normalized;
}

bool ObbGeometryEquivalent(const ObbGeometry& lhs, const ObbGeometry& rhs) {
    const auto nearly_equal = [](const double first, const double second) {
        return std::abs(first - second) <= kEquivalentObbLinearTolerance;
    };
    if (!nearly_equal(lhs[0], rhs[0]) || !nearly_equal(lhs[1], rhs[1])) {
        return false;
    }

    const double angle_difference = lhs[4] - rhs[4];
    const bool direct =
        nearly_equal(lhs[2], rhs[2]) && nearly_equal(lhs[3], rhs[3]) &&
        std::abs(std::remainder(angle_difference, kPi)) <= kEquivalentObbAngleTolerance;
    const bool swapped =
        nearly_equal(lhs[2], rhs[3]) && nearly_equal(lhs[3], rhs[2]) &&
        std::abs(std::remainder(angle_difference - kPi / 2.0, kPi)) <= kEquivalentObbAngleTolerance;
    return direct || swapped;
}

struct LocalObbPair {
    ObbGeometry lhs;
    ObbGeometry rhs;
};

LocalObbPair CenterObbPair(const ObbGeometry& lhs, const ObbGeometry& rhs) {
    const double origin_x = lhs[0] + (rhs[0] - lhs[0]) / 2.0;
    const double origin_y = lhs[1] + (rhs[1] - lhs[1]) / 2.0;
    LocalObbPair local{lhs, rhs};
    local.lhs[0] -= origin_x;
    local.lhs[1] -= origin_y;
    local.rhs[0] -= origin_x;
    local.rhs[1] -= origin_y;
    return local;
}

cv::RotatedRect RotatedRectFromXywha(const ObbGeometry& box) {
    return cv::RotatedRect(cv::Point2f(static_cast<float>(box[0]), static_cast<float>(box[1])),
                           cv::Size2f(static_cast<float>(box[2]), static_cast<float>(box[3])),
                           static_cast<float>(box[4] * 180.0 / kPi));
}

struct ObbOverlapTerms {
    double union_area = 0.0;
    double iou = 0.0;
};

ObbOverlapTerms ComputeObbOverlapTerms(const ObbGeometry& lhs, const ObbGeometry& rhs) {
    const LocalObbPair local = CenterObbPair(lhs, rhs);
    std::vector<cv::Point2f> intersection;
    const int status = cv::rotatedRectangleIntersection(
        RotatedRectFromXywha(local.lhs), RotatedRectFromXywha(local.rhs), intersection);
    const double intersection_area = status == cv::INTERSECT_NONE || intersection.empty()
                                         ? 0.0
                                         : std::abs(cv::contourArea(intersection));
    const double lhs_area = lhs[2] * lhs[3];
    const double rhs_area = rhs[2] * rhs[3];
    const double area_sum = lhs_area + rhs_area;
    const double raw_union_area = area_sum - intersection_area;
    ObbOverlapTerms terms;
    terms.iou =
        raw_union_area > kEpsilon ? std::clamp(intersection_area / raw_union_area, 0.0, 1.0) : 0.0;
    terms.union_area = area_sum / (1.0 + terms.iou);
    return terms;
}

double ObbConvexEnclosureArea(const ObbGeometry& lhs, const ObbGeometry& rhs) {
    const LocalObbPair local = CenterObbPair(lhs, rhs);
    std::array<cv::Point2f, 4> lhs_corners;
    std::array<cv::Point2f, 4> rhs_corners;
    RotatedRectFromXywha(local.lhs).points(lhs_corners.data());
    RotatedRectFromXywha(local.rhs).points(rhs_corners.data());

    std::vector<cv::Point2f> corners;
    corners.reserve(lhs_corners.size() + rhs_corners.size());
    corners.insert(corners.end(), lhs_corners.begin(), lhs_corners.end());
    corners.insert(corners.end(), rhs_corners.begin(), rhs_corners.end());

    std::vector<cv::Point2f> hull;
    cv::convexHull(corners, hull);
    return hull.empty() ? 0.0 : std::abs(cv::contourArea(hull));
}

Eigen::Vector4d ObbEnclosingAabb(const ObbGeometry& box) {
    const double half_width = box[2] / 2.0;
    const double half_height = box[3] / 2.0;
    const double cos_angle = std::abs(std::cos(box[4]));
    const double sin_angle = std::abs(std::sin(box[4]));
    const double extent_x = half_width * cos_angle + half_height * sin_angle;
    const double extent_y = half_width * sin_angle + half_height * cos_angle;
    return Eigen::Vector4d(
        box[0] - extent_x, box[1] - extent_y, box[0] + extent_x, box[1] + extent_y);
}

}  // namespace

AssociationMode ParseAssociationMode(const std::string_view name) {
    static const std::unordered_map<std::string, AssociationMode> modes = {
        {"iou", AssociationMode::kIou},
        {"giou", AssociationMode::kGiou},
        {"diou", AssociationMode::kDiou},
        {"ciou", AssociationMode::kCiou},
        {"hmiou", AssociationMode::kHmiou},
        {"centroid", AssociationMode::kCentroid},
    };
    const std::string normalized = NormalizeName(name);
    const auto found = modes.find(normalized);
    if (found == modes.end()) {
        throw std::invalid_argument("Unknown association function '" + normalized +
                                    "'. Choose from: centroid, ciou, diou, giou, hmiou, iou.");
    }
    return found->second;
}

std::string AssociationModeName(const AssociationMode mode) {
    switch (mode) {
        case AssociationMode::kIou:
            return "iou";
        case AssociationMode::kGiou:
            return "giou";
        case AssociationMode::kDiou:
            return "diou";
        case AssociationMode::kCiou:
            return "ciou";
        case AssociationMode::kHmiou:
            return "hmiou";
        case AssociationMode::kCentroid:
            return "centroid";
    }
    throw std::invalid_argument("Unknown association mode value.");
}

bool AssociationModeRequiresFrameDimensions(const AssociationMode mode) noexcept {
    return mode == AssociationMode::kCentroid;
}

bool AssociationModeSupportsObb(const AssociationMode mode) noexcept {
    return mode == AssociationMode::kIou || mode == AssociationMode::kGiou ||
           mode == AssociationMode::kDiou || mode == AssociationMode::kCiou ||
           mode == AssociationMode::kHmiou || mode == AssociationMode::kCentroid;
}

void ValidateAssociationModeForDetections(const AssociationMode mode, const bool is_obb) {
    if (is_obb && !AssociationModeSupportsObb(mode)) {
        throw std::invalid_argument(
            "Association function '" + AssociationModeName(mode) +
            "' has no oriented-box implementation. Choose from: centroid, ciou, diou, giou, "
            "hmiou, iou.");
    }
}

double AabbAssociationSimilarity(const Eigen::Vector4d& lhs,
                                 const Eigen::Vector4d& rhs,
                                 const AssociationMode mode,
                                 const int frame_width,
                                 const int frame_height) {
    const AabbTerms terms = ComputeAabbTerms(lhs, rhs);
    switch (mode) {
        case AssociationMode::kIou:
            return terms.iou;
        case AssociationMode::kGiou: {
            const double enclosing_area = terms.enclosing_width * terms.enclosing_height;
            const double giou = terms.iou - ((enclosing_area - terms.union_area) /
                                             std::max(enclosing_area, kEpsilon));
            return (giou + 1.0) / 2.0;
        }
        case AssociationMode::kDiou: {
            const double enclosing_diagonal =
                std::pow(terms.enclosing_width, 2.0) + std::pow(terms.enclosing_height, 2.0);
            const double diou =
                terms.iou - terms.center_distance_squared / std::max(enclosing_diagonal, kEpsilon);
            return (diou + 1.0) / 2.0;
        }
        case AssociationMode::kCiou: {
            const double enclosing_diagonal =
                std::pow(terms.enclosing_width, 2.0) + std::pow(terms.enclosing_height, 2.0);
            const double lhs_width = lhs[2] - lhs[0];
            const double lhs_height = lhs[3] - lhs[1];
            const double rhs_width = rhs[2] - rhs[0];
            const double rhs_height = rhs[3] - rhs[1];
            const double angle_difference = std::atan(rhs_width / std::max(rhs_height, 1.0e-7)) -
                                            std::atan(lhs_width / std::max(lhs_height, 1.0e-7));
            const double v = (4.0 / (kPi * kPi)) * angle_difference * angle_difference;
            const double alpha = v / std::max(1.0 - terms.iou + v, 1.0e-7);
            const double ciou =
                terms.iou - terms.center_distance_squared / std::max(enclosing_diagonal, 1.0e-7) -
                alpha * v;
            return (ciou + 1.0) / 2.0;
        }
        case AssociationMode::kHmiou: {
            const double overlap_height =
                std::max(0.0, std::min(lhs[3], rhs[3]) - std::max(lhs[1], rhs[1]));
            const double union_height =
                std::max(1.0e-10, std::max(lhs[3], rhs[3]) - std::min(lhs[1], rhs[1]));
            return terms.iou * (overlap_height / union_height);
        }
        case AssociationMode::kCentroid:
            return CentroidSimilarity((lhs[0] + lhs[2]) / 2.0,
                                      (lhs[1] + lhs[3]) / 2.0,
                                      (rhs[0] + rhs[2]) / 2.0,
                                      (rhs[1] + rhs[3]) / 2.0,
                                      frame_width,
                                      frame_height);
    }
    throw std::invalid_argument("Unknown association mode value.");
}

double ObbAssociationSimilarity(const Eigen::Matrix<double, 5, 1>& lhs,
                                const Eigen::Matrix<double, 5, 1>& rhs,
                                const AssociationMode mode,
                                const int frame_width,
                                const int frame_height) {
    ValidateAssociationModeForDetections(mode, true);
    const ObbGeometry normalized_lhs = NormalizeObbGeometry(lhs);
    const ObbGeometry normalized_rhs = NormalizeObbGeometry(rhs);
    if (mode == AssociationMode::kCentroid) {
        return CentroidSimilarity(normalized_lhs[0],
                                  normalized_lhs[1],
                                  normalized_rhs[0],
                                  normalized_rhs[1],
                                  frame_width,
                                  frame_height);
    }
    if (ObbGeometryEquivalent(normalized_lhs, normalized_rhs)) {
        return 1.0;
    }
    const ObbOverlapTerms overlap = ComputeObbOverlapTerms(normalized_lhs, normalized_rhs);
    if (mode == AssociationMode::kIou) {
        return overlap.iou;
    }
    const Eigen::Vector4d lhs_bounds = ObbEnclosingAabb(normalized_lhs);
    const Eigen::Vector4d rhs_bounds = ObbEnclosingAabb(normalized_rhs);
    const double enclosing_width =
        std::max(lhs_bounds[2], rhs_bounds[2]) - std::min(lhs_bounds[0], rhs_bounds[0]);
    const double enclosing_height =
        std::max(lhs_bounds[3], rhs_bounds[3]) - std::min(lhs_bounds[1], rhs_bounds[1]);
    const double enclosing_diagonal =
        enclosing_width * enclosing_width + enclosing_height * enclosing_height;
    const double center_distance = std::pow(normalized_lhs[0] - normalized_rhs[0], 2.0) +
                                   std::pow(normalized_lhs[1] - normalized_rhs[1], 2.0);
    switch (mode) {
        case AssociationMode::kGiou: {
            const double enclosing_area = std::max(
                ObbConvexEnclosureArea(normalized_lhs, normalized_rhs), overlap.union_area);
            const double giou = overlap.iou - ((enclosing_area - overlap.union_area) /
                                               std::max(enclosing_area, kEpsilon));
            return (giou + 1.0) / 2.0;
        }
        case AssociationMode::kDiou: {
            const double diou =
                overlap.iou - center_distance / std::max(enclosing_diagonal, kEpsilon);
            return (diou + 1.0) / 2.0;
        }
        case AssociationMode::kCiou: {
            constexpr double epsilon = 1.0e-7;
            const double lhs_long_side = std::max(normalized_lhs[2], normalized_lhs[3]);
            const double lhs_short_side = std::min(normalized_lhs[2], normalized_lhs[3]);
            const double rhs_long_side = std::max(normalized_rhs[2], normalized_rhs[3]);
            const double rhs_short_side = std::min(normalized_rhs[2], normalized_rhs[3]);
            const double aspect_difference = std::atan(rhs_long_side / rhs_short_side) -
                                             std::atan(lhs_long_side / lhs_short_side);
            const double v = (4.0 / (kPi * kPi)) * aspect_difference * aspect_difference;
            const double alpha = v / std::max(1.0 - overlap.iou + v, epsilon);
            const double ciou =
                overlap.iou - center_distance / std::max(enclosing_diagonal, epsilon) - alpha * v;
            return (ciou + 1.0) / 2.0;
        }
        case AssociationMode::kHmiou: {
            const double overlap_height = std::max(
                0.0,
                std::min(lhs_bounds[3], rhs_bounds[3]) - std::max(lhs_bounds[1], rhs_bounds[1]));
            const double union_height = std::max(
                1.0e-10,
                std::max(lhs_bounds[3], rhs_bounds[3]) - std::min(lhs_bounds[1], rhs_bounds[1]));
            return overlap.iou * (overlap_height / union_height);
        }
        case AssociationMode::kIou:
        case AssociationMode::kCentroid:
            break;
    }
    throw std::invalid_argument("Unknown association mode value.");
}

Eigen::MatrixXd AabbAssociationMatrix(const Eigen::MatrixXd& lhs,
                                      const Eigen::MatrixXd& rhs,
                                      const AssociationMode mode,
                                      const int frame_width,
                                      const int frame_height) {
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(lhs.rows(), rhs.rows());
    if (lhs.cols() < 4 || rhs.cols() < 4) {
        return result;
    }
    for (int row = 0; row < lhs.rows(); ++row) {
        for (int col = 0; col < rhs.rows(); ++col) {
            result(row, col) = AabbAssociationSimilarity(lhs.row(row).head<4>().transpose(),
                                                         rhs.row(col).head<4>().transpose(),
                                                         mode,
                                                         frame_width,
                                                         frame_height);
        }
    }
    return result;
}

Eigen::MatrixXd ObbAssociationMatrix(const Eigen::MatrixXd& lhs,
                                     const Eigen::MatrixXd& rhs,
                                     const AssociationMode mode,
                                     const int frame_width,
                                     const int frame_height) {
    ValidateAssociationModeForDetections(mode, true);
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(lhs.rows(), rhs.rows());
    if (lhs.cols() < 5 || rhs.cols() < 5) {
        return result;
    }
    for (int row = 0; row < lhs.rows(); ++row) {
        for (int col = 0; col < rhs.rows(); ++col) {
            result(row, col) = ObbAssociationSimilarity(lhs.row(row).head<5>().transpose(),
                                                        rhs.row(col).head<5>().transpose(),
                                                        mode,
                                                        frame_width,
                                                        frame_height);
        }
    }
    return result;
}

Eigen::MatrixXd AssociationMatrix(const Eigen::MatrixXd& lhs,
                                  const Eigen::MatrixXd& rhs,
                                  const bool is_obb,
                                  const AssociationMode mode,
                                  const int frame_width,
                                  const int frame_height) {
    return is_obb ? ObbAssociationMatrix(lhs, rhs, mode, frame_width, frame_height)
                  : AabbAssociationMatrix(lhs, rhs, mode, frame_width, frame_height);
}

}  // namespace boxmot::trackers::base
