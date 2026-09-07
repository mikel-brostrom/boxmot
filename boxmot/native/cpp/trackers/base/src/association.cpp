#include "boxmot/trackers/base/association.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace boxmot::trackers::base {

namespace {

constexpr double kEpsilon = 1.0e-12;
constexpr double kEquivalentObbLinearTolerance = 1.0e-9;
constexpr double kEquivalentObbRelativeTolerance =
    8.0 * static_cast<double>(std::numeric_limits<float>::epsilon());
constexpr double kPi = 3.14159265358979323846;

using ObbGeometry = Eigen::Matrix<double, 5, 1>;

struct Point2d {
    double x = 0.0;
    double y = 0.0;
};

long double Cross(const Point2d& lhs, const Point2d& rhs) {
    return static_cast<long double>(lhs.x) * static_cast<long double>(rhs.y) -
           static_cast<long double>(lhs.y) * static_cast<long double>(rhs.x);
}

Point2d Difference(const Point2d& lhs, const Point2d& rhs) {
    return Point2d{lhs.x - rhs.x, lhs.y - rhs.y};
}

void ValidateObbGeometry(const ObbGeometry& box) {
    if (!box.allFinite()) {
        throw std::invalid_argument("OBB geometry values must be finite.");
    }
    if (box[2] <= 0.0 || box[3] <= 0.0) {
        throw std::invalid_argument("OBB width and height must be strictly positive.");
    }
}

ObbGeometry CanonicalizeObbGeometry(const ObbGeometry& box) {
    ObbGeometry canonical = box;
    canonical[4] = std::remainder(canonical[4], kPi);
    return canonical;
}

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

bool ObbGeometryEquivalent(const ObbGeometry& lhs, const ObbGeometry& rhs) {
    if (lhs[0] != rhs[0] || lhs[1] != rhs[1]) {
        return false;
    }

    const double angle_difference = lhs[4] - rhs[4];
    const bool direct =
        lhs[2] == rhs[2] && lhs[3] == rhs[3] && std::remainder(angle_difference, kPi) == 0.0;
    const double swapped_angle_difference =
        std::abs(std::remainder(angle_difference - kPi / 2.0, kPi));
    const double corner_radius = std::hypot(lhs[2] / 2.0, lhs[3] / 2.0);
    const double corner_displacement =
        corner_radius * (2.0 * std::sin(swapped_angle_difference / 2.0));
    const bool swapped = lhs[2] == rhs[3] && lhs[3] == rhs[2] &&
                         corner_displacement <= kEquivalentObbLinearTolerance &&
                         corner_displacement <= corner_radius * kEquivalentObbRelativeTolerance;
    return direct || swapped;
}

struct LocalObbPair {
    ObbGeometry lhs;
    ObbGeometry rhs;
};

LocalObbPair NormalizeObbPair(const ObbGeometry& lhs, const ObbGeometry& rhs) {
    const long double half_delta_x =
        static_cast<long double>(rhs[0]) / 2.0L - static_cast<long double>(lhs[0]) / 2.0L;
    const long double half_delta_y =
        static_cast<long double>(rhs[1]) / 2.0L - static_cast<long double>(lhs[1]) / 2.0L;
    const long double scale = std::max({std::abs(half_delta_x),
                                        std::abs(half_delta_y),
                                        static_cast<long double>(lhs[2]),
                                        static_cast<long double>(lhs[3]),
                                        static_cast<long double>(rhs[2]),
                                        static_cast<long double>(rhs[3])});
    if (!std::isfinite(scale) || scale <= 0.0L) {
        throw std::invalid_argument("OBB pair geometry exceeds the finite numeric range.");
    }
    const long double normalized_half_delta_x = half_delta_x / scale;
    const long double normalized_half_delta_y = half_delta_y / scale;
    const long double half_distance = std::hypot(normalized_half_delta_x, normalized_half_delta_y);
    long double frame_angle = 0.0L;
    if (half_distance > 0.0L) {
        const long double unit_x = normalized_half_delta_x / half_distance;
        const long double unit_y = normalized_half_delta_y / half_distance;
        frame_angle = 0.5L * std::atan2(2.0L * unit_x * unit_y, unit_x * unit_x - unit_y * unit_y);
    } else {
        frame_angle =
            static_cast<long double>(lhs[4]) / 2.0L + static_cast<long double>(rhs[4]) / 2.0L;
    }
    const long double cosine = std::cos(frame_angle);
    const long double sine = std::sin(frame_angle);
    const long double local_half_delta_x =
        cosine * normalized_half_delta_x + sine * normalized_half_delta_y;
    const long double local_half_delta_y =
        -sine * normalized_half_delta_x + cosine * normalized_half_delta_y;

    LocalObbPair local{lhs, rhs};
    local.lhs[0] = static_cast<double>(-local_half_delta_x);
    local.lhs[1] = static_cast<double>(-local_half_delta_y);
    local.rhs[0] = static_cast<double>(local_half_delta_x);
    local.rhs[1] = static_cast<double>(local_half_delta_y);
    for (const int side_index : {2, 3}) {
        local.lhs[side_index] =
            static_cast<double>(static_cast<long double>(lhs[side_index]) / scale);
        local.rhs[side_index] =
            static_cast<double>(static_cast<long double>(rhs[side_index]) / scale);
    }
    const double frame_angle_double = static_cast<double>(frame_angle);
    local.lhs[4] = std::remainder(lhs[4] - frame_angle_double, kPi);
    local.rhs[4] = std::remainder(rhs[4] - frame_angle_double, kPi);
    if (!local.lhs.allFinite() || !local.rhs.allFinite() || local.lhs[2] <= 0.0 ||
        local.lhs[3] <= 0.0 || local.rhs[2] <= 0.0 || local.rhs[3] <= 0.0) {
        throw std::invalid_argument("OBB pair geometry exceeds the finite numeric range.");
    }
    return local;
}

std::array<Point2d, 4> ObbCorners(const ObbGeometry& box) {
    const double cosine = std::cos(box[4]);
    const double sine = std::sin(box[4]);
    const double half_width = box[2] / 2.0;
    const double half_height = box[3] / 2.0;
    const Point2d width_axis{half_width * cosine, half_width * sine};
    const Point2d height_axis{-half_height * sine, half_height * cosine};
    return {
        Point2d{box[0] - width_axis.x - height_axis.x, box[1] - width_axis.y - height_axis.y},
        Point2d{box[0] + width_axis.x - height_axis.x, box[1] + width_axis.y - height_axis.y},
        Point2d{box[0] + width_axis.x + height_axis.x, box[1] + width_axis.y + height_axis.y},
        Point2d{box[0] - width_axis.x + height_axis.x, box[1] - width_axis.y + height_axis.y},
    };
}

long double HalfPlaneValue(const Point2d& edge_start,
                           const Point2d& edge_end,
                           const Point2d& point) {
    return Cross(Difference(edge_end, edge_start), Difference(point, edge_start));
}

long double HalfPlaneTolerance(const Point2d& edge_start,
                               const Point2d& edge_end,
                               const Point2d& point) {
    const Point2d edge = Difference(edge_end, edge_start);
    const Point2d relative = Difference(point, edge_start);
    constexpr long double factor =
        64.0L * static_cast<long double>(std::numeric_limits<double>::epsilon());
    return factor * (std::abs(static_cast<long double>(edge.x) * relative.y) +
                     std::abs(static_cast<long double>(edge.y) * relative.x));
}

bool InsideHalfPlane(const Point2d& edge_start, const Point2d& edge_end, const Point2d& point) {
    return HalfPlaneValue(edge_start, edge_end, point) >=
           -HalfPlaneTolerance(edge_start, edge_end, point);
}

Point2d SegmentLineIntersection(const Point2d& segment_start,
                                const Point2d& segment_end,
                                const Point2d& line_start,
                                const Point2d& line_end) {
    const long double start_value = HalfPlaneValue(line_start, line_end, segment_start);
    const long double end_value = HalfPlaneValue(line_start, line_end, segment_end);
    const long double denominator = start_value - end_value;
    if (denominator == 0.0L) {
        return segment_end;
    }
    const double interpolation =
        std::clamp(static_cast<double>(start_value / denominator), 0.0, 1.0);
    return Point2d{
        segment_start.x + interpolation * (segment_end.x - segment_start.x),
        segment_start.y + interpolation * (segment_end.y - segment_start.y),
    };
}

std::vector<Point2d> ConvexPolygonIntersection(const std::array<Point2d, 4>& lhs,
                                               const std::array<Point2d, 4>& rhs) {
    std::vector<Point2d> polygon(lhs.begin(), lhs.end());
    for (std::size_t edge_index = 0; edge_index < rhs.size() && !polygon.empty(); ++edge_index) {
        const Point2d& edge_start = rhs[edge_index];
        const Point2d& edge_end = rhs[(edge_index + 1) % rhs.size()];
        std::vector<Point2d> clipped;
        clipped.reserve(polygon.size() + 1);
        Point2d previous = polygon.back();
        bool previous_inside = InsideHalfPlane(edge_start, edge_end, previous);
        for (const Point2d& current : polygon) {
            const bool current_inside = InsideHalfPlane(edge_start, edge_end, current);
            if (current_inside != previous_inside) {
                clipped.push_back(SegmentLineIntersection(previous, current, edge_start, edge_end));
            }
            if (current_inside) {
                clipped.push_back(current);
            }
            previous = current;
            previous_inside = current_inside;
        }
        polygon = std::move(clipped);
    }
    return polygon;
}

double PolygonArea(const std::vector<Point2d>& polygon) {
    if (polygon.size() < 3) {
        return 0.0;
    }
    long double twice_area = 0.0L;
    for (std::size_t index = 0; index < polygon.size(); ++index) {
        twice_area += Cross(polygon[index], polygon[(index + 1) % polygon.size()]);
    }
    return static_cast<double>(std::abs(twice_area) / 2.0L);
}

std::vector<Point2d> ConvexHull(std::vector<Point2d> points) {
    std::sort(points.begin(), points.end(), [](const Point2d& lhs, const Point2d& rhs) {
        return lhs.x < rhs.x || (lhs.x == rhs.x && lhs.y < rhs.y);
    });
    points.erase(std::unique(points.begin(),
                             points.end(),
                             [](const Point2d& lhs, const Point2d& rhs) {
                                 return lhs.x == rhs.x && lhs.y == rhs.y;
                             }),
                 points.end());
    if (points.size() <= 1) {
        return points;
    }

    std::vector<Point2d> hull(2 * points.size());
    std::size_t count = 0;
    for (const Point2d& point : points) {
        while (count >= 2 && Cross(Difference(hull[count - 1], hull[count - 2]),
                                   Difference(point, hull[count - 1])) <= 0.0L) {
            --count;
        }
        hull[count++] = point;
    }
    const std::size_t lower_count = count;
    for (std::size_t index = points.size() - 1; index > 0; --index) {
        const Point2d& point = points[index - 1];
        while (count > lower_count && Cross(Difference(hull[count - 1], hull[count - 2]),
                                            Difference(point, hull[count - 1])) <= 0.0L) {
            --count;
        }
        hull[count++] = point;
    }
    hull.resize(count - 1);
    return hull;
}

std::vector<Point2d> PairConvexHull(const LocalObbPair& pair) {
    const std::array<Point2d, 4> lhs_corners = ObbCorners(pair.lhs);
    const std::array<Point2d, 4> rhs_corners = ObbCorners(pair.rhs);
    std::vector<Point2d> corners;
    corners.reserve(lhs_corners.size() + rhs_corners.size());
    corners.insert(corners.end(), lhs_corners.begin(), lhs_corners.end());
    corners.insert(corners.end(), rhs_corners.begin(), rhs_corners.end());
    return ConvexHull(std::move(corners));
}

struct ObbOverlapTerms {
    double union_area = 0.0;
    double iou = 0.0;
};

ObbOverlapTerms ComputeObbOverlapTerms(const LocalObbPair& pair) {
    const double intersection_area =
        PolygonArea(ConvexPolygonIntersection(ObbCorners(pair.lhs), ObbCorners(pair.rhs)));
    const double lhs_area = pair.lhs[2] * pair.lhs[3];
    const double rhs_area = pair.rhs[2] * pair.rhs[3];
    const double area_sum = lhs_area + rhs_area;
    const double bounded_intersection =
        std::clamp(intersection_area, 0.0, std::min(lhs_area, rhs_area));
    const double raw_union_area = area_sum - bounded_intersection;
    ObbOverlapTerms terms;
    terms.iou =
        raw_union_area > 0.0 ? std::clamp(bounded_intersection / raw_union_area, 0.0, 1.0) : 0.0;
    terms.union_area = raw_union_area;
    return terms;
}

double ObbConvexEnclosureArea(const LocalObbPair& pair) {
    return PolygonArea(PairConvexHull(pair));
}

double ObbMinimumEnclosingRectangleDiagonalSquared(const LocalObbPair& pair) {
    const std::vector<Point2d> hull = PairConvexHull(pair);
    double best_area = std::numeric_limits<double>::infinity();
    double best_diagonal = std::numeric_limits<double>::infinity();
    for (std::size_t edge_index = 0; edge_index < hull.size(); ++edge_index) {
        const Point2d edge = Difference(hull[(edge_index + 1) % hull.size()], hull[edge_index]);
        const double edge_length = std::hypot(edge.x, edge.y);
        if (edge_length == 0.0) {
            continue;
        }
        const Point2d axis{edge.x / edge_length, edge.y / edge_length};
        const Point2d normal{-axis.y, axis.x};
        double min_axis = std::numeric_limits<double>::infinity();
        double max_axis = -std::numeric_limits<double>::infinity();
        double min_normal = std::numeric_limits<double>::infinity();
        double max_normal = -std::numeric_limits<double>::infinity();
        for (const Point2d& point : hull) {
            const double axis_projection = point.x * axis.x + point.y * axis.y;
            const double normal_projection = point.x * normal.x + point.y * normal.y;
            min_axis = std::min(min_axis, axis_projection);
            max_axis = std::max(max_axis, axis_projection);
            min_normal = std::min(min_normal, normal_projection);
            max_normal = std::max(max_normal, normal_projection);
        }
        const double width = max_axis - min_axis;
        const double height = max_normal - min_normal;
        const double area = width * height;
        const double diagonal = width * width + height * height;
        const double area_tolerance = std::isfinite(best_area)
                                          ? 1.0e-12 * std::max(std::abs(area), std::abs(best_area))
                                          : 0.0;
        if (!std::isfinite(best_area) || area < best_area - area_tolerance ||
            (std::abs(area - best_area) <= area_tolerance && diagonal < best_diagonal)) {
            best_area = area;
            best_diagonal = diagonal;
        }
    }
    if (!std::isfinite(best_diagonal) || best_diagonal <= 0.0) {
        throw std::runtime_error("Unable to construct an enclosing rectangle for OBB pair.");
    }
    return best_diagonal;
}

long double NormalizedObbEnclosingHalfHeight(const ObbGeometry& box, const long double scale) {
    const long double normalized_width = static_cast<long double>(box[2]) / scale;
    const long double normalized_height = static_cast<long double>(box[3]) / scale;
    const long double cosine = std::abs(std::cos(box[4]));
    const long double sine = std::abs(std::sin(box[4]));
    return 0.5L * (normalized_width * sine + normalized_height * cosine);
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
            return std::clamp((ciou + 1.0) / 2.0, 0.0, 1.0);
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
    ValidateObbGeometry(lhs);
    ValidateObbGeometry(rhs);
    if (mode == AssociationMode::kCentroid) {
        return CentroidSimilarity(lhs[0], lhs[1], rhs[0], rhs[1], frame_width, frame_height);
    }
    const ObbGeometry canonical_lhs = CanonicalizeObbGeometry(lhs);
    const ObbGeometry canonical_rhs = CanonicalizeObbGeometry(rhs);
    if (ObbGeometryEquivalent(canonical_lhs, canonical_rhs)) {
        return 1.0;
    }
    const LocalObbPair normalized_pair = NormalizeObbPair(canonical_lhs, canonical_rhs);
    const ObbOverlapTerms overlap = ComputeObbOverlapTerms(normalized_pair);
    if (mode == AssociationMode::kIou) {
        return overlap.iou;
    }
    const double center_distance = std::pow(normalized_pair.lhs[0] - normalized_pair.rhs[0], 2.0) +
                                   std::pow(normalized_pair.lhs[1] - normalized_pair.rhs[1], 2.0);
    switch (mode) {
        case AssociationMode::kGiou: {
            const double enclosing_area =
                std::max(ObbConvexEnclosureArea(normalized_pair), overlap.union_area);
            const double giou =
                overlap.iou - ((enclosing_area - overlap.union_area) / enclosing_area);
            return std::clamp((giou + 1.0) / 2.0, 0.0, 1.0);
        }
        case AssociationMode::kDiou: {
            const double enclosing_diagonal =
                ObbMinimumEnclosingRectangleDiagonalSquared(normalized_pair);
            const double diou = overlap.iou - center_distance / enclosing_diagonal;
            return std::clamp((diou + 1.0) / 2.0, 0.0, 1.0);
        }
        case AssociationMode::kCiou: {
            const double enclosing_diagonal =
                ObbMinimumEnclosingRectangleDiagonalSquared(normalized_pair);
            const double lhs_long_side = std::max(canonical_lhs[2], canonical_lhs[3]);
            const double lhs_short_side = std::min(canonical_lhs[2], canonical_lhs[3]);
            const double rhs_long_side = std::max(canonical_rhs[2], canonical_rhs[3]);
            const double rhs_short_side = std::min(canonical_rhs[2], canonical_rhs[3]);
            const double aspect_difference = std::atan(rhs_long_side / rhs_short_side) -
                                             std::atan(lhs_long_side / lhs_short_side);
            const double v = (4.0 / (kPi * kPi)) * aspect_difference * aspect_difference;
            const double alpha = v == 0.0 ? 0.0 : v / (1.0 - overlap.iou + v);
            const double ciou = overlap.iou - center_distance / enclosing_diagonal - alpha * v;
            return std::clamp((ciou + 1.0) / 2.0, 0.0, 1.0);
        }
        case AssociationMode::kHmiou: {
            const long double half_delta_y = static_cast<long double>(canonical_rhs[1]) / 2.0L -
                                             static_cast<long double>(canonical_lhs[1]) / 2.0L;
            const long double scale = std::max({std::abs(half_delta_y),
                                                static_cast<long double>(canonical_lhs[2]),
                                                static_cast<long double>(canonical_lhs[3]),
                                                static_cast<long double>(canonical_rhs[2]),
                                                static_cast<long double>(canonical_rhs[3])});
            const long double lhs_half_height =
                NormalizedObbEnclosingHalfHeight(canonical_lhs, scale);
            const long double rhs_half_height =
                NormalizedObbEnclosingHalfHeight(canonical_rhs, scale);
            const long double center_distance = 2.0L * std::abs(half_delta_y / scale);
            const long double overlap_height =
                std::max(0.0L,
                         std::min({2.0L * lhs_half_height,
                                   2.0L * rhs_half_height,
                                   lhs_half_height + rhs_half_height - center_distance}));
            const long double enclosing_height =
                std::max({2.0L * lhs_half_height,
                          2.0L * rhs_half_height,
                          lhs_half_height + rhs_half_height + center_distance});
            return overlap.iou * static_cast<double>(overlap_height / enclosing_height);
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
