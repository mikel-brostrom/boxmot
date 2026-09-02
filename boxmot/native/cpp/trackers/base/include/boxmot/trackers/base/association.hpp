#pragma once

#include <Eigen/Dense>

#include <string>
#include <string_view>
#include <vector>

namespace boxmot::trackers::base {

enum class AssociationMode {
    kIou,
    kGiou,
    kDiou,
    kCiou,
    kHmiou,
    kCentroid,
};

[[nodiscard]] AssociationMode ParseAssociationMode(std::string_view name);
[[nodiscard]] std::string AssociationModeName(AssociationMode mode);
[[nodiscard]] bool AssociationModeRequiresFrameDimensions(AssociationMode mode) noexcept;
[[nodiscard]] bool AssociationModeSupportsObb(AssociationMode mode) noexcept;
void ValidateAssociationModeForDetections(AssociationMode mode, bool is_obb);

[[nodiscard]] double AabbAssociationSimilarity(const Eigen::Vector4d& lhs,
                                               const Eigen::Vector4d& rhs,
                                               AssociationMode mode,
                                               int frame_width = 0,
                                               int frame_height = 0);

[[nodiscard]] double ObbAssociationSimilarity(const Eigen::Matrix<double, 5, 1>& lhs,
                                              const Eigen::Matrix<double, 5, 1>& rhs,
                                              AssociationMode mode,
                                              int frame_width = 0,
                                              int frame_height = 0);

[[nodiscard]] Eigen::MatrixXd AabbAssociationMatrix(const Eigen::MatrixXd& lhs,
                                                    const Eigen::MatrixXd& rhs,
                                                    AssociationMode mode,
                                                    int frame_width = 0,
                                                    int frame_height = 0);

[[nodiscard]] Eigen::MatrixXd ObbAssociationMatrix(const Eigen::MatrixXd& lhs,
                                                   const Eigen::MatrixXd& rhs,
                                                   AssociationMode mode,
                                                   int frame_width = 0,
                                                   int frame_height = 0);

[[nodiscard]] Eigen::MatrixXd AssociationMatrix(const Eigen::MatrixXd& lhs,
                                                const Eigen::MatrixXd& rhs,
                                                bool is_obb,
                                                AssociationMode mode,
                                                int frame_width = 0,
                                                int frame_height = 0);

template <typename TrackPtr>
[[nodiscard]] Eigen::MatrixXd TrackAssociationDistance(const std::vector<TrackPtr>& lhs,
                                                       const std::vector<TrackPtr>& rhs,
                                                       const AssociationMode mode,
                                                       const int frame_width = 0,
                                                       const int frame_height = 0) {
    Eigen::MatrixXd distance(static_cast<int>(lhs.size()), static_cast<int>(rhs.size()));
    for (int row = 0; row < distance.rows(); ++row) {
        for (int col = 0; col < distance.cols(); ++col) {
            const bool is_obb = lhs[static_cast<std::size_t>(row)]->UsesObb() ||
                                rhs[static_cast<std::size_t>(col)]->UsesObb();
            const double similarity =
                is_obb ? ObbAssociationSimilarity(lhs[static_cast<std::size_t>(row)]->xywha(),
                                                  rhs[static_cast<std::size_t>(col)]->xywha(),
                                                  mode,
                                                  frame_width,
                                                  frame_height)
                       : AabbAssociationSimilarity(lhs[static_cast<std::size_t>(row)]->xyxy(),
                                                   rhs[static_cast<std::size_t>(col)]->xyxy(),
                                                   mode,
                                                   frame_width,
                                                   frame_height);
            distance(row, col) = 1.0 - similarity;
        }
    }
    return distance;
}

}  // namespace boxmot::trackers::base
