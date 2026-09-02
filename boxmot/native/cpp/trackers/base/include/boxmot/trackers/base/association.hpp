#pragma once

#include <Eigen/Dense>

#include <string>
#include <string_view>

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

[[nodiscard]] double AabbAssociationSimilarity(
    const Eigen::Vector4d& lhs,
    const Eigen::Vector4d& rhs,
    AssociationMode mode,
    int frame_width = 0,
    int frame_height = 0
);

[[nodiscard]] double ObbAssociationSimilarity(
    const Eigen::Matrix<double, 5, 1>& lhs,
    const Eigen::Matrix<double, 5, 1>& rhs,
    AssociationMode mode,
    int frame_width = 0,
    int frame_height = 0
);

[[nodiscard]] Eigen::MatrixXd AabbAssociationMatrix(
    const Eigen::MatrixXd& lhs,
    const Eigen::MatrixXd& rhs,
    AssociationMode mode,
    int frame_width = 0,
    int frame_height = 0
);

[[nodiscard]] Eigen::MatrixXd ObbAssociationMatrix(
    const Eigen::MatrixXd& lhs,
    const Eigen::MatrixXd& rhs,
    AssociationMode mode,
    int frame_width = 0,
    int frame_height = 0
);

}  // namespace boxmot::trackers::base
