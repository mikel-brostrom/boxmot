#include "boxmot/trackers/base/assignment.hpp"

#include <Eigen/Core>

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

namespace {

bool HasMatch(
    const boxmot::trackers::base::AssignmentResult& result,
    const int expected_row,
    const int expected_col
) {
    for (const auto& [row, col] : result.matches) {
        if (row == expected_row && col == expected_col) {
            return true;
        }
    }
    return false;
}

int CheckNonfinite() {
    for (const double invalid : {
             std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity(),
         }) {
        Eigen::MatrixXd costs(1, 1);
        costs(0, 0) = invalid;
        const auto result = boxmot::trackers::base::LinearAssignment(
            costs,
            std::numeric_limits<double>::infinity()
        );
        if (!result.matches.empty() || result.unmatched_rows != std::vector<int>{0}
            || result.unmatched_cols != std::vector<int>{0}) {
            return 1;
        }
    }
    return 0;
}

int CheckLargeThresholdNearTie() {
    Eigen::MatrixXd costs(2, 2);
    costs << -0.5 + 1.0e-8, -0.5,
             -0.5, -0.5 + 1.0e-8;
    const auto result = boxmot::trackers::base::LinearAssignment(costs, 1.0e9);
    return result.matches.size() == 2 && HasMatch(result, 0, 1) && HasMatch(result, 1, 0) ? 0 : 1;
}

}  // namespace

int main(const int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "expected one probe mode\n";
        return 2;
    }
    const std::string mode(argv[1]);
    if (mode == "nonfinite") {
        return CheckNonfinite();
    }
    if (mode == "near-tie") {
        return CheckLargeThresholdNearTie();
    }
    std::cerr << "unknown probe mode: " << mode << '\n';
    return 2;
}
