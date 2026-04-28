#pragma once

#include <Eigen/Dense>

#include <utility>
#include <vector>

namespace boxmot::trackers::base {

struct AssignmentResult {
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_rows;
    std::vector<int> unmatched_cols;
};

AssignmentResult LinearAssignment(const Eigen::MatrixXd& cost_matrix, double threshold);

}  // namespace boxmot::trackers::base