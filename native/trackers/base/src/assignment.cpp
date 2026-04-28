#include "boxmot/trackers/base/assignment.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace boxmot::trackers::base {

namespace {

std::vector<int> SolveHungarian(const Eigen::MatrixXd& cost_matrix) {
    const int n = static_cast<int>(cost_matrix.rows());
    const int m = static_cast<int>(cost_matrix.cols());
    const double inf = std::numeric_limits<double>::infinity();

    std::vector<double> u(n + 1, 0.0);
    std::vector<double> v(m + 1, 0.0);
    std::vector<int> p(m + 1, 0);
    std::vector<int> way(m + 1, 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(m + 1, inf);
        std::vector<bool> used(m + 1, false);
        do {
            used[j0] = true;
            const int i0 = p[j0];
            double delta = inf;
            int j1 = 0;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) {
                    continue;
                }
                const double cur = static_cast<double>(cost_matrix(i0 - 1, j - 1)) - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> assignment(n, -1);
    for (int j = 1; j <= m; ++j) {
        if (p[j] > 0 && p[j] <= n) {
            assignment[p[j] - 1] = j - 1;
        }
    }
    return assignment;
}

}  // namespace

AssignmentResult LinearAssignment(const Eigen::MatrixXd& cost_matrix, const double threshold) {
    AssignmentResult result;
    const int rows = static_cast<int>(cost_matrix.rows());
    const int cols = static_cast<int>(cost_matrix.cols());

    if (rows == 0 || cols == 0) {
        result.unmatched_rows.resize(rows);
        result.unmatched_cols.resize(cols);
        for (int i = 0; i < rows; ++i) {
            result.unmatched_rows[i] = i;
        }
        for (int j = 0; j < cols; ++j) {
            result.unmatched_cols[j] = j;
        }
        return result;
    }

    const int size = std::max(rows, cols);
    const double max_cost = cost_matrix.maxCoeff();
    const double pad_cost = std::max(threshold + 1.0, max_cost + 1.0);

    Eigen::MatrixXd square = Eigen::MatrixXd::Constant(size, size, pad_cost);
    square.block(0, 0, rows, cols) = cost_matrix;

    const std::vector<int> assignment = SolveHungarian(square);

    std::vector<bool> matched_rows(rows, false);
    std::vector<bool> matched_cols(cols, false);
    for (int row = 0; row < rows; ++row) {
        const int col = assignment[row];
        if (col >= 0 && col < cols && cost_matrix(row, col) <= threshold) {
            matched_rows[row] = true;
            matched_cols[col] = true;
            result.matches.emplace_back(row, col);
        }
    }

    for (int row = 0; row < rows; ++row) {
        if (!matched_rows[row]) {
            result.unmatched_rows.push_back(row);
        }
    }
    for (int col = 0; col < cols; ++col) {
        if (!matched_cols[col]) {
            result.unmatched_cols.push_back(col);
        }
    }

    return result;
}

}  // namespace boxmot::trackers::base