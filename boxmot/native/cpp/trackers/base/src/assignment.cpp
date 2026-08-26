#include "boxmot/trackers/base/assignment.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace boxmot::trackers::base {

namespace {

bool FullCardinalityMustWin(const Eigen::MatrixXd& costs, const double threshold) {
    if (!std::isfinite(threshold) || !costs.allFinite()) {
        return false;
    }

    const double min_cost = costs.minCoeff();
    const double max_cost = costs.maxCoeff();
    if (max_cost > threshold) {
        return false;
    }

    // A cost-limit solve minimizes sum(cost - threshold) over the selected
    // edges. If even the most expensive additional edge is more beneficial
    // than every possible reshuffle of the existing matching, the optimum is
    // guaranteed to have full cardinality. Solving the original rectangular
    // problem then preserves small cost differences that would otherwise be
    // rounded away by subtracting a very large sentinel threshold (OCSort
    // deliberately uses 1e9 for its effectively-unbounded assignments).
    const int cardinality = std::min<int>(costs.rows(), costs.cols());
    const long double add_edge_margin =
        static_cast<long double>(threshold) - static_cast<long double>(max_cost);
    const long double max_reshuffle =
        static_cast<long double>(std::max(0, cardinality - 1))
        * (static_cast<long double>(max_cost) - static_cast<long double>(min_cost));
    return add_edge_margin > max_reshuffle;
}

Eigen::MatrixXd MakeUnboundedSquare(const Eigen::MatrixXd& costs) {
    const int rows = static_cast<int>(costs.rows());
    const int cols = static_cast<int>(costs.cols());
    const int size = std::max(rows, cols);
    Eigen::MatrixXd square = Eigen::MatrixXd::Zero(size, size);

    if (costs.allFinite()) {
        // Every square assignment contains the same number of padding edges,
        // so zero padding leaves the rectangular optimum unchanged.
        square.block(0, 0, rows, cols) = costs;
        return square;
    }

    // The Hungarian implementation requires finite arithmetic. Normalize the
    // valid edges to [0, 1] and give invalid edges a penalty larger than the
    // total cost of any all-finite assignment. This first maximizes the number
    // of finite matches, then minimizes their original cost.
    double min_cost = std::numeric_limits<double>::infinity();
    double max_cost = -std::numeric_limits<double>::infinity();
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const double cost = costs(row, col);
            if (std::isfinite(cost)) {
                min_cost = std::min(min_cost, cost);
                max_cost = std::max(max_cost, cost);
            }
        }
    }

    const bool has_finite_cost = std::isfinite(min_cost);
    const double range = has_finite_cost ? max_cost - min_cost : 0.0;
    const double blocked_cost = static_cast<double>(size + 1);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const double cost = costs(row, col);
            if (!std::isfinite(cost)) {
                square(row, col) = blocked_cost;
            } else if (range > 0.0) {
                square(row, col) = (cost - min_cost) / range;
            }
        }
    }
    return square;
}

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

    Eigen::MatrixXd square;
    if (FullCardinalityMustWin(cost_matrix, threshold)) {
        square = MakeUnboundedSquare(cost_matrix);
    } else if (std::isfinite(threshold)) {
        // Match lap.lapjv(..., extend_cost=True, cost_limit=threshold): rows
        // and columns must be allowed to remain unmatched *during* the
        // optimization. Solving a full assignment first and rejecting costly
        // pairs afterwards can discard a different, valid match that the
        // threshold-aware optimum would keep.
        //
        // Subtracting the limit from every admissible edge makes a match
        // beneficial exactly when its original cost is <= threshold. The
        // augmented dummy rows/columns have zero cost, so the Hungarian solve
        // jointly chooses the best set of optional one-to-one matches.
        const int size = rows + cols;
        constexpr double blocked_cost = 1.0;
        square = Eigen::MatrixXd::Constant(size, size, blocked_cost);

        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                const double cost = cost_matrix(row, col);
                if (std::isfinite(cost) && cost <= threshold) {
                    square(row, col) = cost - threshold;
                }
            }
            square(row, cols + row) = 0.0;
        }
        for (int col = 0; col < cols; ++col) {
            square(rows + col, col) = 0.0;
        }
        square.block(rows, cols, cols, rows).setZero();
    } else {
        // Callers use +inf when tracker-specific validity checks happen after
        // assignment. Keep every value entering the Hungarian loop finite so
        // NaN/+inf inputs cannot turn `inf - inf` into an endless solver spin.
        square = MakeUnboundedSquare(cost_matrix);
    }

    const std::vector<int> assignment = SolveHungarian(square);

    std::vector<bool> matched_rows(rows, false);
    std::vector<bool> matched_cols(cols, false);
    for (int row = 0; row < rows; ++row) {
        const int col = assignment[row];
        if (col >= 0 && col < cols && std::isfinite(cost_matrix(row, col)) &&
            cost_matrix(row, col) <= threshold) {
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
