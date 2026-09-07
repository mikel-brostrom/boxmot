#include "boxmot/trackers/base/association.hpp"

#include <Eigen/Dense>

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(const int argc, char** argv) {
    try {
        if (argc != 2 && argc != 3 && argc != 11 && argc != 13) {
            throw std::invalid_argument(
                "Usage: boxmot_association_probe <mode> "
                "[aabb [lhs_x1 lhs_y1 lhs_x2 lhs_y2 rhs_x1 rhs_y1 rhs_x2 rhs_y2] | "
                "obb [lhs_cx lhs_cy lhs_w lhs_h lhs_a rhs_cx rhs_cy rhs_w rhs_h rhs_a]]");
        }
        const auto mode = boxmot::trackers::base::ParseAssociationMode(argv[1]);
        double similarity = 0.0;
        const std::string geometry = argc >= 3 ? argv[2] : "aabb";
        if (geometry == "obb-empty-lhs" || geometry == "obb-empty-rhs") {
            Eigen::MatrixXd lhs(geometry == "obb-empty-lhs" ? 0 : 1, 5);
            Eigen::MatrixXd rhs(geometry == "obb-empty-rhs" ? 0 : 1, 5);
            if (lhs.rows() != 0) {
                lhs << 10.0, 10.0, 8.0, 4.0, 0.2;
            }
            if (rhs.rows() != 0) {
                rhs << 12.0, 11.0, 8.0, 4.0, 0.25;
            }
            const Eigen::MatrixXd matrix =
                boxmot::trackers::base::ObbAssociationMatrix(lhs, rhs, mode, 100, 80);
            std::cout << matrix.rows() << ' ' << matrix.cols() << '\n';
            return 0;
        }
        if (geometry == "obb") {
            Eigen::Matrix<double, 5, 1> lhs;
            Eigen::Matrix<double, 5, 1> rhs;
            if (argc == 13) {
                for (int idx = 0; idx < 5; ++idx) {
                    lhs[idx] = std::stod(argv[3 + idx]);
                    rhs[idx] = std::stod(argv[8 + idx]);
                }
            } else {
                lhs << 10.0, 10.0, 8.0, 4.0, 0.2;
                rhs << 12.0, 11.0, 8.0, 4.0, 0.25;
            }
            similarity = boxmot::trackers::base::ObbAssociationSimilarity(lhs, rhs, mode, 100, 80);
        } else if (geometry == "aabb") {
            Eigen::Vector4d lhs(0.0, 0.0, 10.0, 10.0);
            Eigen::Vector4d rhs(5.0, 2.0, 15.0, 12.0);
            if (argc == 11) {
                for (int idx = 0; idx < 4; ++idx) {
                    lhs[idx] = std::stod(argv[3 + idx]);
                    rhs[idx] = std::stod(argv[7 + idx]);
                }
            }
            similarity = boxmot::trackers::base::AabbAssociationSimilarity(lhs, rhs, mode, 100, 80);
        } else {
            throw std::invalid_argument("Unknown probe geometry '" + geometry + "'.");
        }
        std::cout << std::setprecision(17) << similarity << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return 1;
    }
}
