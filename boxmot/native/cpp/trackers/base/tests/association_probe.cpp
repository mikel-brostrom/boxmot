#include "boxmot/trackers/base/association.hpp"

#include <Eigen/Dense>

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(const int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) {
            throw std::invalid_argument("Usage: boxmot_association_probe <mode> [obb]");
        }
        const auto mode = boxmot::trackers::base::ParseAssociationMode(argv[1]);
        double similarity = 0.0;
        if (argc == 3 && std::string(argv[2]) == "obb") {
            Eigen::Matrix<double, 5, 1> lhs;
            Eigen::Matrix<double, 5, 1> rhs;
            lhs << 10.0, 10.0, 8.0, 4.0, 0.2;
            rhs << 12.0, 11.0, 8.0, 4.0, 0.25;
            similarity = boxmot::trackers::base::ObbAssociationSimilarity(lhs, rhs, mode, 100, 80);
        } else {
            const Eigen::Vector4d lhs(0.0, 0.0, 10.0, 10.0);
            const Eigen::Vector4d rhs(5.0, 2.0, 15.0, 12.0);
            similarity = boxmot::trackers::base::AabbAssociationSimilarity(lhs, rhs, mode, 100, 80);
        }
        std::cout << std::setprecision(17) << similarity << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return 1;
    }
}
