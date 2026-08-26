#include "occluboost/cmc.hpp"
#include "occluboost/types.hpp"

#include <opencv2/core.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace {

class CmcMaskProbe final : public occluboost::CameraMotionCompensator {
public:
    using occluboost::CameraMotionCompensator::GenerateMask;

    cv::Mat Apply(const cv::Mat&, const std::vector<occluboost::Detection>&) override {
        return {};
    }
};

}  // namespace

int main(const int argc, char* argv[]) {
    constexpr int width = 64;
    constexpr int height = 48;
    constexpr double scale_x = 0.13;
    constexpr double scale_y = 0.17;

    std::vector<occluboost::Detection> detections;
    const std::string mode = argc > 1 ? argv[1] : "empty";
    if (mode == "aabb") {
        occluboost::Detection detection;
        detection.xyxy << 12.8, 7.9, 159.7, 111.9;
        detections.push_back(detection);
    } else if (mode == "obb") {
        occluboost::Detection detection;
        detection.is_obb = true;
        detection.xywha << 170.25, 95.75, 80.5, 30.25, 0.41;
        detections.push_back(detection);
    } else if (mode != "empty") {
        std::cerr << "Expected mask mode: empty, aabb, or obb\n";
        return 2;
    }

    const cv::Mat mask = CmcMaskProbe::GenerateMask(
        cv::Size(width, height), detections, scale_x, scale_y);
    for (int row = 0; row < mask.rows; ++row) {
        for (int column = 0; column < mask.cols; ++column) {
            std::cout << (mask.at<unsigned char>(row, column) == 0 ? '0' : '1');
        }
    }
    std::cout << '\n';
    return 0;
}
