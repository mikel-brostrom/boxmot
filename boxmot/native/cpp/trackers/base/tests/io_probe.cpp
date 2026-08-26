#include "boxmot/trackers/base/io.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

int main(const int argc, char* argv[]) {
    if (argc == 3 && std::string(argv[1]) == "--matrix-shape") {
        try {
            const Eigen::MatrixXf matrix = boxmot::trackers::base::LoadNumericMatrix(argv[2]);
            std::cout << matrix.rows() << ' ' << matrix.cols() << '\n';
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
        return 0;
    }
    if (argc == 5 && std::string(argv[1]) == "--wanted-frames") {
        try {
            const int orig_fps = std::stoi(argv[2]);
            const int target_fps = std::stoi(argv[3]);
            const int max_frame = std::stoi(argv[4]);
            std::vector<int> frame_ids(static_cast<std::size_t>(max_frame));
            std::iota(frame_ids.begin(), frame_ids.end(), 1);
            const auto wanted = boxmot::trackers::base::ComputeWantedFrames(
                frame_ids, orig_fps, target_fps);
            bool first = true;
            for (const int frame_id : frame_ids) {
                if (wanted.count(frame_id) == 0) {
                    continue;
                }
                std::cout << (first ? "" : " ") << frame_id;
                first = false;
            }
            std::cout << '\n';
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
        return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--list-frames") {
        try {
            const auto frames = boxmot::trackers::base::ListSequenceFrames(argv[2]);
            for (const auto& frame : frames) {
                std::cout << frame.filename().string() << '\n';
            }
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
        return 0;
    }
    if (argc != 2) {
        std::cerr << "expected an image path or a supported probe mode\n";
        return 2;
    }
    try {
        const cv::Mat image = boxmot::trackers::base::LoadNpyImage(argv[1]);
        if (image.empty()) {
            std::cerr << "loaded image is empty\n";
            return 1;
        }
        std::cout << image.rows << ' ' << image.cols << ' ' << image.channels();
        const auto* first_pixel = image.ptr<std::uint8_t>(0);
        for (int channel = 0; channel < image.channels(); ++channel) {
            std::cout << ' ' << static_cast<int>(first_pixel[channel]);
        }
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
