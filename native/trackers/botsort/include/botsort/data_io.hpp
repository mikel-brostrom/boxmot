#pragma once

#include "botsort/types.hpp"

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace botsort {

struct LoadedSequence {
    std::string name;
    Eigen::MatrixXf detections;
    Eigen::MatrixXf embeddings;
    bool is_obb = false;
    std::vector<int> frame_ids;
    std::vector<fs::path> frame_paths;
};

LoadedSequence LoadSequence(const ReplayOptions& options);
std::vector<Detection> SliceDetectionsForFrame(
    const LoadedSequence& sequence,
    int frame_id,
    std::size_t& row_offset,
    float conf_threshold
);
cv::Mat ReadImage(const fs::path& path);
void WriteMotLine(std::ostream& stream, int frame_id, const TrackOutput& track);

}  // namespace botsort
