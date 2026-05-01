#pragma once

#include "ocsort/types.hpp"

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace ocsort {

struct LoadedSequence {
    std::string name;
    Eigen::MatrixXf detections;
    std::vector<int> frame_ids;
    std::vector<fs::path> frame_paths;
    bool is_obb = false;
};

LoadedSequence LoadSequence(const ReplayOptions& options);
std::vector<Detection> SliceDetectionsForFrame(
    const LoadedSequence& sequence,
    int frame_id,
    std::size_t& row_offset,
    float conf_threshold
);
cv::Mat ReadImage(const fs::path& path);
void WriteResultLine(std::ostream& stream, int frame_id, const TrackOutput& track);

}  // namespace ocsort
