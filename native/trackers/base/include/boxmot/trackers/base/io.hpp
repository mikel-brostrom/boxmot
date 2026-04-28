#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <array>
#include <filesystem>
#include <iomanip>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace boxmot::trackers::base {

namespace fs = std::filesystem;

struct LoadedDetectionSequence {
    std::string name;
    Eigen::MatrixXf detections;
    std::vector<int> frame_ids;
    std::vector<fs::path> frame_paths;
    std::unordered_set<int> keep_frames;
    bool is_obb = false;
};

int RoundLikeNumpy(double value);
Eigen::MatrixXf LoadNumericMatrix(const fs::path& path);
fs::path ResolveCacheFile(const fs::path& path_without_suffix);
fs::path SequenceImageDir(const fs::path& seq_dir);
std::vector<fs::path> ListSequenceFrames(const fs::path& img_dir);
int ParseFrameId(const fs::path& path);
int ReadSequenceFps(const fs::path& seq_dir);
std::unordered_set<int> ComputeWantedFrames(const std::vector<int>& frame_ids, int orig_fps, int target_fps);
Eigen::MatrixXf FilterRowsByFrame(const Eigen::MatrixXf& matrix, const std::unordered_set<int>& keep_frames);
LoadedDetectionSequence LoadDetectionSequence(
    const fs::path& mot_root,
    const fs::path& det_emb_root,
    const std::string& detector_name,
    const std::string& sequence_name,
    int target_fps,
    std::string_view tracker_name
);
std::array<cv::Point2f, 4> CanonicalObbCorners(const Eigen::Matrix<double, 5, 1>& box);
void FilterFrames(
    const std::unordered_set<int>& keep_frames,
    std::vector<int>& frame_ids,
    std::vector<fs::path>& frame_paths
);

template <typename Detection, typename DecorateDetection>
std::vector<Detection> SliceReplayDetectionsForFrame(
    const Eigen::MatrixXf& detections_matrix,
    const bool is_obb,
    const int frame_id,
    std::size_t& row_offset,
    const float conf_threshold,
    DecorateDetection decorate_detection
) {
    std::vector<Detection> detections;
    const int total_rows = static_cast<int>(detections_matrix.rows());
    while (
        row_offset < static_cast<std::size_t>(total_rows) &&
        static_cast<int>(detections_matrix(static_cast<int>(row_offset), 0)) < frame_id
    ) {
        ++row_offset;
    }

    int det_ind = 0;
    while (
        row_offset < static_cast<std::size_t>(total_rows) &&
        static_cast<int>(detections_matrix(static_cast<int>(row_offset), 0)) == frame_id
    ) {
        const int row = static_cast<int>(row_offset);
        Detection detection;
        detection.is_obb = is_obb;
        if (is_obb) {
            detection.xywha = detections_matrix.block<1, 5>(row, 1).transpose().template cast<double>();
            detection.conf = detections_matrix(row, 6);
            detection.cls = static_cast<int>(detections_matrix(row, 7));
        } else {
            detection.xyxy = detections_matrix.block<1, 4>(row, 1).transpose().template cast<double>();
            detection.conf = detections_matrix(row, 5);
            detection.cls = static_cast<int>(detections_matrix(row, 6));
        }
        detection.det_ind = det_ind;
        decorate_detection(detection, row);

        if (conf_threshold <= 0.0F || detection.conf >= conf_threshold) {
            detections.push_back(std::move(detection));
        }

        ++det_ind;
        ++row_offset;
    }
    return detections;
}

template <typename Detection>
std::vector<Detection> SliceReplayDetectionsForFrame(
    const Eigen::MatrixXf& detections_matrix,
    const bool is_obb,
    const int frame_id,
    std::size_t& row_offset,
    const float conf_threshold
) {
    return SliceReplayDetectionsForFrame<Detection>(
        detections_matrix,
        is_obb,
        frame_id,
        row_offset,
        conf_threshold,
        [](Detection&, int) {}
    );
}

template <typename TrackOutput>
void WriteMotResultLine(std::ostream& stream, const int frame_id, const TrackOutput& track) {
    if (track.is_obb) {
        const std::array<cv::Point2f, 4> corners = CanonicalObbCorners(track.xywha);

        stream << frame_id << ',' << track.id;
        stream << std::fixed << std::setprecision(6);
        for (const auto& corner : corners) {
            stream << ',' << corner.x << ',' << corner.y;
        }
        stream << ',' << track.conf
               << ',' << track.cls
               << ',' << track.det_ind
               << '\n';
        return;
    }

    const double width = track.xyxy[2] - track.xyxy[0];
    const double height = track.xyxy[3] - track.xyxy[1];
    stream << frame_id << ','
           << track.id << ','
           << RoundLikeNumpy(track.xyxy[0]) << ','
           << RoundLikeNumpy(track.xyxy[1]) << ','
           << RoundLikeNumpy(width) << ','
           << RoundLikeNumpy(height) << ','
           << std::fixed << std::setprecision(6) << track.conf << ','
           << (track.cls + 1) << ','
           << track.det_ind << '\n';
}

}  // namespace boxmot::trackers::base
