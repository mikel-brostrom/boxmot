#pragma once

#include "boxmot/trackers/base/native_runtime.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace boxmot::trackers::base {

inline std::string LiveDetectionShapeMessage(const std::string_view tracker_name) {
    return "Native " + std::string(tracker_name) +
           " live tracking supports AABB detections with 6 columns or OBB detections with 7 columns.";
}

inline void ValidateLiveDetectionShape(const int det_rows, const int det_cols, const std::string_view tracker_name) {
    if (det_rows < 0 || det_cols < 0) {
        throw std::runtime_error("Negative matrix dimensions are not allowed.");
    }
    if (det_cols == 0 && det_rows == 0) {
        return;
    }
    if (det_cols != 6 && det_cols != 7) {
        throw std::runtime_error(LiveDetectionShapeMessage(tracker_name));
    }
}

template <typename Detection>
std::vector<Detection> ConvertLiveDetections(
    const float* dets,
    const int det_rows,
    const int det_cols,
    const std::string_view tracker_name
) {
    ValidateLiveDetectionShape(det_rows, det_cols, tracker_name);
    if (det_rows == 0) {
        return {};
    }
    if (dets == nullptr) {
        throw std::runtime_error("Detection data pointer is null.");
    }

    std::vector<Detection> converted;
    converted.reserve(static_cast<std::size_t>(det_rows));
    for (int row = 0; row < det_rows; ++row) {
        const float* det_row = dets + (row * det_cols);
        Detection detection;
        detection.is_obb = det_cols == 7;
        if (detection.is_obb) {
            detection.xywha << det_row[0], det_row[1], det_row[2], det_row[3], det_row[4];
            detection.conf = det_row[5];
            detection.cls = static_cast<int>(det_row[6]);
        } else {
            detection.xyxy << det_row[0], det_row[1], det_row[2], det_row[3];
            detection.conf = det_row[4];
            detection.cls = static_cast<int>(det_row[5]);
        }
        detection.det_ind = row;
        converted.push_back(std::move(detection));
    }
    return converted;
}

inline cv::Mat WrapLiveImage(
    const std::uint8_t* image_data,
    const int image_rows,
    const int image_cols,
    const int image_channels,
    const std::string_view tracker_name
) {
    if (image_data == nullptr) {
        throw std::runtime_error("Image data pointer is null.");
    }
    if (image_rows <= 0 || image_cols <= 0) {
        throw std::runtime_error("Image dimensions must be positive.");
    }

    return cv::Mat(
        image_rows,
        image_cols,
        CvImageType(
            image_channels,
            "Native " + std::string(tracker_name) + " live tracking supports uint8 images with 1, 3, or 4 channels."
        ),
        const_cast<std::uint8_t*>(image_data)
    );
}

template <typename TrackOutput>
void WriteLiveOutputs(
    const std::vector<TrackOutput>& tracks,
    float* out_tracks,
    const int out_capacity_rows,
    const int out_cols,
    const std::string_view tracker_name
) {
    if (out_capacity_rows < 0 || out_cols < 0) {
        throw std::runtime_error("Negative output matrix dimensions are not allowed.");
    }
    if (out_cols != 9) {
        throw std::runtime_error("Native " + std::string(tracker_name) +
                                 " live tracking expects an output buffer with 9 columns.");
    }
    if (tracks.size() > static_cast<std::size_t>(out_capacity_rows)) {
        throw std::runtime_error("Native " + std::string(tracker_name) +
                                 " output buffer is too small for the current frame.");
    }
    if (!tracks.empty() && out_tracks == nullptr) {
        throw std::runtime_error("Output track buffer is null.");
    }

    for (std::size_t row = 0; row < tracks.size(); ++row) {
        const TrackOutput& track = tracks[row];
        float* out_row = out_tracks + (row * static_cast<std::size_t>(out_cols));
        if (track.is_obb) {
            out_row[0] = static_cast<float>(track.xywha[0]);
            out_row[1] = static_cast<float>(track.xywha[1]);
            out_row[2] = static_cast<float>(track.xywha[2]);
            out_row[3] = static_cast<float>(track.xywha[3]);
            out_row[4] = static_cast<float>(track.xywha[4]);
            out_row[5] = static_cast<float>(track.id);
            out_row[6] = track.conf;
            out_row[7] = static_cast<float>(track.cls);
            out_row[8] = static_cast<float>(track.det_ind);
            continue;
        }

        out_row[0] = static_cast<float>(track.xyxy[0]);
        out_row[1] = static_cast<float>(track.xyxy[1]);
        out_row[2] = static_cast<float>(track.xyxy[2]);
        out_row[3] = static_cast<float>(track.xyxy[3]);
        out_row[4] = static_cast<float>(track.id);
        out_row[5] = track.conf;
        out_row[6] = static_cast<float>(track.cls);
        out_row[7] = static_cast<float>(track.det_ind);
        out_row[8] = 0.0F;
    }
}

template <typename TrackOutput>
bool LiveOutputUsesObb(const std::vector<TrackOutput>& tracks, const int det_cols) {
    return (!tracks.empty() && tracks.front().is_obb) || det_cols == 7;
}

}  // namespace boxmot::trackers::base
