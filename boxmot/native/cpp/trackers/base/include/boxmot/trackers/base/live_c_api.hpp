#pragma once

#include "boxmot/native/runtime.hpp"
#include "boxmot/trackers/base/c_api_v2.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace boxmot::trackers::base {

inline std::string LiveDetectionShapeMessage(const std::string_view tracker_name) {
    return "Native " + std::string(tracker_name) +
           " live tracking supports AABB geometry with 4 columns or OBB geometry with 5 columns.";
}

inline void ValidateLiveDetectionBatch(const BoxMOTDetectionBatchV2& batch,
                                       const std::string_view tracker_name) {
    if (batch.abi_version != BOXMOT_TRACKER_ABI_VERSION) {
        throw std::runtime_error("Unsupported native tracker ABI version.");
    }
    if (batch.rows < 0 || batch.geometry_cols < 0 || batch.embedding_cols < 0) {
        throw std::runtime_error("Negative matrix dimensions are not allowed.");
    }
    if (batch.geometry_cols != 4 && batch.geometry_cols != 5) {
        throw std::runtime_error(LiveDetectionShapeMessage(tracker_name));
    }
    if (batch.rows > 0 && (batch.geometry == nullptr || batch.scores == nullptr ||
                           batch.class_ids == nullptr || batch.detection_indices == nullptr)) {
        throw std::runtime_error("Non-empty native detection buffers must not be null.");
    }
    if (batch.embedding_cols > 0 && batch.rows > 0 && batch.embeddings == nullptr) {
        throw std::runtime_error("Non-empty native embedding buffer is null.");
    }
}

template <typename Detection, bool SupportsEmbeddings = false>
std::vector<Detection> ConvertLiveDetectionsV2(const BoxMOTDetectionBatchV2& batch,
                                               const std::string_view tracker_name) {
    ValidateLiveDetectionBatch(batch, tracker_name);
    if (batch.rows == 0) {
        return {};
    }

    std::vector<Detection> converted;
    converted.reserve(static_cast<std::size_t>(batch.rows));
    for (std::int64_t row = 0; row < batch.rows; ++row) {
        const float* geometry = batch.geometry + (row * batch.geometry_cols);
        for (std::int32_t column = 0; column < batch.geometry_cols; ++column) {
            if (!std::isfinite(geometry[column])) {
                throw std::runtime_error(
                    "Native tracker detections must contain only finite values.");
            }
        }
        if (!std::isfinite(batch.scores[row]) || batch.scores[row] < 0.0F ||
            batch.scores[row] > 1.0F) {
            throw std::runtime_error("Native tracker scores must be finite values in [0, 1].");
        }
        if (batch.class_ids[row] < 0) {
            throw std::runtime_error("Native tracker class IDs must be non-negative.");
        }
        if (batch.detection_indices[row] < 0) {
            throw std::runtime_error("Native tracker detection indices must be non-negative.");
        }
        Detection detection;
        detection.is_obb = batch.geometry_cols == 5;
        if (detection.is_obb) {
            if (geometry[2] <= 0.0F || geometry[3] <= 0.0F) {
                throw std::runtime_error(
                    "Native OBB detections must have positive width and height.");
            }
            detection.xywha << geometry[0], geometry[1], geometry[2], geometry[3], geometry[4];
        } else {
            if (geometry[2] <= geometry[0] || geometry[3] <= geometry[1]) {
                throw std::runtime_error(
                    "Native AABB detections must satisfy x2 > x1 and y2 > y1.");
            }
            detection.xyxy << geometry[0], geometry[1], geometry[2], geometry[3];
        }
        detection.conf = batch.scores[row];
        detection.cls = batch.class_ids[row];
        detection.det_ind = batch.detection_indices[row];
        if constexpr (SupportsEmbeddings) {
            if (batch.embedding_cols > 0) {
                detection.embedding.resize(batch.embedding_cols);
                const float* embedding = batch.embeddings + (row * batch.embedding_cols);
                for (std::int32_t column = 0; column < batch.embedding_cols; ++column) {
                    if (!std::isfinite(embedding[column])) {
                        throw std::runtime_error(
                            "Native tracker embeddings must contain only finite values.");
                    }
                    detection.embedding(column) = embedding[column];
                }
            }
        } else if (batch.embedding_cols != 0) {
            throw std::runtime_error("This native tracker does not accept embeddings.");
        }
        converted.push_back(std::move(detection));
    }
    return converted;
}

inline cv::Mat WrapLiveImage(const std::uint8_t* image_data,
                             const int image_rows,
                             const int image_cols,
                             const int image_channels,
                             const std::string_view tracker_name) {
    if (image_data == nullptr) {
        throw std::runtime_error("Image data pointer is null.");
    }
    if (image_rows <= 0 || image_cols <= 0) {
        throw std::runtime_error("Image dimensions must be positive.");
    }

    return cv::Mat(
        image_rows,
        image_cols,
        boxmot::native::CvImageType(
            image_channels,
            "Native " + std::string(tracker_name) +
                " live tracking supports uint8 images with 1, 3, or 4 channels."),
        const_cast<std::uint8_t*>(image_data));
}

inline cv::Mat WrapOptionalLiveImage(const std::uint8_t* image_data,
                                     const int image_rows,
                                     const int image_cols,
                                     const int image_channels,
                                     const std::string_view tracker_name) {
    const bool image_omitted =
        image_data == nullptr && image_rows == 0 && image_cols == 0 && image_channels == 0;
    if (image_omitted) {
        return {};
    }
    return WrapLiveImage(image_data, image_rows, image_cols, image_channels, tracker_name);
}

inline cv::Mat WrapOptionalLiveImageV2(const BoxMOTImageV2* image,
                                       const std::string_view tracker_name) {
    if (image == nullptr) {
        return {};
    }
    return WrapLiveImage(image->data, image->rows, image->cols, image->channels, tracker_name);
}

inline void FreeLiveOutputV2(BoxMOTTrackBatchV2* output) noexcept {
    if (output == nullptr) {
        return;
    }
    delete[] output->geometry;
    delete[] output->scores;
    delete[] output->track_ids;
    delete[] output->class_ids;
    delete[] output->detection_indices;
    delete output;
}

template <typename TrackOutput>
BoxMOTTrackBatchV2* AllocateLiveOutputV2(const std::vector<TrackOutput>& tracks,
                                         const std::int32_t geometry_cols) {
    if (geometry_cols != 4 && geometry_cols != 5) {
        throw std::runtime_error("Native output geometry must have 4 or 5 columns.");
    }
    auto output = std::make_unique<BoxMOTTrackBatchV2>();
    output->abi_version = BOXMOT_TRACKER_ABI_VERSION;
    output->geometry = nullptr;
    output->scores = nullptr;
    output->track_ids = nullptr;
    output->class_ids = nullptr;
    output->detection_indices = nullptr;
    output->rows = static_cast<std::int64_t>(tracks.size());
    output->geometry_cols = geometry_cols;
    if (tracks.empty()) {
        return output.release();
    }

    try {
        output->geometry = new float[tracks.size() * static_cast<std::size_t>(geometry_cols)];
        output->scores = new float[tracks.size()];
        output->track_ids = new std::int64_t[tracks.size()];
        output->class_ids = new std::int64_t[tracks.size()];
        output->detection_indices = new std::int64_t[tracks.size()];
        for (std::size_t row = 0; row < tracks.size(); ++row) {
            const TrackOutput& track = tracks[row];
            if ((geometry_cols == 5) != track.is_obb) {
                throw std::runtime_error("Native tracker returned mixed or unexpected geometry.");
            }
            float* geometry = output->geometry + (row * static_cast<std::size_t>(geometry_cols));
            if (track.is_obb) {
                for (std::int32_t column = 0; column < 5; ++column) {
                    geometry[column] = static_cast<float>(track.xywha[column]);
                }
            } else {
                for (std::int32_t column = 0; column < 4; ++column) {
                    geometry[column] = static_cast<float>(track.xyxy[column]);
                }
            }
            output->scores[row] = track.conf;
            output->track_ids[row] = static_cast<std::int64_t>(track.id);
            output->class_ids[row] = static_cast<std::int64_t>(track.cls);
            output->detection_indices[row] = static_cast<std::int64_t>(track.det_ind);
        }
    } catch (...) {
        FreeLiveOutputV2(output.release());
        throw;
    }
    return output.release();
}

}  // namespace boxmot::trackers::base
