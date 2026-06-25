#include "botsort/c_api.hpp"

#include "boxmot/trackers/base/live_c_api.hpp"
#include "botsort/tracker.hpp"
#include "botsort/types.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_last_error;

botsort::Config ConvertConfig(const BoxMOTBotSortConfig& config) {
    botsort::Config native_config;
    native_config.track_high_thresh = config.track_high_thresh;
    native_config.track_low_thresh = config.track_low_thresh;
    native_config.new_track_thresh = config.new_track_thresh;
    native_config.track_buffer = config.track_buffer;
    native_config.match_thresh = config.match_thresh;
    native_config.proximity_thresh = config.proximity_thresh;
    native_config.appearance_thresh = config.appearance_thresh;
    native_config.cmc_method = config.cmc_method == nullptr ? "ecc" : std::string(config.cmc_method);
    native_config.frame_rate = config.frame_rate;
    native_config.fuse_first_associate = config.fuse_first_associate != 0;
    native_config.with_reid = config.with_reid != 0;
    native_config.max_obs = config.max_obs;
    native_config.reid_model_path = config.reid_model_path == nullptr ? "" : std::string(config.reid_model_path);
    native_config.reid_preprocess = config.reid_preprocess == nullptr ? "resize_pad" : std::string(config.reid_preprocess);
    return native_config;
}

std::vector<botsort::Detection> ConvertDetections(
    const float* dets,
    const int det_rows,
    const int det_cols,
    const float* embs,
    const int emb_rows,
    const int emb_cols
) {
    if (det_rows < 0 || det_cols < 0 || emb_rows < 0 || emb_cols < 0) {
        throw std::runtime_error("Negative matrix dimensions are not allowed.");
    }
    if (embs != nullptr && emb_rows != det_rows) {
        throw std::runtime_error("Detection and embedding row counts must match.");
    }

    std::vector<botsort::Detection> converted =
        boxmot::trackers::base::ConvertLiveDetections<botsort::Detection>(dets, det_rows, det_cols, "BoTSORT");
    for (std::size_t row = 0; row < converted.size(); ++row) {
        if (embs != nullptr && emb_cols > 0) {
            botsort::Detection& detection = converted[row];
            detection.embedding.resize(emb_cols);
            const float* emb_row = embs + (row * static_cast<std::size_t>(emb_cols));
            for (int col = 0; col < emb_cols; ++col) {
                detection.embedding(col) = emb_row[col];
            }
        }
    }
    return converted;
}

}  // namespace

struct BoxMOTBotSortHandle {
    explicit BoxMOTBotSortHandle(botsort::Config tracker_config)
        : config(std::move(tracker_config)), tracker(std::make_unique<botsort::BotSortTracker>(config)) {}

    botsort::Config config;
    std::unique_ptr<botsort::BotSortTracker> tracker;
};

extern "C" {

BoxMOTBotSortHandle* boxmot_botsort_create(const BoxMOTBotSortConfig* config) {
    try {
        if (config == nullptr) {
            throw std::runtime_error("Native BoTSORT config is required.");
        }
        botsort::Config native_config = ConvertConfig(*config);
        g_last_error.clear();
        return new BoxMOTBotSortHandle(std::move(native_config));
    } catch (const std::exception& exc) {
        boxmot::trackers::base::SetLastError(g_last_error, exc.what());
        return nullptr;
    } catch (...) {
        boxmot::trackers::base::SetLastError(g_last_error, "Unknown native BoTSORT creation failure");
        return nullptr;
    }
}

void boxmot_botsort_destroy(BoxMOTBotSortHandle* handle) {
    delete handle;
}

int boxmot_botsort_reset(BoxMOTBotSortHandle* handle) {
    return boxmot::trackers::base::GuardCall([&]() {
        if (handle == nullptr) {
            throw std::runtime_error("Native BoTSORT handle is null.");
        }
        handle->tracker = std::make_unique<botsort::BotSortTracker>(handle->config);
    }, g_last_error, "Unknown native BoTSORT failure");
}

int boxmot_botsort_update(
    BoxMOTBotSortHandle* handle,
    const float* dets,
    const int det_rows,
    const int det_cols,
    const float* embs,
    const int emb_rows,
    const int emb_cols,
    const std::uint8_t* image_data,
    const int image_rows,
    const int image_cols,
    const int image_channels,
    float* out_tracks,
    const int out_capacity_rows,
    const int out_cols,
    int* out_rows,
    int* out_is_obb
) {
    return boxmot::trackers::base::GuardCall([&]() {
        if (handle == nullptr || handle->tracker == nullptr) {
            throw std::runtime_error("Native BoTSORT handle is not initialized.");
        }
        if (out_rows == nullptr || out_is_obb == nullptr) {
            throw std::runtime_error("Output pointers are null.");
        }
        const std::vector<botsort::Detection> detections =
            ConvertDetections(dets, det_rows, det_cols, embs, emb_rows, emb_cols);
        const cv::Mat image =
            boxmot::trackers::base::WrapLiveImage(image_data, image_rows, image_cols, image_channels, "BoTSORT");
        const std::vector<botsort::TrackOutput> tracks = handle->tracker->Update(detections, image);
        boxmot::trackers::base::WriteLiveOutputs(tracks, out_tracks, out_capacity_rows, out_cols, "BoTSORT");
        *out_rows = static_cast<int>(tracks.size());
        *out_is_obb = boxmot::trackers::base::LiveOutputUsesObb(tracks, det_cols) ? 1 : 0;
    }, g_last_error, "Unknown native BoTSORT failure");
}

int boxmot_botsort_last_reid_time_ms(BoxMOTBotSortHandle* handle, double* out_reid_time_ms) {
    return boxmot::trackers::base::GuardCall([&]() {
        if (handle == nullptr || handle->tracker == nullptr) {
            throw std::runtime_error("Native BoTSORT handle is not initialized.");
        }
        if (out_reid_time_ms == nullptr) {
            throw std::runtime_error("Output ReID timing pointer is null.");
        }
        *out_reid_time_ms = handle->tracker->LastReIdTimeMs();
    }, g_last_error, "Unknown native BoTSORT failure");
}

int boxmot_botsort_last_reid_preprocess_time_ms(BoxMOTBotSortHandle* handle, double* out_time_ms) {
    return boxmot::trackers::base::GuardCall([&]() {
        if (handle == nullptr || handle->tracker == nullptr) {
            throw std::runtime_error("Native BoTSORT handle is not initialized.");
        }
        if (out_time_ms == nullptr) {
            throw std::runtime_error("Output ReID timing pointer is null.");
        }
        *out_time_ms = handle->tracker->LastReIdPreprocessTimeMs();
    }, g_last_error, "Unknown native BoTSORT failure");
}

int boxmot_botsort_last_reid_process_time_ms(BoxMOTBotSortHandle* handle, double* out_time_ms) {
    return boxmot::trackers::base::GuardCall([&]() {
        if (handle == nullptr || handle->tracker == nullptr) {
            throw std::runtime_error("Native BoTSORT handle is not initialized.");
        }
        if (out_time_ms == nullptr) {
            throw std::runtime_error("Output ReID timing pointer is null.");
        }
        *out_time_ms = handle->tracker->LastReIdProcessTimeMs();
    }, g_last_error, "Unknown native BoTSORT failure");
}

int boxmot_botsort_last_reid_postprocess_time_ms(BoxMOTBotSortHandle* handle, double* out_time_ms) {
    return boxmot::trackers::base::GuardCall([&]() {
        if (handle == nullptr || handle->tracker == nullptr) {
            throw std::runtime_error("Native BoTSORT handle is not initialized.");
        }
        if (out_time_ms == nullptr) {
            throw std::runtime_error("Output ReID timing pointer is null.");
        }
        *out_time_ms = handle->tracker->LastReIdPostprocessTimeMs();
    }, g_last_error, "Unknown native BoTSORT failure");
}

const char* boxmot_botsort_last_error() {
    return g_last_error.c_str();
}

}  // extern "C"
