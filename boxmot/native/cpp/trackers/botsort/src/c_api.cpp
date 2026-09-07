#include "botsort/c_api.hpp"

#include "botsort/tracker.hpp"
#include "botsort/types.hpp"
#include "boxmot/trackers/base/live_c_api.hpp"

#include <opencv2/core.hpp>

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
    native_config.second_match_thresh = config.second_match_thresh;
    native_config.unconfirmed_match_thresh = config.unconfirmed_match_thresh;
    native_config.unconfirmed_emb_scale = config.unconfirmed_emb_scale;
    native_config.cmc_method =
        config.cmc_method == nullptr ? "ecc" : std::string(config.cmc_method);
    native_config.frame_rate = config.frame_rate;
    native_config.fuse_first_associate = config.fuse_first_associate != 0;
    native_config.use_embeddings = config.use_embeddings != 0;
    native_config.max_obs = config.max_obs;
    native_config.asso_func = config.asso_func == nullptr ? "iou" : std::string(config.asso_func);
    return native_config;
}

}  // namespace

struct BoxMOTBotSortHandle {
    explicit BoxMOTBotSortHandle(botsort::Config tracker_config)
        : config(std::move(tracker_config)),
          tracker(std::make_unique<botsort::BotSortTracker>(config)) {}

    botsort::Config config;
    std::unique_ptr<botsort::BotSortTracker> tracker;
};

extern "C" {

BoxMOTBotSortHandle* boxmot_botsort_create(const BoxMOTBotSortConfig* config) {
    try {
        if (config == nullptr) {
            throw std::runtime_error("Native BotSort config is required.");
        }
        botsort::Config native_config = ConvertConfig(*config);
        g_last_error.clear();
        return new BoxMOTBotSortHandle(std::move(native_config));
    } catch (const std::exception& exc) {
        boxmot::native::SetLastError(g_last_error, exc.what());
        return nullptr;
    } catch (...) {
        boxmot::native::SetLastError(g_last_error,
                                             "Unknown native BotSort creation failure");
        return nullptr;
    }
}

void boxmot_botsort_destroy(BoxMOTBotSortHandle* handle) {
    delete handle;
}

int boxmot_botsort_reset(BoxMOTBotSortHandle* handle) {
    return boxmot::native::GuardCall(
        [&]() {
            if (handle == nullptr) {
                throw std::runtime_error("Native BotSort handle is null.");
            }
            handle->tracker = std::make_unique<botsort::BotSortTracker>(handle->config);
        },
        g_last_error,
        "Unknown native BotSort failure");
}

int boxmot_botsort_update_v2(BoxMOTBotSortHandle* handle,
                             const BoxMOTDetectionBatchV2* detections,
                             const BoxMOTImageV2* image,
                             BoxMOTTrackBatchV2** output) {
    return boxmot::native::GuardCall(
        [&]() {
            if (handle == nullptr || handle->tracker == nullptr) {
                throw std::runtime_error("Native BotSort handle is not initialized.");
            }
            if (detections == nullptr || output == nullptr) {
                throw std::runtime_error("Native BotSort input/output pointers are null.");
            }
            *output = nullptr;
            const std::vector<botsort::Detection> converted =
                boxmot::trackers::base::ConvertLiveDetectionsV2<botsort::Detection, true>(
                    *detections, "BotSort");
            const cv::Mat image_mat =
                boxmot::trackers::base::WrapOptionalLiveImageV2(image, "BotSort");
            const bool cmc_needs_image =
                !handle->config.cmc_method.empty() && handle->config.cmc_method != "none";
            if (image_mat.empty() && cmc_needs_image) {
                throw std::runtime_error("Native BotSort requires an image when CMC is active.");
            }
            if (handle->config.use_embeddings && detections->rows > 0 &&
                detections->embedding_cols <= 0) {
                throw std::runtime_error("Native BotSort requires precomputed embeddings.");
            }
            const std::vector<botsort::TrackOutput> tracks =
                handle->tracker->Update(converted, image_mat);
            *output =
                boxmot::trackers::base::AllocateLiveOutputV2(tracks, detections->geometry_cols);
        },
        g_last_error,
        "Unknown native BotSort failure");
}

void boxmot_botsort_result_free_v2(BoxMOTTrackBatchV2* output) {
    boxmot::trackers::base::FreeLiveOutputV2(output);
}

const char* boxmot_botsort_last_error() {
    return g_last_error.c_str();
}

}  // extern "C"
