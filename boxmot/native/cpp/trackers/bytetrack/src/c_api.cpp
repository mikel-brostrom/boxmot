#include "bytetrack/c_api.hpp"

#include "boxmot/trackers/base/live_c_api.hpp"
#include "bytetrack/tracker.hpp"
#include "bytetrack/types.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_last_error;

bytetrack::Config ConvertConfig(const BoxMOTByteTrackConfig& config) {
    bytetrack::Config native_config;
    native_config.min_conf = config.min_conf;
    native_config.track_thresh = config.track_thresh;
    native_config.match_thresh = config.match_thresh;
    native_config.track_buffer = config.track_buffer;
    native_config.frame_rate = config.frame_rate;
    native_config.max_obs = config.max_obs;
    native_config.asso_func = config.asso_func == nullptr ? "iou" : std::string(config.asso_func);
    return native_config;
}

}  // namespace

struct BoxMOTByteTrackHandle {
    explicit BoxMOTByteTrackHandle(bytetrack::Config tracker_config)
        : config(std::move(tracker_config)),
          tracker(std::make_unique<bytetrack::ByteTrackTracker>(config)) {}

    bytetrack::Config config;
    std::unique_ptr<bytetrack::ByteTrackTracker> tracker;
};

extern "C" {

BoxMOTByteTrackHandle* boxmot_bytetrack_create(const BoxMOTByteTrackConfig* config) {
    try {
        if (config == nullptr) {
            throw std::runtime_error("Native ByteTrack config is required.");
        }
        bytetrack::Config native_config = ConvertConfig(*config);
        g_last_error.clear();
        return new BoxMOTByteTrackHandle(std::move(native_config));
    } catch (const std::exception& exc) {
        boxmot::native::SetLastError(g_last_error, exc.what());
        return nullptr;
    } catch (...) {
        boxmot::native::SetLastError(g_last_error,
                                             "Unknown native ByteTrack creation failure");
        return nullptr;
    }
}

void boxmot_bytetrack_destroy(BoxMOTByteTrackHandle* handle) {
    delete handle;
}

int boxmot_bytetrack_reset(BoxMOTByteTrackHandle* handle) {
    return boxmot::native::GuardCall(
        [&]() {
            if (handle == nullptr) {
                throw std::runtime_error("Native ByteTrack handle is null.");
            }
            handle->tracker = std::make_unique<bytetrack::ByteTrackTracker>(handle->config);
        },
        g_last_error,
        "Unknown native ByteTrack failure");
}

int boxmot_bytetrack_update_v2(BoxMOTByteTrackHandle* handle,
                               const BoxMOTDetectionBatchV2* detections,
                               const BoxMOTImageV2* image,
                               BoxMOTTrackBatchV2** output) {
    return boxmot::native::GuardCall(
        [&]() {
            if (handle == nullptr || handle->tracker == nullptr) {
                throw std::runtime_error("Native ByteTrack handle is not initialized.");
            }
            if (detections == nullptr || output == nullptr) {
                throw std::runtime_error("Native ByteTrack input/output pointers are null.");
            }
            *output = nullptr;
            const std::vector<bytetrack::Detection> converted =
                boxmot::trackers::base::ConvertLiveDetectionsV2<bytetrack::Detection>(*detections,
                                                                                      "ByteTrack");
            const cv::Mat image_mat =
                boxmot::trackers::base::WrapOptionalLiveImageV2(image, "ByteTrack");
            const std::vector<bytetrack::TrackOutput> tracks =
                handle->tracker->Update(converted, image_mat);
            *output =
                boxmot::trackers::base::AllocateLiveOutputV2(tracks, detections->geometry_cols);
        },
        g_last_error,
        "Unknown native ByteTrack failure");
}

void boxmot_bytetrack_result_free_v2(BoxMOTTrackBatchV2* output) {
    boxmot::trackers::base::FreeLiveOutputV2(output);
}

const char* boxmot_bytetrack_last_error() {
    return g_last_error.c_str();
}

}  // extern "C"
