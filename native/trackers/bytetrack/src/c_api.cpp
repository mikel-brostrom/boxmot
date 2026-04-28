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
    return native_config;
}

}  // namespace

struct BoxMOTByteTrackHandle {
    explicit BoxMOTByteTrackHandle(bytetrack::Config tracker_config)
        : config(std::move(tracker_config)), tracker(std::make_unique<bytetrack::ByteTrackTracker>(config)) {}

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
        boxmot::trackers::base::SetLastError(g_last_error, exc.what());
        return nullptr;
    } catch (...) {
        boxmot::trackers::base::SetLastError(g_last_error, "Unknown native ByteTrack creation failure");
        return nullptr;
    }
}

void boxmot_bytetrack_destroy(BoxMOTByteTrackHandle* handle) {
    delete handle;
}

int boxmot_bytetrack_reset(BoxMOTByteTrackHandle* handle) {
    return boxmot::trackers::base::GuardCall([&]() {
        if (handle == nullptr) {
            throw std::runtime_error("Native ByteTrack handle is null.");
        }
        handle->tracker = std::make_unique<bytetrack::ByteTrackTracker>(handle->config);
    }, g_last_error, "Unknown native ByteTrack failure");
}

int boxmot_bytetrack_update(
    BoxMOTByteTrackHandle* handle,
    const float* dets,
    const int det_rows,
    const int det_cols,
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
            throw std::runtime_error("Native ByteTrack handle is not initialized.");
        }
        if (out_rows == nullptr || out_is_obb == nullptr) {
            throw std::runtime_error("Output pointers are null.");
        }
        const std::vector<bytetrack::Detection> detections =
            boxmot::trackers::base::ConvertLiveDetections<bytetrack::Detection>(
                dets,
                det_rows,
                det_cols,
                "ByteTrack"
            );
        const cv::Mat image =
            boxmot::trackers::base::WrapLiveImage(image_data, image_rows, image_cols, image_channels, "ByteTrack");
        const std::vector<bytetrack::TrackOutput> tracks = handle->tracker->Update(detections, image);
        boxmot::trackers::base::WriteLiveOutputs(tracks, out_tracks, out_capacity_rows, out_cols, "ByteTrack");
        *out_rows = static_cast<int>(tracks.size());
        *out_is_obb = boxmot::trackers::base::LiveOutputUsesObb(tracks, det_cols) ? 1 : 0;
    }, g_last_error, "Unknown native ByteTrack failure");
}

const char* boxmot_bytetrack_last_error() {
    return g_last_error.c_str();
}

}  // extern "C"
