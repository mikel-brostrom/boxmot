#include "sfsort/c_api.hpp"

#include "boxmot/trackers/base/live_c_api.hpp"
#include "sfsort/tracker.hpp"
#include "sfsort/types.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_last_error;

sfsort::Config ConvertConfig(const BoxMOTSFSORTConfig& config) {
    sfsort::Config native_config;
    native_config.high_th = config.high_th;
    native_config.match_th_first = config.match_th_first;
    native_config.new_track_th = config.new_track_th;
    native_config.low_th = config.low_th;
    native_config.match_th_second = config.match_th_second;
    native_config.dynamic_tuning = config.dynamic_tuning != 0;
    native_config.cth = config.cth;
    native_config.high_th_m = config.high_th_m;
    native_config.new_track_th_m = config.new_track_th_m;
    native_config.match_th_first_m = config.match_th_first_m;
    native_config.obb_theta_damping = config.obb_theta_damping;
    native_config.marginal_timeout = config.marginal_timeout;
    native_config.central_timeout = config.central_timeout;
    native_config.frame_width = config.frame_width;
    native_config.frame_height = config.frame_height;
    native_config.horizontal_margin = config.horizontal_margin;
    native_config.vertical_margin = config.vertical_margin;
    native_config.frame_rate = config.frame_rate;
    native_config.max_obs = config.max_obs;
    return native_config;
}

}  // namespace

struct BoxMOTSFSORTHandle {
    explicit BoxMOTSFSORTHandle(sfsort::Config tracker_config)
        : config(std::move(tracker_config)), tracker(std::make_unique<sfsort::SFSORTTracker>(config)) {}

    sfsort::Config config;
    std::unique_ptr<sfsort::SFSORTTracker> tracker;
};

extern "C" {

BoxMOTSFSORTHandle* boxmot_sfsort_create(const BoxMOTSFSORTConfig* config) {
    try {
        if (config == nullptr) {
            throw std::runtime_error("Native SFSORT config is required.");
        }
        sfsort::Config native_config = ConvertConfig(*config);
        g_last_error.clear();
        return new BoxMOTSFSORTHandle(std::move(native_config));
    } catch (const std::exception& exc) {
        boxmot::trackers::base::SetLastError(g_last_error, exc.what());
        return nullptr;
    } catch (...) {
        boxmot::trackers::base::SetLastError(g_last_error, "Unknown native SFSORT creation failure");
        return nullptr;
    }
}

void boxmot_sfsort_destroy(BoxMOTSFSORTHandle* handle) {
    delete handle;
}

int boxmot_sfsort_reset(BoxMOTSFSORTHandle* handle) {
    return boxmot::trackers::base::GuardCall([&]() {
        if (handle == nullptr) {
            throw std::runtime_error("Native SFSORT handle is null.");
        }
        handle->tracker = std::make_unique<sfsort::SFSORTTracker>(handle->config);
    }, g_last_error, "Unknown native SFSORT failure");
}

int boxmot_sfsort_update(
    BoxMOTSFSORTHandle* handle,
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
            throw std::runtime_error("Native SFSORT handle is not initialized.");
        }
        if (out_rows == nullptr || out_is_obb == nullptr) {
            throw std::runtime_error("Output pointers are null.");
        }
        const std::vector<sfsort::Detection> detections =
            boxmot::trackers::base::ConvertLiveDetections<sfsort::Detection>(dets, det_rows, det_cols, "SFSORT");
        const cv::Mat image =
            boxmot::trackers::base::WrapLiveImage(image_data, image_rows, image_cols, image_channels, "SFSORT");
        const std::vector<sfsort::TrackOutput> tracks = handle->tracker->Update(detections, image);
        boxmot::trackers::base::WriteLiveOutputs(tracks, out_tracks, out_capacity_rows, out_cols, "SFSORT");
        *out_rows = static_cast<int>(tracks.size());
        *out_is_obb = boxmot::trackers::base::LiveOutputUsesObb(tracks, det_cols) ? 1 : 0;
    }, g_last_error, "Unknown native SFSORT failure");
}

const char* boxmot_sfsort_last_error() {
    return g_last_error.c_str();
}

}  // extern "C"
