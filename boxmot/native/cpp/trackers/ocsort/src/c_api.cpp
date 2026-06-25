#include "ocsort/c_api.hpp"

#include "boxmot/trackers/base/live_c_api.hpp"
#include "ocsort/tracker.hpp"
#include "ocsort/types.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_last_error;

ocsort::Config ConvertConfig(const BoxMOTOCSORTConfig& config) {
    ocsort::Config native_config;
    native_config.min_conf = config.min_conf;
    native_config.det_thresh = config.det_thresh;
    native_config.iou_threshold = config.iou_threshold;
    native_config.max_age = config.max_age;
    native_config.min_hits = config.min_hits;
    native_config.delta_t = config.delta_t;
    native_config.use_byte = config.use_byte != 0;
    native_config.inertia = config.inertia;
    native_config.q_xy_scaling = config.q_xy_scaling;
    native_config.q_s_scaling = config.q_s_scaling;
    native_config.max_obs = config.max_obs;
    return native_config;
}

}  // namespace

struct BoxMOTOCSORTHandle {
    explicit BoxMOTOCSORTHandle(ocsort::Config tracker_config)
        : config(std::move(tracker_config)), tracker(std::make_unique<ocsort::OCSORTTracker>(config)) {}

    ocsort::Config config;
    std::unique_ptr<ocsort::OCSORTTracker> tracker;
};

extern "C" {

BoxMOTOCSORTHandle* boxmot_ocsort_create(const BoxMOTOCSORTConfig* config) {
    try {
        if (config == nullptr) {
            throw std::runtime_error("Native OCSORT config is required.");
        }
        ocsort::Config native_config = ConvertConfig(*config);
        g_last_error.clear();
        return new BoxMOTOCSORTHandle(std::move(native_config));
    } catch (const std::exception& exc) {
        boxmot::trackers::base::SetLastError(g_last_error, exc.what());
        return nullptr;
    } catch (...) {
        boxmot::trackers::base::SetLastError(g_last_error, "Unknown native OCSORT creation failure");
        return nullptr;
    }
}

void boxmot_ocsort_destroy(BoxMOTOCSORTHandle* handle) {
    delete handle;
}

int boxmot_ocsort_reset(BoxMOTOCSORTHandle* handle) {
    return boxmot::trackers::base::GuardCall([&]() {
        if (handle == nullptr) {
            throw std::runtime_error("Native OCSORT handle is null.");
        }
        handle->tracker = std::make_unique<ocsort::OCSORTTracker>(handle->config);
    }, g_last_error, "Unknown native OCSORT failure");
}

int boxmot_ocsort_update(
    BoxMOTOCSORTHandle* handle,
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
            throw std::runtime_error("Native OCSORT handle is not initialized.");
        }
        if (out_rows == nullptr || out_is_obb == nullptr) {
            throw std::runtime_error("Output pointers are null.");
        }
        const std::vector<ocsort::Detection> detections =
            boxmot::trackers::base::ConvertLiveDetections<ocsort::Detection>(dets, det_rows, det_cols, "OCSORT");
        const cv::Mat image =
            boxmot::trackers::base::WrapLiveImage(image_data, image_rows, image_cols, image_channels, "OCSORT");
        const std::vector<ocsort::TrackOutput> tracks = handle->tracker->Update(detections, image);
        boxmot::trackers::base::WriteLiveOutputs(tracks, out_tracks, out_capacity_rows, out_cols, "OCSORT");
        *out_rows = static_cast<int>(tracks.size());
        *out_is_obb = boxmot::trackers::base::LiveOutputUsesObb(tracks, det_cols) ? 1 : 0;
    }, g_last_error, "Unknown native OCSORT failure");
}

const char* boxmot_ocsort_last_error() {
    return g_last_error.c_str();
}

}  // extern "C"
