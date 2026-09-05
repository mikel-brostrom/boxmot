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
    native_config.asso_func = config.asso_func == nullptr ? "iou" : std::string(config.asso_func);
    return native_config;
}

}  // namespace

struct BoxMOTOCSORTHandle {
    explicit BoxMOTOCSORTHandle(ocsort::Config tracker_config)
        : config(std::move(tracker_config)),
          tracker(std::make_unique<ocsort::OCSORTTracker>(config)) {}

    ocsort::Config config;
    std::unique_ptr<ocsort::OCSORTTracker> tracker;
};

extern "C" {

BoxMOTOCSORTHandle* boxmot_ocsort_create(const BoxMOTOCSORTConfig* config) {
    try {
        if (config == nullptr) {
            throw std::runtime_error("Native OcSort config is required.");
        }
        ocsort::Config native_config = ConvertConfig(*config);
        g_last_error.clear();
        return new BoxMOTOCSORTHandle(std::move(native_config));
    } catch (const std::exception& exc) {
        boxmot::native::SetLastError(g_last_error, exc.what());
        return nullptr;
    } catch (...) {
        boxmot::native::SetLastError(g_last_error,
                                             "Unknown native OcSort creation failure");
        return nullptr;
    }
}

void boxmot_ocsort_destroy(BoxMOTOCSORTHandle* handle) {
    delete handle;
}

int boxmot_ocsort_reset(BoxMOTOCSORTHandle* handle) {
    return boxmot::native::GuardCall(
        [&]() {
            if (handle == nullptr) {
                throw std::runtime_error("Native OcSort handle is null.");
            }
            handle->tracker = std::make_unique<ocsort::OCSORTTracker>(handle->config);
        },
        g_last_error,
        "Unknown native OcSort failure");
}

int boxmot_ocsort_update_v2(BoxMOTOCSORTHandle* handle,
                            const BoxMOTDetectionBatchV2* detections,
                            const BoxMOTImageV2* image,
                            BoxMOTTrackBatchV2** output) {
    return boxmot::native::GuardCall(
        [&]() {
            if (handle == nullptr || handle->tracker == nullptr) {
                throw std::runtime_error("Native OcSort handle is not initialized.");
            }
            if (detections == nullptr || output == nullptr) {
                throw std::runtime_error("Native OcSort input/output pointers are null.");
            }
            *output = nullptr;
            const std::vector<ocsort::Detection> converted =
                boxmot::trackers::base::ConvertLiveDetectionsV2<ocsort::Detection>(*detections,
                                                                                   "OcSort");
            const cv::Mat image_mat =
                boxmot::trackers::base::WrapOptionalLiveImageV2(image, "OcSort");
            const std::vector<ocsort::TrackOutput> tracks =
                handle->tracker->Update(converted, image_mat);
            *output =
                boxmot::trackers::base::AllocateLiveOutputV2(tracks, detections->geometry_cols);
        },
        g_last_error,
        "Unknown native OcSort failure");
}

void boxmot_ocsort_result_free_v2(BoxMOTTrackBatchV2* output) {
    boxmot::trackers::base::FreeLiveOutputV2(output);
}

const char* boxmot_ocsort_last_error() {
    return g_last_error.c_str();
}

}  // extern "C"
