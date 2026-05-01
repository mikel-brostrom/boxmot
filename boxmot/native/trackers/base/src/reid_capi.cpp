// Standalone C ABI for the native ONNX ReID model.
//
// Exposes ``OnnxReIdModel`` to host languages (currently used by Python via
// ctypes) so that the eval pipeline can populate the embedding cache using the
// exact same C++ inference path the C++ trackers use at replay time. This
// eliminates the small numerical drift between the Python ONNX backend (which
// goes through CoreML EP with fp16 internally) and the native tracker's own
// inference that previously caused metric differences when switching backends.

#include "boxmot/trackers/base/reid_capi.h"
#include "boxmot/trackers/base/reid_onnx.hpp"
#include "boxmot/trackers/base/native_runtime.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;

using ::boxmot::trackers::base::OnnxReIdModel;
using ::boxmot::trackers::base::ReIdBackend;
using ::boxmot::trackers::base::ReIdDevice;

cv::Mat WrapImage(const std::uint8_t* image_data, int image_rows, int image_cols, int image_channels) {
    if (image_data == nullptr) {
        throw std::runtime_error("Image data pointer is null.");
    }
    if (image_rows <= 0 || image_cols <= 0) {
        throw std::runtime_error("Image dimensions must be positive.");
    }
    const int cv_type = ::boxmot::trackers::base::CvImageType(
        image_channels, "Native ReID supports uint8 images with 1, 3, or 4 channels.");
    return cv::Mat(image_rows, image_cols, cv_type, const_cast<std::uint8_t*>(image_data));
}

cv::Rect BoxFromXyxy(const float* row, const cv::Size& image_size) {
    Eigen::Vector4d xyxy;
    xyxy << row[0], row[1], row[2], row[3];
    return ::boxmot::trackers::base::ClampBoxToImage(xyxy, image_size);
}

}  // namespace

struct BoxMOTReIdHandle {
    explicit BoxMOTReIdHandle(std::unique_ptr<OnnxReIdModel> model) : reid(std::move(model)) {}

    std::unique_ptr<OnnxReIdModel> reid;
    int feature_dim = 0;  // 0 == not yet probed
};

extern "C" {

int boxmot_reid_capi_create(const char* model_path, const char* preprocess, void** out_handle) {
    return ::boxmot::trackers::base::GuardCall(
        [&]() {
            if (out_handle == nullptr) {
                throw std::runtime_error("out_handle pointer is null.");
            }
            *out_handle = nullptr;
            if (model_path == nullptr || std::string(model_path).empty()) {
                throw std::runtime_error("Native ReID model path is required.");
            }
            std::string preprocess_name = preprocess == nullptr ? std::string("resize_pad") : std::string(preprocess);
            auto model = std::make_unique<OnnxReIdModel>(
                std::string(model_path), preprocess_name, ReIdBackend::kAuto, ReIdDevice::kAuto);
            if (!model->valid()) {
                throw std::runtime_error("Failed to initialize native ReID model.");
            }
            *out_handle = static_cast<void*>(new BoxMOTReIdHandle(std::move(model)));
        },
        g_last_error,
        "Unknown native ReID creation failure");
}

void boxmot_reid_capi_destroy(void* handle) {
    delete static_cast<BoxMOTReIdHandle*>(handle);
}

int boxmot_reid_capi_feature_dim(void* handle, int* out_feature_dim) {
    return ::boxmot::trackers::base::GuardCall(
        [&]() {
            if (handle == nullptr) {
                throw std::runtime_error("Native ReID handle is null.");
            }
            if (out_feature_dim == nullptr) {
                throw std::runtime_error("out_feature_dim pointer is null.");
            }
            auto* state = static_cast<BoxMOTReIdHandle*>(handle);
            if (state->feature_dim == 0) {
                // Probe with a tiny dummy frame so we don't have to expose
                // model introspection through the ORT pimpl.
                cv::Mat dummy(64, 64, CV_8UC3, cv::Scalar(0, 0, 0));
                std::vector<cv::Rect> boxes{cv::Rect(0, 0, 32, 32)};
                auto features = state->reid->GetFeaturesForBoxes(boxes, dummy);
                if (features.empty()) {
                    throw std::runtime_error("Native ReID probe returned no features.");
                }
                state->feature_dim = static_cast<int>(features.front().size());
            }
            *out_feature_dim = state->feature_dim;
        },
        g_last_error,
        "Unknown native ReID feature-dim probe failure");
}

int boxmot_reid_capi_compute_features(
    void* handle,
    const float* boxes_xyxy,
    int n_boxes,
    const std::uint8_t* image_data,
    int image_rows,
    int image_cols,
    int image_channels,
    float* out_features,
    int out_capacity_floats
) {
    return ::boxmot::trackers::base::GuardCall(
        [&]() {
            if (handle == nullptr) {
                throw std::runtime_error("Native ReID handle is null.");
            }
            auto* state = static_cast<BoxMOTReIdHandle*>(handle);
            if (n_boxes < 0) {
                throw std::runtime_error("Negative box count is not allowed.");
            }
            if (n_boxes == 0) {
                return;
            }
            if (boxes_xyxy == nullptr) {
                throw std::runtime_error("boxes_xyxy pointer is null.");
            }
            if (out_features == nullptr) {
                throw std::runtime_error("out_features pointer is null.");
            }

            cv::Mat image = WrapImage(image_data, image_rows, image_cols, image_channels);
            const cv::Size image_size = image.size();

            std::vector<cv::Rect> boxes;
            boxes.reserve(static_cast<std::size_t>(n_boxes));
            for (int i = 0; i < n_boxes; ++i) {
                boxes.push_back(BoxFromXyxy(boxes_xyxy + static_cast<std::ptrdiff_t>(i) * 4, image_size));
            }

            auto features = state->reid->GetFeaturesForBoxes(boxes, image);
            if (features.size() != static_cast<std::size_t>(n_boxes)) {
                throw std::runtime_error("Native ReID returned a different number of features than boxes.");
            }
            const int feature_dim = features.empty() ? 0 : static_cast<int>(features.front().size());
            if (state->feature_dim == 0) {
                state->feature_dim = feature_dim;
            } else if (state->feature_dim != feature_dim) {
                throw std::runtime_error("Native ReID returned a feature dimension that changed mid-stream.");
            }

            const std::size_t required = static_cast<std::size_t>(n_boxes) * static_cast<std::size_t>(feature_dim);
            if (out_capacity_floats < 0 || static_cast<std::size_t>(out_capacity_floats) < required) {
                throw std::runtime_error("Output buffer is too small for the requested features.");
            }

            for (int i = 0; i < n_boxes; ++i) {
                const Eigen::VectorXf& feature = features[static_cast<std::size_t>(i)];
                if (static_cast<int>(feature.size()) != feature_dim) {
                    throw std::runtime_error("Inconsistent feature dimension in native ReID output.");
                }
                std::copy(
                    feature.data(),
                    feature.data() + feature_dim,
                    out_features + static_cast<std::ptrdiff_t>(i) * feature_dim);
            }
        },
        g_last_error,
        "Unknown native ReID feature computation failure");
}

const char* boxmot_reid_capi_last_error(void) {
    return g_last_error.c_str();
}

}  // extern "C"
