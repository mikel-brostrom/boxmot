#include "boxmot/trackers/base/reid_onnx.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(BOXMOT_HAS_ONNXRUNTIME)
#include <onnxruntime_cxx_api.h>
#if defined(__APPLE__)
#include <coreml_provider_factory.h>
#endif
#endif

namespace boxmot::trackers::base {

namespace {

constexpr double kPi = 3.14159265358979323846;

std::string EnvOr(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    std::string result(value);
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

ReIdBackend ResolveBackend(ReIdBackend requested) {
    if (requested != ReIdBackend::kAuto) {
        return requested;
    }
    const std::string env = EnvOr("BOXMOT_REID_BACKEND", "auto");
    if (env == "opencv" || env == "dnn" || env == "opencv_dnn") {
        return ReIdBackend::kOpenCvDnn;
    }
    if (env == "ort" || env == "onnxruntime" || env == "onnx_runtime") {
        return ReIdBackend::kOnnxRuntime;
    }
#if defined(BOXMOT_HAS_ONNXRUNTIME)
    return ReIdBackend::kOnnxRuntime;
#else
    return ReIdBackend::kOpenCvDnn;
#endif
}

ReIdDevice ResolveDevice(ReIdDevice requested) {
    if (requested != ReIdDevice::kAuto) {
        return requested;
    }
    const std::string env = EnvOr("BOXMOT_REID_DEVICE", "auto");
    if (env == "cpu") return ReIdDevice::kCpu;
    if (env == "cuda" || env == "gpu") return ReIdDevice::kCuda;
    if (env == "coreml" || env == "mps" || env == "metal") return ReIdDevice::kCoreMl;
#if defined(__APPLE__)
    return ReIdDevice::kCoreMl;
#else
    return ReIdDevice::kCuda;  // honoured only if the EP is available; otherwise we fall back to CPU
#endif
}

const char* BackendName(ReIdBackend backend) {
    switch (backend) {
        case ReIdBackend::kOpenCvDnn: return "opencv_dnn";
        case ReIdBackend::kOnnxRuntime: return "onnxruntime";
        default: return "auto";
    }
}

const char* DeviceName(ReIdDevice device) {
    switch (device) {
        case ReIdDevice::kCpu: return "cpu";
        case ReIdDevice::kCuda: return "cuda";
        case ReIdDevice::kCoreMl: return "coreml";
        default: return "auto";
    }
}

}  // namespace

#if defined(BOXMOT_HAS_ONNXRUNTIME)

struct OnnxReIdModel::OrtSession {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "boxmot_reid"};
    Ort::SessionOptions options{};
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator{};
    std::string input_name;
    std::string output_name;
    std::vector<int64_t> input_shape;  // [1, 3, H, W]
    ReIdDevice resolved_device = ReIdDevice::kCpu;
};

#else

struct OnnxReIdModel::OrtSession {};

#endif  // BOXMOT_HAS_ONNXRUNTIME

cv::Rect ClampBoxToImage(const Eigen::Vector4d& xyxy, const cv::Size& image_size) {
    // Match Python `box.round().astype("int")` followed by `min(w, x2)` / `min(h, y2)`
    // (see boxmot/reid/backends/base_backend.py::get_crops). Using truncation or a
    // `width - 1` upper bound shifts the crop by up to a pixel and changes the
    // resampled tensor enough to drift L2-normalised ReID features.
    const int x1 = std::clamp(static_cast<int>(std::lround(xyxy[0])), 0, image_size.width);
    const int y1 = std::clamp(static_cast<int>(std::lround(xyxy[1])), 0, image_size.height);
    const int x2 = std::clamp(static_cast<int>(std::lround(xyxy[2])), 0, image_size.width);
    const int y2 = std::clamp(static_cast<int>(std::lround(xyxy[3])), 0, image_size.height);
    const int width = std::max(0, x2 - x1);
    const int height = std::max(0, y2 - y1);
    return cv::Rect(x1, y1, width, height);
}

Eigen::Vector4d ObbToEnclosingXyxy(const Eigen::Matrix<double, 5, 1>& xywha) {
    const cv::RotatedRect rect(
        cv::Point2f(static_cast<float>(xywha[0]), static_cast<float>(xywha[1])),
        cv::Size2f(
            static_cast<float>(std::max(xywha[2], 1.0e-4)),
            static_cast<float>(std::max(xywha[3], 1.0e-4))
        ),
        static_cast<float>(xywha[4] * 180.0 / kPi)
    );
    std::array<cv::Point2f, 4> corners{};
    rect.points(corners.data());

    double x1 = corners[0].x;
    double y1 = corners[0].y;
    double x2 = corners[0].x;
    double y2 = corners[0].y;
    for (const auto& point : corners) {
        x1 = std::min(x1, static_cast<double>(point.x));
        y1 = std::min(y1, static_cast<double>(point.y));
        x2 = std::max(x2, static_cast<double>(point.x));
        y2 = std::max(y2, static_cast<double>(point.y));
    }
    Eigen::Vector4d enclosing;
    enclosing << x1, y1, x2, y2;
    return enclosing;
}

OnnxReIdModel::OnnxReIdModel(
    fs::path model_path,
    std::string preprocess_name,
    ReIdBackend backend,
    ReIdDevice device
) : model_path_(std::move(model_path)),
    preprocess_name_(std::move(preprocess_name)),
    input_size_(LooksLikeLmbnModel(model_path_) ? cv::Size(128, 384) : cv::Size(128, 256)),
    mean_(0.485, 0.456, 0.406),
    std_(0.229, 0.224, 0.225),
    backend_(ResolveBackend(backend)),
    device_(ResolveDevice(device)) {
    if (model_path_.empty()) {
        return;
    }
    if (!fs::exists(model_path_)) {
        throw std::runtime_error("Native ReID ONNX model not found: " + model_path_.string());
    }
    if (model_path_.extension() != ".onnx") {
        throw std::runtime_error("Native ReID currently supports ONNX models only: " + model_path_.string());
    }

    if (backend_ == ReIdBackend::kOnnxRuntime) {
#if defined(BOXMOT_HAS_ONNXRUNTIME)
        ort_ = std::make_unique<OrtSession>();
        ort_->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        ort_->options.SetIntraOpNumThreads(1);

        ReIdDevice resolved = device_;
        bool provider_added = false;

        if (resolved == ReIdDevice::kCoreMl) {
#if defined(__APPLE__)
            uint32_t coreml_flags = 0;
            const OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CoreML(
                static_cast<OrtSessionOptions*>(ort_->options), coreml_flags);
            if (status == nullptr) {
                provider_added = true;
            } else {
                Ort::GetApi().ReleaseStatus(const_cast<OrtStatus*>(status));
                resolved = ReIdDevice::kCpu;
            }
#else
            resolved = ReIdDevice::kCpu;
#endif
        } else if (resolved == ReIdDevice::kCuda) {
            try {
                OrtCUDAProviderOptions cuda_opts{};
                ort_->options.AppendExecutionProvider_CUDA(cuda_opts);
                provider_added = true;
            } catch (const Ort::Exception&) {
                resolved = ReIdDevice::kCpu;
            }
        }

        (void)provider_added;
        ort_->resolved_device = resolved;
        device_ = resolved;

        ort_->session = std::make_unique<Ort::Session>(
            ort_->env, model_path_.string().c_str(), ort_->options);

        Ort::AllocatedStringPtr in_name = ort_->session->GetInputNameAllocated(0, ort_->allocator);
        Ort::AllocatedStringPtr out_name = ort_->session->GetOutputNameAllocated(0, ort_->allocator);
        ort_->input_name = in_name.get();
        ort_->output_name = out_name.get();
        ort_->input_shape = {1, 3,
                             static_cast<int64_t>(input_size_.height),
                             static_cast<int64_t>(input_size_.width)};
        initialized_ = true;
#else
        // ORT not compiled in: fall back to OpenCV DNN.
        backend_ = ReIdBackend::kOpenCvDnn;
        device_ = ReIdDevice::kCpu;
#endif
    }

    if (backend_ == ReIdBackend::kOpenCvDnn) {
        net_ = cv::dnn::readNetFromONNX(model_path_.string());
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        device_ = ReIdDevice::kCpu;
        initialized_ = true;
    }

    std::cerr << "[boxmot] native ReID using backend=" << BackendName(backend_)
              << " device=" << DeviceName(device_)
              << " model=" << model_path_.filename().string() << '\n';
}

OnnxReIdModel::~OnnxReIdModel() = default;
OnnxReIdModel::OnnxReIdModel(OnnxReIdModel&&) noexcept = default;
OnnxReIdModel& OnnxReIdModel::operator=(OnnxReIdModel&&) noexcept = default;

std::vector<Eigen::VectorXf> OnnxReIdModel::GetFeaturesForBoxes(
    const std::vector<cv::Rect>& boxes,
    const cv::Mat& image
) const {
    std::vector<Eigen::VectorXf> features;
    features.reserve(boxes.size());
    if (!initialized_ || boxes.empty()) {
        return features;
    }

    // Both backends here run a single crop per call: ORT to keep the API uniform,
    // OpenCV DNN because its batched path mis-handles N>1 for these exported ReID
    // heads (collapses batch into the feature dim before the final Gemm).
    for (const auto& box : boxes) {
        const cv::Mat processed = PreprocessCrop(ExtractCrop(box, image));
        if (backend_ == ReIdBackend::kOnnxRuntime) {
            features.push_back(RunOrt(processed));
        } else {
            features.push_back(RunOpenCv(processed));
        }
    }
    return features;
}

std::vector<Eigen::VectorXf> OnnxReIdModel::GetFeaturesForObbBoxes(
    const std::vector<Eigen::Matrix<double, 5, 1>>& boxes,
    const cv::Mat& image
) const {
    std::vector<Eigen::VectorXf> features;
    features.reserve(boxes.size());
    if (!initialized_ || boxes.empty()) {
        return features;
    }
    for (const auto& obb : boxes) {
        const cv::Mat processed = PreprocessCrop(ExtractObbCrop(obb, image));
        if (backend_ == ReIdBackend::kOnnxRuntime) {
            features.push_back(RunOrt(processed));
        } else {
            features.push_back(RunOpenCv(processed));
        }
    }
    return features;
}

Eigen::VectorXf OnnxReIdModel::RunOpenCv(const cv::Mat& processed_crop) const {
    cv::Mat blob = BuildInputBlob({processed_crop});
    net_.setInput(blob);
    cv::Mat output = net_.forward();
    cv::Mat reshaped = output.reshape(1, 1);
    if (reshaped.rows != 1) {
        throw std::runtime_error("Native ReID ONNX output shape is incompatible with single-crop inference.");
    }
    return NormalizeFeature(reshaped.ptr<float>(0), reshaped.cols);
}

Eigen::VectorXf OnnxReIdModel::RunOrt(const cv::Mat& processed_crop) const {
#if defined(BOXMOT_HAS_ONNXRUNTIME)
    cv::Mat blob = BuildInputBlob({processed_crop});  // [1,3,H,W] CV_32F, contiguous
    const size_t element_count = static_cast<size_t>(input_size_.height) *
                                 static_cast<size_t>(input_size_.width) * 3UL;

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        reinterpret_cast<float*>(blob.data),
        element_count,
        ort_->input_shape.data(),
        ort_->input_shape.size()
    );

    const char* input_names[] = {ort_->input_name.c_str()};
    const char* output_names[] = {ort_->output_name.c_str()};
    auto output_tensors = ort_->session->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );

    Ort::Value& output = output_tensors.front();
    const auto type_info = output.GetTensorTypeAndShapeInfo();
    const size_t feature_dim = type_info.GetElementCount();
    const float* data = output.GetTensorData<float>();
    return NormalizeFeature(data, static_cast<int>(feature_dim));
#else
    (void)processed_crop;
    throw std::runtime_error("OnnxReIdModel built without ONNX Runtime support.");
#endif
}

cv::Mat OnnxReIdModel::PreprocessCrop(const cv::Mat& crop) const {
    cv::Mat prepared;
    if (preprocess_name_ == "resize_pad") {
        prepared = ResizePad(crop, input_size_);
    } else {
        cv::resize(crop, prepared, input_size_, 0.0, 0.0, cv::INTER_LINEAR);
    }
    cv::cvtColor(prepared, prepared, cv::COLOR_BGR2RGB);
    prepared.convertTo(prepared, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> channels;
    cv::split(prepared, channels);
    for (int index = 0; index < 3; ++index) {
        channels[index] = (channels[index] - mean_[index]) / std_[index];
    }
    cv::merge(channels, prepared);
    return prepared;
}

cv::Mat OnnxReIdModel::ExtractCrop(const cv::Rect& box, const cv::Mat& image) const {
    if (image.empty() || box.width <= 0 || box.height <= 0) {
        return cv::Mat(input_size_, CV_8UC3, cv::Scalar(0, 0, 0));
    }
    const cv::Rect safe = box & cv::Rect(0, 0, image.cols, image.rows);
    if (safe.width <= 0 || safe.height <= 0) {
        return cv::Mat(input_size_, CV_8UC3, cv::Scalar(0, 0, 0));
    }
    return image(safe).clone();
}

cv::Mat OnnxReIdModel::ExtractObbCrop(
    const Eigen::Matrix<double, 5, 1>& xywha,
    const cv::Mat& image
) const {
    if (image.empty()) {
        return cv::Mat(input_size_, CV_8UC3, cv::Scalar(0, 0, 0));
    }
    const double cx = xywha[0];
    const double cy = xywha[1];
    const int dst_w = std::max(1, static_cast<int>(std::round(xywha[2])));
    const int dst_h = std::max(1, static_cast<int>(std::round(xywha[3])));
    const double angle_deg = xywha[4] * 180.0 / 3.14159265358979323846;

    // ``warpAffine`` performs ``dst(x,y) = src(M * [x,y,1]^T)`` so we need a
    // matrix that takes a destination pixel (x,y) in the axis-aligned crop and
    // returns the source coordinate in the rotated original. Equivalently:
    // rotate the source about the OBB centre by -angle, then translate so the
    // OBB centre lands at the crop centre.
    cv::Mat rotation = cv::getRotationMatrix2D(
        cv::Point2f(static_cast<float>(cx), static_cast<float>(cy)),
        angle_deg,
        1.0
    );
    rotation.at<double>(0, 2) += (dst_w * 0.5) - cx;
    rotation.at<double>(1, 2) += (dst_h * 0.5) - cy;

    cv::Mat crop;
    cv::warpAffine(
        image, crop, rotation,
        cv::Size(dst_w, dst_h),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0)
    );
    return crop;
}

cv::Mat OnnxReIdModel::BuildInputBlob(const std::vector<cv::Mat>& processed_crops) const {
    const int batch = static_cast<int>(processed_crops.size());
    const int dims[] = {batch, 3, input_size_.height, input_size_.width};
    cv::Mat blob(4, dims, CV_32F, cv::Scalar(0));

    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        std::vector<cv::Mat> channels;
        cv::split(processed_crops[batch_index], channels);
        for (int channel_index = 0; channel_index < 3; ++channel_index) {
            for (int row = 0; row < input_size_.height; ++row) {
                float* dst = blob.ptr<float>(batch_index, channel_index, row);
                const float* src = channels[channel_index].ptr<float>(row);
                std::copy(src, src + input_size_.width, dst);
            }
        }
    }
    return blob;
}

Eigen::VectorXf OnnxReIdModel::NormalizeFeature(const float* data, int size) {
    Eigen::VectorXf feature(size);
    for (int index = 0; index < size; ++index) {
        feature[index] = data[index];
    }
    const float norm = feature.norm();
    if (norm > 1.0e-12F) {
        feature /= norm;
    }
    return feature;
}

bool OnnxReIdModel::LooksLikeLmbnModel(const fs::path& model_path) {
    const std::string name = model_path.filename().string();
    return name.find("lmbn") != std::string::npos;
}

cv::Mat OnnxReIdModel::ResizePad(const cv::Mat& crop, const cv::Size& target_size) {
    if (crop.empty()) {
        return cv::Mat(target_size, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    const double scale = std::min(
        static_cast<double>(target_size.width) / static_cast<double>(crop.cols),
        static_cast<double>(target_size.height) / static_cast<double>(crop.rows)
    );
    const int resized_width = std::max(1, static_cast<int>(std::round(crop.cols * scale)));
    const int resized_height = std::max(1, static_cast<int>(std::round(crop.rows * scale)));

    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);

    const int pad_left = (target_size.width - resized_width) / 2;
    const int pad_right = target_size.width - resized_width - pad_left;
    const int pad_top = (target_size.height - resized_height) / 2;
    const int pad_bottom = target_size.height - resized_height - pad_top;

    cv::Mat padded;
    cv::copyMakeBorder(
        resized, padded, pad_top, pad_bottom, pad_left, pad_right,
        cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0)
    );
    return padded;
}

std::optional<OnnxReIdModel> MaybeCreateOnnxReIdModel(
    const fs::path& model_path,
    const std::string& preprocess_name,
    ReIdBackend backend,
    ReIdDevice device
) {
    if (model_path.empty()) {
        return std::nullopt;
    }
    return OnnxReIdModel(model_path, preprocess_name, backend, device);
}

}  // namespace boxmot::trackers::base
