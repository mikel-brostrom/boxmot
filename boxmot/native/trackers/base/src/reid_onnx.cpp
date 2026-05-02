#include "boxmot/trackers/base/reid_onnx.hpp"

#include "boxmot/trackers/base/reid_inference_backend.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

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

}  // namespace

cv::Rect ClampBoxToImage(const Eigen::Vector4d& xyxy, const cv::Size& image_size) {
    const int x1 = std::clamp(static_cast<int>(xyxy[0]), 0, image_size.width - 1);
    const int y1 = std::clamp(static_cast<int>(xyxy[1]), 0, image_size.height - 1);
    const int x2 = std::clamp(static_cast<int>(xyxy[2]), 0, image_size.width - 1);
    const int y2 = std::clamp(static_cast<int>(xyxy[3]), 0, image_size.height - 1);
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

    inference_ = MakeReIdInferenceBackend(model_path_, backend_, device_, input_size_);
    if (!inference_) {
        throw std::runtime_error("Failed to initialise native ReID inference backend.");
    }
    backend_ = inference_->kind();
    device_ = inference_->device();
    initialized_ = true;
}

OnnxReIdModel::~OnnxReIdModel() = default;
OnnxReIdModel::OnnxReIdModel(OnnxReIdModel&&) noexcept = default;
OnnxReIdModel& OnnxReIdModel::operator=(OnnxReIdModel&&) noexcept = default;

OnnxReIdModel::CropBatch OnnxReIdModel::Preprocess(
    const std::vector<cv::Rect>& boxes,
    const cv::Mat& image
) const {
    CropBatch batch;
    if (!initialized_ || boxes.empty()) {
        return batch;
    }

    std::vector<cv::Mat> processed;
    processed.reserve(boxes.size());
    for (const auto& box : boxes) {
        processed.push_back(PreprocessCrop(ExtractCrop(box, image)));
    }
    batch.blob = BuildInputBlob(processed);
    batch.count = boxes.size();
    return batch;
}

OnnxReIdModel::RawFeatures OnnxReIdModel::Process(const CropBatch& crops) const {
    RawFeatures raw;
    raw.count = crops.count;
    if (!initialized_ || crops.count == 0 || !inference_) {
        return raw;
    }

    // The OpenCV DNN backend mishandles N>1 for these exported ReID heads
    // (collapses batch into the feature dim before the final Gemm), so we
    // forward each crop individually. The ORT backend does the same to keep
    // outputs identical across backends.
    const int per_crop_floats = 3 * input_size_.height * input_size_.width;
    const int dims_single[] = {1, 3, input_size_.height, input_size_.width};

    for (std::size_t i = 0; i < crops.count; ++i) {
        cv::Mat single(4, dims_single, CV_32F,
                       reinterpret_cast<float*>(crops.blob.data) +
                           static_cast<std::ptrdiff_t>(i) * per_crop_floats);
        std::vector<float> feature = inference_->Forward(single);
        if (raw.feature_dim == 0) {
            raw.feature_dim = feature.size();
            raw.data.resize(raw.feature_dim * crops.count);
        } else if (feature.size() != raw.feature_dim) {
            throw std::runtime_error(
                "Native ReID returned a feature dimension that changed mid-batch.");
        }
        std::copy(feature.begin(), feature.end(),
                  raw.data.begin() + static_cast<std::ptrdiff_t>(i * raw.feature_dim));
    }
    return raw;
}

std::vector<Eigen::VectorXf> OnnxReIdModel::Postprocess(const RawFeatures& raw) const {
    std::vector<Eigen::VectorXf> features;
    features.reserve(raw.count);
    if (raw.count == 0 || raw.feature_dim == 0) {
        return features;
    }
    for (std::size_t i = 0; i < raw.count; ++i) {
        const float* row = raw.data.data() + i * raw.feature_dim;
        features.push_back(NormalizeFeature(row, static_cast<int>(raw.feature_dim)));
    }
    return features;
}

std::vector<Eigen::VectorXf> OnnxReIdModel::GetFeaturesForBoxes(
    const std::vector<cv::Rect>& boxes,
    const cv::Mat& image
) const {
    return Postprocess(Process(Preprocess(boxes, image)));
}

OnnxReIdModel::CropBatch OnnxReIdModel::PreprocessObb(
    const std::vector<Eigen::Matrix<double, 5, 1>>& boxes,
    const cv::Mat& image
) const {
    CropBatch batch;
    if (!initialized_ || boxes.empty()) {
        return batch;
    }

    std::vector<cv::Mat> processed;
    processed.reserve(boxes.size());
    for (const auto& obb : boxes) {
        processed.push_back(PreprocessCrop(ExtractObbCrop(obb, image)));
    }
    batch.blob = BuildInputBlob(processed);
    batch.count = boxes.size();
    return batch;
}

std::vector<Eigen::VectorXf> OnnxReIdModel::GetFeaturesForObbBoxes(
    const std::vector<Eigen::Matrix<double, 5, 1>>& boxes,
    const cv::Mat& image
) const {
    return Postprocess(Process(PreprocessObb(boxes, image)));
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
