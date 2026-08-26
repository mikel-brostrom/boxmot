#include "boxmot/trackers/base/reid_onnx.hpp"

#include "boxmot/trackers/base/reid_inference_backend.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace boxmot::trackers::base {

namespace {

constexpr double kPi = 3.14159265358979323846;

// ONNX uses protobuf, but the native tracker intentionally does not depend on
// the ONNX protobuf library. This small, bounds-checked reader extracts only
// ModelProto.graph.input[*].type.tensor_type.shape. Unknown fields are skipped
// according to the protobuf wire format, so it remains compatible with newer
// ONNX producers.
struct ProtoField {
    std::uint32_t number = 0;
    std::uint32_t wire_type = 0;
    std::uint64_t varint = 0;
    std::string_view bytes;
};

class ProtoReader {
public:
    explicit ProtoReader(std::string_view data)
        : cursor_(reinterpret_cast<const std::uint8_t*>(data.data())),
          end_(cursor_ + data.size()) {}

    bool Next(ProtoField& field) {
        if (cursor_ == end_) {
            return false;
        }

        const std::uint64_t key = ReadVarint();
        field = ProtoField{};
        field.number = static_cast<std::uint32_t>(key >> 3U);
        field.wire_type = static_cast<std::uint32_t>(key & 0x07U);
        if (field.number == 0) {
            throw std::runtime_error("Invalid zero field number in ONNX protobuf.");
        }

        switch (field.wire_type) {
            case 0:
                field.varint = ReadVarint();
                break;
            case 1:
                Skip(8);
                break;
            case 2: {
                const std::uint64_t length = ReadVarint();
                if (length > static_cast<std::uint64_t>(end_ - cursor_)) {
                    throw std::runtime_error("Truncated length-delimited ONNX protobuf field.");
                }
                field.bytes = std::string_view(
                    reinterpret_cast<const char*>(cursor_),
                    static_cast<std::size_t>(length));
                cursor_ += static_cast<std::ptrdiff_t>(length);
                break;
            }
            case 5:
                Skip(4);
                break;
            default:
                throw std::runtime_error("Unsupported ONNX protobuf wire type.");
        }
        return true;
    }

private:
    std::uint64_t ReadVarint() {
        std::uint64_t value = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            if (cursor_ == end_) {
                throw std::runtime_error("Truncated ONNX protobuf varint.");
            }
            const std::uint8_t byte = *cursor_++;
            value |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
            if ((byte & 0x80U) == 0) {
                return value;
            }
        }
        throw std::runtime_error("Oversized ONNX protobuf varint.");
    }

    void Skip(std::ptrdiff_t count) {
        if (count < 0 || count > end_ - cursor_) {
            throw std::runtime_error("Truncated fixed-width ONNX protobuf field.");
        }
        cursor_ += count;
    }

    const std::uint8_t* cursor_;
    const std::uint8_t* end_;
};

struct OnnxInputSpec {
    cv::Size size;
    int batch_size = 0;  // 0 means dynamic
};

std::int64_t ParseDimension(std::string_view message) {
    ProtoReader reader(message);
    ProtoField field;
    while (reader.Next(field)) {
        if (field.number == 1 && field.wire_type == 0) {
            if (field.varint <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
                const auto value = static_cast<std::int64_t>(field.varint);
                return value > 0 ? value : -1;
            }
            return -1;
        }
        if (field.number == 2 && field.wire_type == 2 && !field.bytes.empty()) {
            return -1;
        }
    }
    return -1;
}

std::vector<std::int64_t> ParseTensorShape(std::string_view message) {
    std::vector<std::int64_t> dimensions;
    ProtoReader reader(message);
    ProtoField field;
    while (reader.Next(field)) {
        if (field.number == 1 && field.wire_type == 2) {
            dimensions.push_back(ParseDimension(field.bytes));
        }
    }
    return dimensions;
}

std::vector<std::int64_t> ParseTensorType(std::string_view message) {
    ProtoReader reader(message);
    ProtoField field;
    while (reader.Next(field)) {
        if (field.number == 2 && field.wire_type == 2) {
            return ParseTensorShape(field.bytes);
        }
    }
    return {};
}

std::vector<std::int64_t> ParseValueInfo(std::string_view message) {
    ProtoReader reader(message);
    ProtoField field;
    while (reader.Next(field)) {
        if (field.number != 2 || field.wire_type != 2) {
            continue;
        }
        ProtoReader type_reader(field.bytes);
        ProtoField type_field;
        while (type_reader.Next(type_field)) {
            if (type_field.number == 1 && type_field.wire_type == 2) {
                return ParseTensorType(type_field.bytes);
            }
        }
    }
    return {};
}

std::optional<std::vector<std::int64_t>> ParseGraphInput(std::string_view message) {
    ProtoReader reader(message);
    ProtoField field;
    while (reader.Next(field)) {
        if (field.number == 11 && field.wire_type == 2) {
            std::vector<std::int64_t> shape = ParseValueInfo(field.bytes);
            if (shape.size() == 4 && (shape[1] == 3 || shape[1] < 0)) {
                return shape;
            }
        }
    }
    return std::nullopt;
}

OnnxInputSpec InspectOnnxInput(const fs::path& model_path) {
    std::ifstream stream(model_path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("Failed to open native ReID ONNX model: " + model_path.string());
    }
    const std::streampos end = stream.tellg();
    if (end <= 0) {
        throw std::runtime_error("Native ReID ONNX model is empty: " + model_path.string());
    }
    if (static_cast<std::uintmax_t>(end) > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("Native ReID ONNX model is too large to inspect.");
    }

    std::string model(static_cast<std::size_t>(end), '\0');
    stream.seekg(0, std::ios::beg);
    if (!stream.read(model.data(), static_cast<std::streamsize>(model.size()))) {
        throw std::runtime_error("Failed to read native ReID ONNX model: " + model_path.string());
    }

    ProtoReader reader(model);
    ProtoField field;
    std::optional<std::vector<std::int64_t>> shape;
    while (reader.Next(field)) {
        if (field.number == 7 && field.wire_type == 2) {
            shape = ParseGraphInput(field.bytes);
            break;
        }
    }
    if (!shape.has_value()) {
        throw std::runtime_error(
            "Native ReID ONNX model has no rank-4 image input in NCHW layout: " +
            model_path.string());
    }

    const auto& dims = *shape;
    if (dims[1] != 3) {
        throw std::runtime_error(
            "Native ReID ONNX input must have three statically-known channels.");
    }
    if (dims[2] <= 0 || dims[3] <= 0) {
        throw std::runtime_error(
            "Native ReID ONNX input height and width must be static positive dimensions.");
    }
    if (dims[2] > std::numeric_limits<int>::max() ||
        dims[3] > std::numeric_limits<int>::max() ||
        dims[0] > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Native ReID ONNX input dimensions exceed supported integer sizes.");
    }

    OnnxInputSpec spec;
    spec.size = cv::Size(static_cast<int>(dims[3]), static_cast<int>(dims[2]));
    spec.batch_size = dims[0] > 0 ? static_cast<int>(dims[0]) : 0;
    return spec;
}

double WrapCentered(double angle, double period) {
    const double half_period = period * 0.5;
    double wrapped = std::fmod(angle + half_period, period);
    if (wrapped < 0.0) {
        wrapped += period;
    }
    wrapped -= half_period;
    if (std::abs(wrapped) <= 1.0e-6) {
        return 0.0;
    }
    return wrapped;
}

Eigen::Matrix<double, 5, 1> CanonicalizeObbForInput(
    const Eigen::Matrix<double, 5, 1>& xywha,
    const cv::Size& input_size
) {
    Eigen::Matrix<double, 5, 1> canonical = xywha;
    double& width = canonical[2];
    double& height = canonical[3];
    double& angle = canonical[4];
    for (int index = 0; index < 5; ++index) {
        if (!std::isfinite(canonical[index])) {
            throw std::runtime_error("Native ReID OBB crop coordinates must be finite.");
        }
    }
    if (width <= 0.0 || height <= 0.0) {
        throw std::runtime_error("Native ReID OBB crop width and height must be positive.");
    }

    // ReID models are normally portrait, so align the box's longer physical
    // side with the input tensor's longer axis. A square input deliberately
    // uses the vertical axis as a deterministic tie-break.
    const bool target_long_axis_is_vertical = input_size.height >= input_size.width;
    const bool box_long_axis_is_vertical = height > width;
    if (target_long_axis_is_vertical != box_long_axis_is_vertical) {
        std::swap(width, height);
        angle += kPi * 0.5;
    }

    constexpr double kSquareRelativeTolerance = 1.0e-3;
    const bool nearly_square = std::abs(width - height) <=
        1.0e-6 + kSquareRelativeTolerance * std::abs(height);
    // Center the undirected rectangle angle around zero. Besides making
    // equivalent encodings identical, this avoids a 180-degree crop flip
    // when detector jitter crosses the modulo boundary.
    angle = WrapCentered(angle, nearly_square ? kPi * 0.5 : kPi);
    return canonical;
}

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

    const OnnxInputSpec input_spec = InspectOnnxInput(model_path_);
    input_size_ = input_spec.size;
    input_batch_size_ = input_spec.batch_size;
    inference_ = MakeReIdInferenceBackend(
        model_path_, backend_, device_, input_size_, input_batch_size_);
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

    // ONNX Runtime accepts the runtime N of a dynamic graph, so forward the
    // complete staged crop batch in one call. OpenCV DNN remains crop-by-crop
    // for dynamic graphs because some exported ReID heads mishandle N>1.
    // Fixed-batch graphs are chunked, with the final chunk zero-padded to the
    // graph's exact N and trimmed back to its logical row count.
    const int per_crop_floats = 3 * input_size_.height * input_size_.width;
    const bool use_dynamic_batch =
        input_batch_size_ == 0 && inference_->supports_dynamic_batch();
    if (use_dynamic_batch && crops.count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Native ReID crop batch exceeds supported integer dimensions.");
    }
    const int execution_batch = input_batch_size_ > 0
        ? input_batch_size_
        : (use_dynamic_batch ? static_cast<int>(crops.count) : 1);
    const int dims_execution[] = {
        execution_batch, 3, input_size_.height, input_size_.width};

    for (std::size_t offset = 0; offset < crops.count;
         offset += static_cast<std::size_t>(execution_batch)) {
        const std::size_t logical_count = std::min(
            static_cast<std::size_t>(execution_batch), crops.count - offset);

        cv::Mat execution_blob;
        if (logical_count == static_cast<std::size_t>(execution_batch)) {
            execution_blob = cv::Mat(
                4, dims_execution, CV_32F,
                reinterpret_cast<float*>(crops.blob.data) +
                    static_cast<std::ptrdiff_t>(offset) * per_crop_floats);
        } else {
            execution_blob = cv::Mat(4, dims_execution, CV_32F, cv::Scalar(0));
            const std::size_t logical_floats =
                logical_count * static_cast<std::size_t>(per_crop_floats);
            const float* source = reinterpret_cast<const float*>(crops.blob.data) +
                static_cast<std::ptrdiff_t>(offset) * per_crop_floats;
            std::copy(source, source + logical_floats,
                      reinterpret_cast<float*>(execution_blob.data));
        }

        std::vector<float> feature = inference_->Forward(execution_blob);
        if (feature.size() % static_cast<std::size_t>(execution_batch) != 0) {
            throw std::runtime_error(
                "Native ReID output size is not divisible by the execution batch size.");
        }
        const std::size_t current_feature_dim =
            feature.size() / static_cast<std::size_t>(execution_batch);
        if (raw.feature_dim == 0) {
            raw.feature_dim = current_feature_dim;
            raw.data.resize(raw.feature_dim * crops.count);
        } else if (current_feature_dim != raw.feature_dim) {
            throw std::runtime_error(
                "Native ReID returned a feature dimension that changed mid-batch.");
        }

        for (std::size_t row = 0; row < logical_count; ++row) {
            const auto source_begin = feature.begin() +
                static_cast<std::ptrdiff_t>(row * raw.feature_dim);
            const auto destination_begin = raw.data.begin() +
                static_cast<std::ptrdiff_t>((offset + row) * raw.feature_dim);
            std::copy(source_begin,
                      source_begin + static_cast<std::ptrdiff_t>(raw.feature_dim),
                      destination_begin);
        }
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
    const Eigen::Matrix<double, 5, 1> canonical =
        CanonicalizeObbForInput(xywha, input_size_);
    const double cx = canonical[0];
    const double cy = canonical[1];
    const double max_output_side = static_cast<double>(
        std::max(input_size_.height, input_size_.width));
    const double scale = std::min(
        1.0, max_output_side / std::max(canonical[2], canonical[3]));
    const int dst_w = std::max(
        1, static_cast<int>(std::round(canonical[2] * scale)));
    const int dst_h = std::max(
        1, static_cast<int>(std::round(canonical[3] * scale)));
    const double angle_deg = canonical[4] * 180.0 / kPi;

    // ``warpAffine`` performs ``dst(x,y) = src(M * [x,y,1]^T)`` so we need a
    // matrix that takes a destination pixel (x,y) in the axis-aligned crop and
    // returns the source coordinate in the rotated original. Equivalently:
    // rotate the source about the OBB centre by -angle, then translate so the
    // OBB centre lands at the crop centre.
    cv::Mat rotation = cv::getRotationMatrix2D(
        cv::Point2f(static_cast<float>(cx), static_cast<float>(cy)),
        angle_deg,
        scale
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
        feature[index] = std::isfinite(data[index]) ? data[index] : 0.0F;
    }
    const float norm = feature.norm();
    if (std::isfinite(norm) && norm > 1.0e-12F) {
        feature /= norm;
    } else if (!std::isfinite(norm)) {
        feature.setZero();
    }
    return feature;
}

cv::Mat OnnxReIdModel::ResizePad(const cv::Mat& crop, const cv::Size& target_size) {
    if (crop.empty()) {
        return cv::Mat(target_size, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    const double scale = std::min(
        static_cast<double>(target_size.width) / static_cast<double>(crop.cols),
        static_cast<double>(target_size.height) / static_cast<double>(crop.rows)
    );
    // Match Python's ``int(value)`` semantics exactly (positive-value floor).
    const int resized_width = std::max(1, static_cast<int>(crop.cols * scale));
    const int resized_height = std::max(1, static_cast<int>(crop.rows * scale));

    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);

    const int pad_left = (target_size.width - resized_width) / 2;
    const int pad_right = target_size.width - resized_width - pad_left;
    const int pad_top = (target_size.height - resized_height) / 2;
    const int pad_bottom = target_size.height - resized_height - pad_top;

    cv::Mat padded;
    cv::copyMakeBorder(
        resized, padded, pad_top, pad_bottom, pad_left, pad_right,
        // Crops are still BGR here. Python pads with ImageNet's BGR mean
        // before converting the complete image to RGB.
        cv::BORDER_CONSTANT, cv::Scalar(104, 116, 124)
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
