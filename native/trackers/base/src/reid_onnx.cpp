#include "boxmot/trackers/base/reid_onnx.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace boxmot::trackers::base {

namespace {

constexpr double kPi = 3.14159265358979323846;

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

OnnxReIdModel::OnnxReIdModel(fs::path model_path, std::string preprocess_name)
    : model_path_(std::move(model_path)),
      preprocess_name_(std::move(preprocess_name)),
      input_size_(LooksLikeLmbnModel(model_path_) ? cv::Size(128, 384) : cv::Size(128, 256)),
      mean_(0.485, 0.456, 0.406),
      std_(0.229, 0.224, 0.225) {
    if (model_path_.empty()) {
        return;
    }
    if (!fs::exists(model_path_)) {
        throw std::runtime_error("Native ReID ONNX model not found: " + model_path_.string());
    }
    if (model_path_.extension() != ".onnx") {
        throw std::runtime_error("Native ReID currently supports ONNX models only: " + model_path_.string());
    }
    net_ = cv::dnn::readNetFromONNX(model_path_.string());
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    initialized_ = true;
}

std::vector<Eigen::VectorXf> OnnxReIdModel::GetFeaturesForBoxes(
    const std::vector<cv::Rect>& boxes,
    const cv::Mat& image
) const {
    std::vector<Eigen::VectorXf> features;
    features.reserve(boxes.size());
    if (!initialized_ || boxes.empty()) {
        return features;
    }

    // OpenCV DNN can mis-handle the batch dimension for these exported ReID heads,
    // collapsing N>1 into the feature dimension before the final Gemm. Running one
    // crop at a time avoids that shape corruption and keeps native live tracking stable.
    for (const auto& box : boxes) {
        cv::Mat blob = BuildInputBlob({PreprocessCrop(ExtractCrop(box, image))});
        net_.setInput(blob);
        cv::Mat output = net_.forward();
        cv::Mat reshaped = output.reshape(1, 1);
        if (reshaped.rows != 1) {
            throw std::runtime_error("Native ReID ONNX output shape is incompatible with single-crop inference.");
        }
        features.push_back(NormalizeFeature(reshaped.row(0)));
    }
    return features;
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

Eigen::VectorXf OnnxReIdModel::NormalizeFeature(const cv::Mat& feature_row) {
    Eigen::VectorXf feature(feature_row.cols);
    for (int index = 0; index < feature_row.cols; ++index) {
        feature[index] = feature_row.at<float>(0, index);
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
    const std::string& preprocess_name
) {
    if (model_path.empty()) {
        return std::nullopt;
    }
    return OnnxReIdModel(model_path, preprocess_name);
}

}  // namespace boxmot::trackers::base
