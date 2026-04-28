#include "occluboost/reid_onnx.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace occluboost {

namespace {

cv::Rect ClampRect(const Eigen::Vector4d& xyxy, const cv::Size& image_size) {
    const int x1 = std::max(0, static_cast<int>(std::round(xyxy[0])));
    const int y1 = std::max(0, static_cast<int>(std::round(xyxy[1])));
    const int x2 = std::min(image_size.width, static_cast<int>(std::round(xyxy[2])));
    const int y2 = std::min(image_size.height, static_cast<int>(std::round(xyxy[3])));
    const int width = std::max(0, x2 - x1);
    const int height = std::max(0, y2 - y1);
    return cv::Rect(x1, y1, width, height);
}

}  // namespace

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

std::vector<Eigen::VectorXf> OnnxReIdModel::GetFeatures(
    const std::vector<Detection>& detections,
    const cv::Mat& image
) const {
    std::vector<Eigen::VectorXf> features;
    features.reserve(detections.size());
    if (!initialized_ || detections.empty()) {
        return features;
    }

    for (const auto& detection : detections) {
        cv::Mat blob = BuildInputBlob({PreprocessCrop(ExtractCrop(detection, image))});
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

cv::Mat OnnxReIdModel::ExtractCrop(const Detection& detection, const cv::Mat& image) const {
    if (image.empty()) {
        return cv::Mat(input_size_, CV_8UC3, cv::Scalar(0, 0, 0));
    }
    const cv::Rect rect = ClampRect(detection.xyxy, image.size());
    if (rect.width <= 0 || rect.height <= 0) {
        return cv::Mat(input_size_, CV_8UC3, cv::Scalar(0, 0, 0));
    }
    return image(rect).clone();
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

}  // namespace occluboost
