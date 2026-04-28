#pragma once

#include "botsort/types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <optional>
#include <string>
#include <vector>

namespace botsort {

class OnnxReIdModel {
public:
    OnnxReIdModel(fs::path model_path, std::string preprocess_name = "resize_pad");

    [[nodiscard]] bool valid() const { return initialized_; }
    [[nodiscard]] const fs::path& model_path() const { return model_path_; }
    [[nodiscard]] const std::string& preprocess_name() const { return preprocess_name_; }

    std::vector<Eigen::VectorXf> GetFeatures(const std::vector<Detection>& detections, const cv::Mat& image) const;

private:
    cv::Mat PreprocessCrop(const cv::Mat& crop) const;
    cv::Mat ExtractCrop(const Detection& detection, const cv::Mat& image) const;
    cv::Mat BuildInputBlob(const std::vector<cv::Mat>& processed_crops) const;
    static Eigen::VectorXf NormalizeFeature(const cv::Mat& feature_row);
    static bool LooksLikeLmbnModel(const fs::path& model_path);
    static cv::Mat ResizePad(const cv::Mat& crop, const cv::Size& target_size);

    fs::path model_path_;
    std::string preprocess_name_;
    cv::Size input_size_;
    cv::Scalar mean_;
    cv::Scalar std_;
    mutable cv::dnn::Net net_;
    bool initialized_ = false;
};

std::optional<OnnxReIdModel> MaybeCreateOnnxReIdModel(
    const fs::path& model_path,
    const std::string& preprocess_name = "resize_pad"
);

}  // namespace botsort