#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace boxmot::trackers::base {

namespace fs = std::filesystem;

// Inference backend used by ``OnnxReIdModel``. ``kAuto`` resolves at construction
// time based on availability and the ``BOXMOT_REID_BACKEND`` env override.
enum class ReIdBackend {
    kAuto,
    kOpenCvDnn,
    kOnnxRuntime,
};

// Hardware target. ``kAuto`` selects the best available execution provider for
// the chosen backend (CoreML on macOS, CUDA on Linux when present, else CPU).
enum class ReIdDevice {
    kAuto,
    kCpu,
    kCuda,
    kCoreMl,
};

// Generic ONNX ReID inference shared by all native trackers.
//
// The model has no knowledge of tracker-specific Detection types; callers pass
// already-resolved AABB boxes in image coordinates and receive L2-normalised
// feature vectors back. Per-tracker glue is responsible for converting OBB
// detections to enclosing AABB rects before invoking ``GetFeaturesForBoxes``.
//
// Backend / device selection (in order of priority):
//   1. Explicit ctor arguments.
//   2. Env vars ``BOXMOT_REID_BACKEND`` (opencv|ort|auto) and
//      ``BOXMOT_REID_DEVICE`` (cpu|cuda|coreml|auto).
//   3. ``kAuto`` defaults: prefer ONNX Runtime if compiled in, otherwise
//      OpenCV DNN; on ORT prefer CoreML on macOS, CUDA on Linux if the
//      provider is registered, else CPU.
class OnnxReIdModel {
public:
    OnnxReIdModel(
        fs::path model_path,
        std::string preprocess_name = "resize_pad",
        ReIdBackend backend = ReIdBackend::kAuto,
        ReIdDevice device = ReIdDevice::kAuto
    );
    ~OnnxReIdModel();

    OnnxReIdModel(const OnnxReIdModel&) = delete;
    OnnxReIdModel& operator=(const OnnxReIdModel&) = delete;
    OnnxReIdModel(OnnxReIdModel&&) noexcept;
    OnnxReIdModel& operator=(OnnxReIdModel&&) noexcept;

    [[nodiscard]] bool valid() const { return initialized_; }
    [[nodiscard]] const fs::path& model_path() const { return model_path_; }
    [[nodiscard]] const std::string& preprocess_name() const { return preprocess_name_; }
    [[nodiscard]] cv::Size input_size() const { return input_size_; }
    [[nodiscard]] ReIdBackend backend() const { return backend_; }
    [[nodiscard]] ReIdDevice device() const { return device_; }

    std::vector<Eigen::VectorXf> GetFeaturesForBoxes(
        const std::vector<cv::Rect>& boxes,
        const cv::Mat& image
    ) const;

    // Oriented variant: ``boxes`` are ``(cx, cy, w, h, theta_rad)``. Each crop
    // is extracted by warping the rotated rectangle so its long/short axes
    // align with the destination (axis-aligned) crop, eliminating background
    // pixels that an enclosing-AABB crop would include.
    std::vector<Eigen::VectorXf> GetFeaturesForObbBoxes(
        const std::vector<Eigen::Matrix<double, 5, 1>>& boxes,
        const cv::Mat& image
    ) const;

private:
    cv::Mat PreprocessCrop(const cv::Mat& crop) const;
    cv::Mat ExtractCrop(const cv::Rect& box, const cv::Mat& image) const;
    cv::Mat ExtractObbCrop(const Eigen::Matrix<double, 5, 1>& xywha, const cv::Mat& image) const;
    cv::Mat BuildInputBlob(const std::vector<cv::Mat>& processed_crops) const;
    static Eigen::VectorXf NormalizeFeature(const float* data, int size);
    static bool LooksLikeLmbnModel(const fs::path& model_path);
    static cv::Mat ResizePad(const cv::Mat& crop, const cv::Size& target_size);

    Eigen::VectorXf RunOpenCv(const cv::Mat& processed_crop) const;
    Eigen::VectorXf RunOrt(const cv::Mat& processed_crop) const;

    fs::path model_path_;
    std::string preprocess_name_;
    cv::Size input_size_;
    cv::Scalar mean_;
    cv::Scalar std_;
    ReIdBackend backend_ = ReIdBackend::kAuto;
    ReIdDevice device_ = ReIdDevice::kAuto;
    mutable cv::dnn::Net net_;

    // Opaque ORT session pimpl so callers don't need ORT headers transitively.
    struct OrtSession;
    std::unique_ptr<OrtSession> ort_;

    bool initialized_ = false;
};

// Convenience factory: returns ``std::nullopt`` when ``model_path`` is empty
// so callers can use it from ``std::optional<OnnxReIdModel>`` fields.
std::optional<OnnxReIdModel> MaybeCreateOnnxReIdModel(
    const fs::path& model_path,
    const std::string& preprocess_name = "resize_pad",
    ReIdBackend backend = ReIdBackend::kAuto,
    ReIdDevice device = ReIdDevice::kAuto
);

// Compute the enclosing AABB of an oriented bounding box ``(cx, cy, w, h, theta)``.
Eigen::Vector4d ObbToEnclosingXyxy(const Eigen::Matrix<double, 5, 1>& xywha);

// Clamp an ``[x1, y1, x2, y2]`` rect into the bounds of ``image_size``.
cv::Rect ClampBoxToImage(const Eigen::Vector4d& xyxy, const cv::Size& image_size);

// Generic per-detection ReID dispatch shared by all native trackers.
//
// ``Detection`` must expose ``bool is_obb``, ``Eigen::Vector4d xyxy`` and
// ``Eigen::Matrix<double, 5, 1> xywha``. AABB rows are clamped to the image
// then routed through ``GetFeaturesForBoxes``; OBB rows are warped to a
// straightened axis-aligned crop via ``GetFeaturesForObbBoxes``. Original
// ordering is preserved in the returned vector.
template <typename Detection>
std::vector<Eigen::VectorXf> GetReIdFeaturesForDetections(
    const OnnxReIdModel& model,
    const std::vector<Detection>& detections,
    const cv::Mat& image
) {
    std::vector<cv::Rect> aabb_boxes;
    std::vector<std::size_t> aabb_idx;
    std::vector<Eigen::Matrix<double, 5, 1>> obb_boxes;
    std::vector<std::size_t> obb_idx;
    aabb_boxes.reserve(detections.size());
    obb_boxes.reserve(detections.size());
    const cv::Size image_size = image.size();
    for (std::size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        if (det.is_obb) {
            obb_boxes.push_back(det.xywha);
            obb_idx.push_back(i);
        } else {
            aabb_boxes.push_back(ClampBoxToImage(det.xyxy, image_size));
            aabb_idx.push_back(i);
        }
    }

    std::vector<Eigen::VectorXf> features(detections.size());
    if (!aabb_boxes.empty()) {
        const auto aabb_features = model.GetFeaturesForBoxes(aabb_boxes, image);
        for (std::size_t k = 0; k < aabb_idx.size(); ++k) {
            features[aabb_idx[k]] = aabb_features[k];
        }
    }
    if (!obb_boxes.empty()) {
        const auto obb_features = model.GetFeaturesForObbBoxes(obb_boxes, image);
        for (std::size_t k = 0; k < obb_idx.size(); ++k) {
            features[obb_idx[k]] = obb_features[k];
        }
    }
    return features;
}

}  // namespace boxmot::trackers::base
