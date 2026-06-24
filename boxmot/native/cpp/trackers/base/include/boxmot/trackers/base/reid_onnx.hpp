#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <chrono>
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

class ReIdInferenceBackend;

// Generic ONNX ReID inference shared by all native trackers.
//
// Pipeline mirrors the Python ``BaseModelBackend`` surface so timing
// instrumentation can attribute work to the same three buckets:
//
//   1. ``Preprocess(boxes, image)``  → crop + resize + standardize + blob assembly
//   2. ``Process(blob)``             → backend-dispatched model forward pass
//   3. ``Postprocess(raw_features)`` → reshape into rows + L2 normalization
//
// ``GetFeaturesForBoxes`` remains the single-shot composite for callers that
// don't care about per-stage timing.
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
    // Intermediate buffers exchanged between the staged methods. Kept opaque
    // so callers don't need to know the layout (single-blob N×3×H×W float32
    // for crops, row-major N×feature_dim float32 for raw features).
    struct CropBatch {
        cv::Mat blob;          // CV_32F, dims = [N, 3, H, W]
        std::size_t count = 0; // logical N (covers the empty case)
    };

    struct RawFeatures {
        std::vector<float> data; // size == count * feature_dim
        std::size_t count = 0;
        std::size_t feature_dim = 0;
    };

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

    // Stage 1 (AABB): build the (N, 3, H, W) preprocessed crop blob for a
    // list of axis-aligned boxes against ``image``. Boxes that fall outside
    // the image are replaced by zero-filled crops so the output count always
    // equals the input count.
    CropBatch Preprocess(const std::vector<cv::Rect>& boxes, const cv::Mat& image) const;

    // Stage 1 (OBB): same contract as ``Preprocess`` but each crop is warped
    // from the rotated rectangle ``(cx, cy, w, h, theta_rad)`` so its
    // long/short axes align with the destination crop, eliminating
    // background pixels that an enclosing-AABB crop would include.
    CropBatch PreprocessObb(
        const std::vector<Eigen::Matrix<double, 5, 1>>& boxes,
        const cv::Mat& image
    ) const;

    // Stage 2: run the model forward pass on each crop, returning raw
    // (un-normalised) features as a flat row-major buffer of shape
    // ``(count, feature_dim)``. Empty batches return an empty buffer.
    RawFeatures Process(const CropBatch& crops) const;

    // Stage 3: convert raw features into a vector of L2-normalised
    // ``Eigen::VectorXf`` rows. Mirrors the legacy single-shot output type.
    std::vector<Eigen::VectorXf> Postprocess(const RawFeatures& raw) const;

    // Composite shortcuts: ``Preprocess`` → ``Process`` → ``Postprocess``.
    std::vector<Eigen::VectorXf> GetFeaturesForBoxes(
        const std::vector<cv::Rect>& boxes,
        const cv::Mat& image
    ) const;

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

    fs::path model_path_;
    std::string preprocess_name_;
    cv::Size input_size_;
    cv::Scalar mean_;
    cv::Scalar std_;
    ReIdBackend backend_ = ReIdBackend::kAuto;
    ReIdDevice device_ = ReIdDevice::kAuto;

    std::unique_ptr<ReIdInferenceBackend> inference_;

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

// Bundle of ReID features plus per-phase wall-clock timings (milliseconds)
// produced by ``GetReIdFeaturesForDetections``. The trackers fan these out
// through their C ABI so Python can attribute work to the same three buckets
// as the pure-Python ReID runtime.
struct TimedReIdFeatures {
    std::vector<Eigen::VectorXf> features;
    double preprocess_ms = 0.0;
    double process_ms = 0.0;
    double postprocess_ms = 0.0;
};

// Generic per-detection ReID dispatch shared by all native trackers.
//
// ``Detection`` must expose ``bool is_obb``, ``Eigen::Vector4d xyxy`` and
// ``Eigen::Matrix<double, 5, 1> xywha``. AABB rows are clamped to the image
// then routed through ``Preprocess``; OBB rows are warped to a straightened
// axis-aligned crop via ``PreprocessObb``. Raw features from both groups
// are interleaved into a single ``Process``/``Postprocess`` invocation so
// per-phase timing reflects the work actually performed by the model.
// Original detection ordering is preserved in the returned features.
template <typename Detection>
TimedReIdFeatures GetReIdFeaturesForDetections(
    const OnnxReIdModel& model,
    const std::vector<Detection>& detections,
    const cv::Mat& image
) {
    using clock = std::chrono::steady_clock;
    auto ms = [](clock::time_point start, clock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    TimedReIdFeatures result;
    if (detections.empty()) {
        return result;
    }

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

    result.features.assign(detections.size(), Eigen::VectorXf{});

    auto run_aabb = [&](const std::vector<cv::Rect>& boxes, const std::vector<std::size_t>& indices) {
        if (boxes.empty()) {
            return;
        }
        auto t0 = clock::now();
        OnnxReIdModel::CropBatch crops = model.Preprocess(boxes, image);
        auto t1 = clock::now();
        OnnxReIdModel::RawFeatures raw = model.Process(crops);
        auto t2 = clock::now();
        std::vector<Eigen::VectorXf> features = model.Postprocess(raw);
        auto t3 = clock::now();

        result.preprocess_ms += ms(t0, t1);
        result.process_ms += ms(t1, t2);
        result.postprocess_ms += ms(t2, t3);

        for (std::size_t k = 0; k < indices.size(); ++k) {
            result.features[indices[k]] = features[k];
        }
    };

    auto run_obb = [&](const std::vector<Eigen::Matrix<double, 5, 1>>& boxes, const std::vector<std::size_t>& indices) {
        if (boxes.empty()) {
            return;
        }
        auto t0 = clock::now();
        OnnxReIdModel::CropBatch crops = model.PreprocessObb(boxes, image);
        auto t1 = clock::now();
        OnnxReIdModel::RawFeatures raw = model.Process(crops);
        auto t2 = clock::now();
        std::vector<Eigen::VectorXf> features = model.Postprocess(raw);
        auto t3 = clock::now();

        result.preprocess_ms += ms(t0, t1);
        result.process_ms += ms(t1, t2);
        result.postprocess_ms += ms(t2, t3);

        for (std::size_t k = 0; k < indices.size(); ++k) {
            result.features[indices[k]] = features[k];
        }
    };

    run_aabb(aabb_boxes, aabb_idx);
    run_obb(obb_boxes, obb_idx);
    return result;
}

}  // namespace boxmot::trackers::base
