// Strategy interface for the ONNX ReID model's forward pass.
//
// ``OnnxReIdModel`` performs detection-driven cropping, mean/std normalization,
// and L2 normalization itself. The actual model forward pass is delegated to
// an implementation of this interface, allowing the host to swap inference
// libraries (OpenCV DNN, ONNX Runtime, ...) and execution providers (CPU,
// CUDA, CoreML, ...) without touching the orchestration code.

#pragma once

#include <opencv2/core.hpp>

#include <filesystem>
#include <memory>
#include <vector>

namespace boxmot::trackers::base {

namespace fs = std::filesystem;

enum class ReIdBackend;
enum class ReIdDevice;

class ReIdInferenceBackend {
public:
    virtual ~ReIdInferenceBackend() = default;

    // Run the model on a preprocessed CV_32F blob shaped (N, 3, H, W).
    // Returns raw features (un-normalized) as a flat row-major float buffer.
    virtual std::vector<float> Forward(const cv::Mat& blob) const = 0;

    [[nodiscard]] virtual ReIdBackend kind() const = 0;
    [[nodiscard]] virtual ReIdDevice device() const = 0;
    // OpenCV DNN remains crop-by-crop for dynamic graphs because some
    // exported ReID heads mishandle N>1. ONNX Runtime supports the graph's
    // runtime N and can execute the complete staged crop batch at once.
    [[nodiscard]] virtual bool supports_dynamic_batch() const { return false; }
};

// Build the appropriate inference backend for the given model + preferences.
// ``input_size`` is required so the ORT backend can pre-build its tensor shape.
// Returns ``nullptr`` if the requested backend is unavailable in this build.
std::unique_ptr<ReIdInferenceBackend> MakeReIdInferenceBackend(
    const fs::path& model_path,
    ReIdBackend requested_backend,
    ReIdDevice requested_device,
    const cv::Size& input_size,
    int input_batch_size);

}  // namespace boxmot::trackers::base
