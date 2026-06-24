// Strategy interface for the ONNX ReID model's per-crop forward pass.
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

    // Run the model on a single (1, 3, H, W) preprocessed CV_32F blob.
    // Returns the raw feature vector (un-normalized) as a flat float buffer.
    virtual std::vector<float> Forward(const cv::Mat& blob) const = 0;

    [[nodiscard]] virtual ReIdBackend kind() const = 0;
    [[nodiscard]] virtual ReIdDevice device() const = 0;
};

// Build the appropriate inference backend for the given model + preferences.
// ``input_size`` is required so the ORT backend can pre-build its tensor shape.
// Returns ``nullptr`` if the requested backend is unavailable in this build.
std::unique_ptr<ReIdInferenceBackend> MakeReIdInferenceBackend(
    const fs::path& model_path,
    ReIdBackend requested_backend,
    ReIdDevice requested_device,
    const cv::Size& input_size);

}  // namespace boxmot::trackers::base
