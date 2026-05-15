#pragma once
// TensorRT-accelerated ReID inference backend for the BoxMOT DeepStream adapter.
//
// Mirrors DeepStream's NvMultiObjectTracker ReID pipeline:
//   1. Crop objects from the frame (GPU-accelerated via CUDA)
//   2. Resize to network input dimensions
//   3. Apply color format conversion + normalization: y = netScaleFactor * (x - offsets)
//   4. Run TensorRT engine in batched mode
//   5. L2-normalize output feature vectors
//
// This implementation uses the same preprocessing conventions as DeepStream's
// ReID module so models trained for/used with NvDeepSORT can be directly reused.

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace boxmot::deepstream {

namespace fs = std::filesystem;

struct TensorRTReIdConfig {
    // Model paths
    std::string onnx_file;
    std::string tlt_encoded_model;
    std::string tlt_model_key;
    std::string model_engine_file;
    std::string calibration_table_file;

    // Network configuration
    int batch_size = 100;
    int network_mode = 1;  // 0=FP32, 1=FP16, 2=INT8
    int workspace_size = 20;  // MB
    std::vector<int> infer_dims = {128, 64, 3};  // H, W, C or C, H, W based on inputOrder
    int input_order = 1;  // 0=NCHW, 1=NHWC
    int color_format = 0;  // 0=RGB, 1=BGR

    // Preprocessing
    float net_scale_factor = 1.0f;
    std::vector<float> offsets = {0.0f, 0.0f, 0.0f};

    // Feature output
    int reid_feature_size = 256;
    int reid_history_size = 100;
    bool add_feature_normalization = true;
    bool keep_aspect_ratio = true;

    // Derived dimensions (computed from infer_dims + input_order)
    int InputHeight() const {
        return (input_order == 0) ? infer_dims[1] : infer_dims[0];
    }
    int InputWidth() const {
        return (input_order == 0) ? infer_dims[2] : infer_dims[1];
    }
    int InputChannels() const {
        return (input_order == 0) ? infer_dims[0] : infer_dims[2];
    }
};

// Custom TensorRT logger
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
    void SetVerbosity(Severity level) { verbosity_ = level; }
private:
    Severity verbosity_ = Severity::kWARNING;
};

// RAII wrapper for CUDA memory
struct CudaBuffer {
    void* ptr = nullptr;
    size_t size = 0;

    CudaBuffer() = default;
    explicit CudaBuffer(size_t bytes);
    ~CudaBuffer();
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;
};

// TensorRT-accelerated ReID model.
//
// Lifecycle:
//   1. Construct with config → builds/loads TensorRT engine
//   2. Call ExtractFeatures() with cropped object images
//   3. Returns L2-normalized feature vectors
class TensorRTReIdModel {
public:
    explicit TensorRTReIdModel(const TensorRTReIdConfig& config);
    ~TensorRTReIdModel();

    TensorRTReIdModel(const TensorRTReIdModel&) = delete;
    TensorRTReIdModel& operator=(const TensorRTReIdModel&) = delete;

    // Extract ReID features for a batch of object crops (already resized to
    // network input dimensions). Returns N feature vectors, each of size
    // reid_feature_size.
    std::vector<std::vector<float>> ExtractFeatures(
        const std::vector<cv::Mat>& crops
    );

    // Extract ReID features directly from bounding boxes in a frame.
    // Handles cropping, resizing, and preprocessing internally.
    std::vector<std::vector<float>> ExtractFeaturesFromFrame(
        const cv::Mat& frame,
        const std::vector<cv::Rect>& boxes
    );

    [[nodiscard]] bool IsValid() const { return engine_ != nullptr; }
    [[nodiscard]] int FeatureSize() const { return config_.reid_feature_size; }
    [[nodiscard]] int BatchSize() const { return config_.batch_size; }
    [[nodiscard]] const TensorRTReIdConfig& Config() const { return config_; }

private:
    bool BuildEngine();
    bool LoadEngine(const std::string& engine_path);
    bool SaveEngine(const std::string& engine_path);

    // Preprocess a single crop: resize + color convert + normalize
    cv::Mat PreprocessCrop(const cv::Mat& crop) const;

    // Run inference on a batch (up to batch_size crops)
    std::vector<std::vector<float>> RunBatch(const std::vector<cv::Mat>& preprocessed);

    // L2 normalize a feature vector in-place
    static void L2Normalize(std::vector<float>& feature);

    TensorRTReIdConfig config_;
    TrtLogger logger_;

    // TensorRT objects
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // GPU buffers
    CudaBuffer input_buffer_;
    CudaBuffer output_buffer_;

    // CUDA stream for async operations
    cudaStream_t stream_ = nullptr;

    // Input/output tensor indices
    int input_index_ = -1;
    int output_index_ = -1;
};

}  // namespace boxmot::deepstream
