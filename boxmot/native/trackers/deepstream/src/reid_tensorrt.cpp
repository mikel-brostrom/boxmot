// TensorRT-accelerated ReID inference implementation.
//
// This mirrors DeepStream's NvMultiObjectTracker ReID pipeline exactly:
//   - Crop each detection from the frame
//   - Resize to inferDims (preserving aspect ratio if keepAspc=1)
//   - Convert color format (RGB/BGR)
//   - Normalize: y = netScaleFactor * (x - offsets)
//   - Transpose to inputOrder (NCHW/NHWC)
//   - Run batched TensorRT inference
//   - L2-normalize output embeddings

#include "deepstream/reid_tensorrt.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace boxmot::deepstream {

// ---------------------------------------------------------------------------
// TrtLogger
// ---------------------------------------------------------------------------
void TrtLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= verbosity_) {
        const char* prefix = "";
        switch (severity) {
            case Severity::kINTERNAL_ERROR: prefix = "[FATAL] "; break;
            case Severity::kERROR: prefix = "[ERROR] "; break;
            case Severity::kWARNING: prefix = "[WARN]  "; break;
            case Severity::kINFO: prefix = "[INFO]  "; break;
            case Severity::kVERBOSE: prefix = "[DEBUG] "; break;
        }
        std::cerr << "[BoxMOT-DS-ReID] " << prefix << msg << std::endl;
    }
}

// ---------------------------------------------------------------------------
// CudaBuffer
// ---------------------------------------------------------------------------
CudaBuffer::CudaBuffer(size_t bytes) : size(bytes) {
    if (bytes > 0) {
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDA malloc failed: ") + cudaGetErrorString(err));
        }
    }
}

CudaBuffer::~CudaBuffer() {
    if (ptr) {
        cudaFree(ptr);
    }
}

CudaBuffer::CudaBuffer(CudaBuffer&& other) noexcept
    : ptr(other.ptr), size(other.size) {
    other.ptr = nullptr;
    other.size = 0;
}

CudaBuffer& CudaBuffer::operator=(CudaBuffer&& other) noexcept {
    if (this != &other) {
        if (ptr) cudaFree(ptr);
        ptr = other.ptr;
        size = other.size;
        other.ptr = nullptr;
        other.size = 0;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// TensorRTReIdModel
// ---------------------------------------------------------------------------

TensorRTReIdModel::TensorRTReIdModel(const TensorRTReIdConfig& config)
    : config_(config) {
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to create CUDA stream: ") + cudaGetErrorString(err));
    }

    // Try to load a pre-built engine first
    if (!config_.model_engine_file.empty()) {
        if (LoadEngine(config_.model_engine_file)) {
            return;
        }
    }

    // Build engine from ONNX or TLT model
    if (!BuildEngine()) {
        throw std::runtime_error("Failed to build TensorRT engine for ReID model.");
    }

    // Save the built engine for future use
    if (!config_.model_engine_file.empty()) {
        SaveEngine(config_.model_engine_file);
    }
}

TensorRTReIdModel::~TensorRTReIdModel() {
    context_.reset();
    engine_.reset();
    runtime_.reset();
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

bool TensorRTReIdModel::BuildEngine() {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger_));
    if (!builder) return false;

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(
            1U << static_cast<uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    if (!network) return false;

    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger_));
    if (!parser) return false;

    // Parse the ONNX model
    std::string model_path;
    if (!config_.onnx_file.empty()) {
        model_path = config_.onnx_file;
    } else if (!config_.tlt_encoded_model.empty()) {
        // For TLT models, we'd need to decrypt first. For now, support ONNX directly.
        // DeepStream's TLT support requires the NVIDIA TAO toolkit.
        std::cerr << "[BoxMOT-DS-ReID] TLT model support requires conversion to ONNX. "
                  << "Please convert using tao-converter or provide an ONNX file.\n";
        return false;
    } else {
        std::cerr << "[BoxMOT-DS-ReID] No model file specified.\n";
        return false;
    }

    if (!parser->parseFromFile(model_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "[BoxMOT-DS-ReID] Failed to parse ONNX model: " << model_path << "\n";
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "  " << parser->getError(i)->desc() << "\n";
        }
        return false;
    }

    // Configure builder
    auto build_config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    if (!build_config) return false;

    build_config->setMemoryPoolLimit(
        nvinfer1::MemoryPoolType::kWORKSPACE,
        static_cast<size_t>(config_.workspace_size) * (1ULL << 20));

    // Set precision mode matching DeepStream's networkMode
    switch (config_.network_mode) {
        case 1:  // FP16
            if (builder->platformHasFastFp16()) {
                build_config->setFlag(nvinfer1::BuilderFlag::kFP16);
            }
            break;
        case 2:  // INT8
            if (builder->platformHasFastInt8()) {
                build_config->setFlag(nvinfer1::BuilderFlag::kINT8);
                // INT8 calibration would be handled here with calibration_table_file
                // For now we fall back to FP16 if no calibration table
                if (config_.calibration_table_file.empty()) {
                    if (builder->platformHasFastFp16()) {
                        build_config->setFlag(nvinfer1::BuilderFlag::kFP16);
                    }
                }
            }
            break;
        default:  // FP32
            break;
    }

    // Set dynamic batch size via optimization profile
    auto profile = builder->createOptimizationProfile();
    auto input = network->getInput(0);
    auto input_dims = input->getDimensions();

    // Override batch dimension
    nvinfer1::Dims4 min_dims(1, input_dims.d[1], input_dims.d[2], input_dims.d[3]);
    nvinfer1::Dims4 opt_dims(config_.batch_size / 2,
                             input_dims.d[1], input_dims.d[2], input_dims.d[3]);
    nvinfer1::Dims4 max_dims(config_.batch_size,
                             input_dims.d[1], input_dims.d[2], input_dims.d[3]);

    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, min_dims);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, max_dims);
    build_config->addOptimizationProfile(profile);

    // Build the engine
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *build_config));
    if (!serialized) {
        std::cerr << "[BoxMOT-DS-ReID] Failed to build serialized network.\n";
        return false;
    }

    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(logger_));
    if (!runtime_) return false;

    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(serialized->data(), serialized->size()));
    if (!engine_) return false;

    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext());
    if (!context_) return false;

    // Resolve tensor indices
    input_index_ = -1;
    output_index_ = -1;
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_index_ = i;
        } else {
            output_index_ = i;
        }
    }

    // Allocate GPU buffers for max batch
    const int h = config_.InputHeight();
    const int w = config_.InputWidth();
    const int c = config_.InputChannels();
    const size_t input_size = static_cast<size_t>(config_.batch_size) * c * h * w * sizeof(float);
    const size_t output_size = static_cast<size_t>(config_.batch_size) *
                               config_.reid_feature_size * sizeof(float);

    input_buffer_ = CudaBuffer(input_size);
    output_buffer_ = CudaBuffer(output_size);

    std::cerr << "[BoxMOT-DS-ReID] TensorRT engine built successfully. "
              << "Input: " << config_.batch_size << "x" << c << "x" << h << "x" << w
              << ", Feature size: " << config_.reid_feature_size << "\n";

    return true;
}

bool TensorRTReIdModel::LoadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return false;

    const auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<size_t>(size));
    if (!file.read(buffer.data(), size)) return false;

    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(logger_));
    if (!runtime_) return false;

    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!engine_) return false;

    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext());
    if (!context_) return false;

    // Resolve tensor indices
    input_index_ = -1;
    output_index_ = -1;
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_index_ = i;
        } else {
            output_index_ = i;
        }
    }

    // Allocate GPU buffers
    const int h = config_.InputHeight();
    const int w = config_.InputWidth();
    const int c = config_.InputChannels();
    const size_t input_size = static_cast<size_t>(config_.batch_size) * c * h * w * sizeof(float);
    const size_t output_size = static_cast<size_t>(config_.batch_size) *
                               config_.reid_feature_size * sizeof(float);

    input_buffer_ = CudaBuffer(input_size);
    output_buffer_ = CudaBuffer(output_size);

    std::cerr << "[BoxMOT-DS-ReID] Loaded TensorRT engine from: " << engine_path << "\n";
    return true;
}

bool TensorRTReIdModel::SaveEngine(const std::string& engine_path) {
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        engine_->serialize());
    if (!serialized) return false;

    std::ofstream file(engine_path, std::ios::binary);
    if (!file.is_open()) return false;

    file.write(static_cast<const char*>(serialized->data()),
               static_cast<std::streamsize>(serialized->size()));
    return file.good();
}

cv::Mat TensorRTReIdModel::PreprocessCrop(const cv::Mat& crop) const {
    if (crop.empty()) {
        return cv::Mat::zeros(config_.InputHeight(), config_.InputWidth(), CV_32FC3);
    }

    const int target_h = config_.InputHeight();
    const int target_w = config_.InputWidth();

    cv::Mat resized;
    if (config_.keep_aspect_ratio) {
        // Resize preserving aspect ratio, pad with zeros (matching DeepStream's keepAspc)
        const float scale = std::min(
            static_cast<float>(target_w) / static_cast<float>(crop.cols),
            static_cast<float>(target_h) / static_cast<float>(crop.rows));
        const int new_w = static_cast<int>(crop.cols * scale);
        const int new_h = static_cast<int>(crop.rows * scale);

        cv::Mat scaled;
        cv::resize(crop, scaled, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

        resized = cv::Mat::zeros(target_h, target_w, crop.type());
        const int dx = (target_w - new_w) / 2;
        const int dy = (target_h - new_h) / 2;
        scaled.copyTo(resized(cv::Rect(dx, dy, new_w, new_h)));
    } else {
        cv::resize(crop, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);
    }

    // Color format conversion (input is BGR from OpenCV)
    cv::Mat color_converted;
    if (config_.color_format == 0) {  // RGB
        cv::cvtColor(resized, color_converted, cv::COLOR_BGR2RGB);
    } else {  // BGR (already correct)
        color_converted = resized;
    }

    // Convert to float and normalize: y = netScaleFactor * (x - offsets)
    cv::Mat float_img;
    color_converted.convertTo(float_img, CV_32F);

    // Apply offset subtraction and scale (matching DeepStream's preprocessing)
    std::vector<cv::Mat> channels;
    cv::split(float_img, channels);
    for (int c = 0; c < 3; ++c) {
        float offset = (c < static_cast<int>(config_.offsets.size())) ? config_.offsets[c] : 0.0f;
        channels[c] = (channels[c] - offset) * config_.net_scale_factor;
    }
    cv::merge(channels, float_img);

    // Transpose to NCHW if needed (DeepStream default for most ReID models)
    if (config_.input_order == 0) {  // NCHW
        // Create CHW blob
        cv::Mat blob;
        cv::dnn::blobFromImage(float_img, blob, 1.0, cv::Size(), cv::Scalar(), false, false);
        // blobFromImage returns (1, C, H, W), reshape to (C, H, W)
        return blob.reshape(1, {3, target_h, target_w});
    }

    // NHWC - already in correct layout
    return float_img;
}

std::vector<std::vector<float>> TensorRTReIdModel::RunBatch(
    const std::vector<cv::Mat>& preprocessed
) {
    if (preprocessed.empty() || !engine_ || !context_) {
        return {};
    }

    const int batch = static_cast<int>(preprocessed.size());
    const int h = config_.InputHeight();
    const int w = config_.InputWidth();
    const int c = config_.InputChannels();
    const size_t per_image_floats = static_cast<size_t>(c * h * w);
    const size_t per_image_bytes = per_image_floats * sizeof(float);

    // Assemble batch into contiguous host memory
    std::vector<float> host_input(static_cast<size_t>(batch) * per_image_floats);
    for (int i = 0; i < batch; ++i) {
        const cv::Mat& img = preprocessed[i];
        if (img.total() * img.channels() != per_image_floats) {
            // Fill with zeros if shape mismatch
            std::memset(host_input.data() + i * per_image_floats, 0, per_image_bytes);
            continue;
        }
        std::memcpy(host_input.data() + i * per_image_floats,
                    img.ptr<float>(),
                    per_image_bytes);
    }

    // Copy input to GPU
    cudaMemcpyAsync(input_buffer_.ptr, host_input.data(),
                    static_cast<size_t>(batch) * per_image_bytes,
                    cudaMemcpyHostToDevice, stream_);

    // Set input shape for dynamic batch
    const char* input_name = engine_->getIOTensorName(input_index_);
    const char* output_name = engine_->getIOTensorName(output_index_);

    nvinfer1::Dims4 input_dims(batch, c, h, w);
    context_->setInputShape(input_name, input_dims);

    // Set tensor addresses
    context_->setTensorAddress(input_name, input_buffer_.ptr);
    context_->setTensorAddress(output_name, output_buffer_.ptr);

    // Execute inference
    bool success = context_->enqueueV3(stream_);
    if (!success) {
        std::cerr << "[BoxMOT-DS-ReID] TensorRT inference failed.\n";
        return {};
    }

    // Copy output back to host
    const size_t output_bytes = static_cast<size_t>(batch) *
                                config_.reid_feature_size * sizeof(float);
    std::vector<float> host_output(static_cast<size_t>(batch) * config_.reid_feature_size);
    cudaMemcpyAsync(host_output.data(), output_buffer_.ptr,
                    output_bytes, cudaMemcpyDeviceToHost, stream_);

    cudaStreamSynchronize(stream_);

    // Split into per-object feature vectors
    std::vector<std::vector<float>> features(batch);
    for (int i = 0; i < batch; ++i) {
        features[i].assign(
            host_output.begin() + i * config_.reid_feature_size,
            host_output.begin() + (i + 1) * config_.reid_feature_size);

        if (config_.add_feature_normalization) {
            L2Normalize(features[i]);
        }
    }

    return features;
}

void TensorRTReIdModel::L2Normalize(std::vector<float>& feature) {
    float norm = 0.0f;
    for (float val : feature) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    if (norm > 1e-10f) {
        for (float& val : feature) {
            val /= norm;
        }
    }
}

std::vector<std::vector<float>> TensorRTReIdModel::ExtractFeatures(
    const std::vector<cv::Mat>& crops
) {
    if (crops.empty()) return {};

    std::vector<std::vector<float>> all_features;
    all_features.reserve(crops.size());

    // Process in batches of batch_size
    for (size_t start = 0; start < crops.size(); start += config_.batch_size) {
        const size_t end = std::min(start + static_cast<size_t>(config_.batch_size),
                                    crops.size());

        std::vector<cv::Mat> batch_preprocessed;
        batch_preprocessed.reserve(end - start);
        for (size_t i = start; i < end; ++i) {
            batch_preprocessed.push_back(PreprocessCrop(crops[i]));
        }

        auto batch_features = RunBatch(batch_preprocessed);
        all_features.insert(all_features.end(),
                           batch_features.begin(), batch_features.end());
    }

    return all_features;
}

std::vector<std::vector<float>> TensorRTReIdModel::ExtractFeaturesFromFrame(
    const cv::Mat& frame,
    const std::vector<cv::Rect>& boxes
) {
    if (boxes.empty() || frame.empty()) return {};

    // Crop objects from the frame
    std::vector<cv::Mat> crops;
    crops.reserve(boxes.size());
    for (const auto& box : boxes) {
        // Clamp box to frame bounds
        int x1 = std::max(0, box.x);
        int y1 = std::max(0, box.y);
        int x2 = std::min(frame.cols, box.x + box.width);
        int y2 = std::min(frame.rows, box.y + box.height);

        if (x2 <= x1 || y2 <= y1) {
            crops.emplace_back();  // empty mat
        } else {
            crops.push_back(frame(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone());
        }
    }

    return ExtractFeatures(crops);
}

}  // namespace boxmot::deepstream
