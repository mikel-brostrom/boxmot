// Inference backend strategy implementations for ``OnnxReIdModel``.

#include "boxmot/trackers/base/reid_inference_backend.hpp"
#include "boxmot/trackers/base/reid_onnx.hpp"

#include <opencv2/dnn.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

#if defined(BOXMOT_HAS_ONNXRUNTIME)
#include <onnxruntime_cxx_api.h>
#if defined(__APPLE__)
#include <coreml_provider_factory.h>
#endif
#endif

namespace boxmot::trackers::base {

namespace {

const char* BackendName(ReIdBackend backend) {
    switch (backend) {
        case ReIdBackend::kOpenCvDnn: return "opencv_dnn";
        case ReIdBackend::kOnnxRuntime: return "onnxruntime";
        default: return "auto";
    }
}

const char* DeviceName(ReIdDevice device) {
    switch (device) {
        case ReIdDevice::kCpu: return "cpu";
        case ReIdDevice::kCuda: return "cuda";
        case ReIdDevice::kCoreMl: return "coreml";
        default: return "auto";
    }
}

// ---------------------------------------------------------------------------
// OpenCV DNN backend (CPU only)
// ---------------------------------------------------------------------------

class OpenCvDnnInferenceBackend final : public ReIdInferenceBackend {
public:
    explicit OpenCvDnnInferenceBackend(const fs::path& model_path)
        : net_(cv::dnn::readNetFromONNX(model_path.string())) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    std::vector<float> Forward(const cv::Mat& blob) const override {
        net_.setInput(blob);
        cv::Mat output = net_.forward();
        cv::Mat reshaped = output.reshape(1, 1);
        if (reshaped.rows != 1) {
            throw std::runtime_error(
                "Native ReID ONNX output shape is incompatible with single-crop inference.");
        }
        const float* data = reshaped.ptr<float>(0);
        return std::vector<float>(data, data + reshaped.cols);
    }

    ReIdBackend kind() const override { return ReIdBackend::kOpenCvDnn; }
    ReIdDevice device() const override { return ReIdDevice::kCpu; }

private:
    mutable cv::dnn::Net net_;
};

// ---------------------------------------------------------------------------
// ONNX Runtime backend (CPU / CUDA / CoreML)
// ---------------------------------------------------------------------------

#if defined(BOXMOT_HAS_ONNXRUNTIME)

class OnnxRuntimeInferenceBackend final : public ReIdInferenceBackend {
public:
    OnnxRuntimeInferenceBackend(
        const fs::path& model_path,
        ReIdDevice requested_device,
        const cv::Size& input_size
    )
        : env_(ORT_LOGGING_LEVEL_WARNING, "boxmot_reid"),
          input_shape_{1, 3,
                       static_cast<int64_t>(input_size.height),
                       static_cast<int64_t>(input_size.width)} {
        options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options_.SetIntraOpNumThreads(1);

        ReIdDevice resolved = requested_device;
        if (resolved == ReIdDevice::kCoreMl) {
#if defined(__APPLE__)
            uint32_t coreml_flags = 0;
            const OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CoreML(
                static_cast<OrtSessionOptions*>(options_), coreml_flags);
            if (status != nullptr) {
                Ort::GetApi().ReleaseStatus(const_cast<OrtStatus*>(status));
                resolved = ReIdDevice::kCpu;
            }
#else
            resolved = ReIdDevice::kCpu;
#endif
        } else if (resolved == ReIdDevice::kCuda) {
            try {
                OrtCUDAProviderOptions cuda_opts{};
                options_.AppendExecutionProvider_CUDA(cuda_opts);
            } catch (const Ort::Exception&) {
                resolved = ReIdDevice::kCpu;
            }
        }
        resolved_device_ = resolved;

        session_ = std::make_unique<Ort::Session>(
            env_, model_path.string().c_str(), options_);

        Ort::AllocatedStringPtr in_name = session_->GetInputNameAllocated(0, allocator_);
        Ort::AllocatedStringPtr out_name = session_->GetOutputNameAllocated(0, allocator_);
        input_name_ = in_name.get();
        output_name_ = out_name.get();
    }

    std::vector<float> Forward(const cv::Mat& blob) const override {
        const size_t element_count = static_cast<size_t>(input_shape_[2]) *
                                     static_cast<size_t>(input_shape_[3]) * 3UL;

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            reinterpret_cast<float*>(blob.data),
            element_count,
            input_shape_.data(),
            input_shape_.size()
        );

        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1
        );

        Ort::Value& output = output_tensors.front();
        const auto type_info = output.GetTensorTypeAndShapeInfo();
        const size_t feature_dim = type_info.GetElementCount();
        const float* data = output.GetTensorData<float>();
        return std::vector<float>(data, data + feature_dim);
    }

    ReIdBackend kind() const override { return ReIdBackend::kOnnxRuntime; }
    ReIdDevice device() const override { return resolved_device_; }

private:
    Ort::Env env_;
    Ort::SessionOptions options_{};
    Ort::AllocatorWithDefaultOptions allocator_{};
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
    std::array<int64_t, 4> input_shape_;
    ReIdDevice resolved_device_ = ReIdDevice::kCpu;
};

#endif  // BOXMOT_HAS_ONNXRUNTIME

}  // namespace

std::unique_ptr<ReIdInferenceBackend> MakeReIdInferenceBackend(
    const fs::path& model_path,
    ReIdBackend requested_backend,
    ReIdDevice requested_device,
    const cv::Size& input_size
) {
    std::unique_ptr<ReIdInferenceBackend> backend;

    if (requested_backend == ReIdBackend::kOnnxRuntime) {
#if defined(BOXMOT_HAS_ONNXRUNTIME)
        backend = std::make_unique<OnnxRuntimeInferenceBackend>(
            model_path, requested_device, input_size);
#else
        // ORT not compiled in: fall back to OpenCV DNN.
        (void)requested_device;
        backend = std::make_unique<OpenCvDnnInferenceBackend>(model_path);
#endif
    } else {
        (void)input_size;
        (void)requested_device;
        backend = std::make_unique<OpenCvDnnInferenceBackend>(model_path);
    }

    if (backend) {
        std::cerr << "[boxmot] native ReID inference backend=" << BackendName(backend->kind())
                  << " device=" << DeviceName(backend->device())
                  << " model=" << model_path.filename().string() << '\n';
    }
    return backend;
}

}  // namespace boxmot::trackers::base
