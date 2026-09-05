// Inference backend strategy implementations for ``OnnxReIdModel``.

#include "boxmot/reid/reid_inference_backend.hpp"
#include "boxmot/reid/reid_onnx.hpp"

#include <opencv2/dnn.hpp>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

#if defined(BOXMOT_HAS_ONNXRUNTIME)
#include <onnxruntime_cxx_api.h>
#if defined(__APPLE__)
#include <coreml_provider_factory.h>
#endif
#endif

namespace boxmot::reid {

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
        const cv::Size& input_size,
        int input_batch_size
    )
        : env_(ORT_LOGGING_LEVEL_WARNING, "boxmot_reid"),
          input_shape_{input_batch_size, 3,
                       static_cast<int64_t>(input_size.height),
                       static_cast<int64_t>(input_size.width)} {
        options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        const bool automatic_device = requested_device == ReIdDevice::kAuto;
        ReIdDevice resolved = requested_device;
        if (automatic_device) {
#if defined(__APPLE__)
            resolved = ReIdDevice::kCoreMl;
#else
            resolved = ReIdDevice::kCuda;
#endif
        }
        if (!automatic_device &&
            (resolved == ReIdDevice::kCuda || resolved == ReIdDevice::kCoreMl)) {
            options_.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
        }
        if (resolved == ReIdDevice::kCoreMl) {
#if defined(__APPLE__)
            uint32_t coreml_flags = 0;
            const OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CoreML(
                static_cast<OrtSessionOptions*>(options_), coreml_flags);
            if (status != nullptr) {
                const std::string detail = Ort::GetApi().GetErrorMessage(status);
                Ort::GetApi().ReleaseStatus(const_cast<OrtStatus*>(status));
                if (!automatic_device) {
                    throw std::runtime_error(
                        "CoreML was explicitly requested for native ReID but is unavailable: " +
                        detail);
                }
                resolved = ReIdDevice::kCpu;
            }
#else
            if (!automatic_device) {
                throw std::runtime_error(
                    "CoreML was explicitly requested for native ReID but is unavailable on this platform.");
            }
            resolved = ReIdDevice::kCpu;
#endif
        } else if (resolved == ReIdDevice::kCuda) {
            try {
                OrtCUDAProviderOptions cuda_opts{};
                options_.AppendExecutionProvider_CUDA(cuda_opts);
            } catch (const Ort::Exception& error) {
                if (!automatic_device) {
                    throw std::runtime_error(
                        "CUDA was explicitly requested for native ReID but is unavailable: " +
                        std::string(error.what()));
                }
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
        if (blob.empty() || blob.dims != 4 || blob.type() != CV_32F ||
            blob.size[0] <= 0 || blob.size[1] != input_shape_[1] ||
            blob.size[2] != input_shape_[2] || blob.size[3] != input_shape_[3] ||
            !blob.isContinuous()) {
            throw std::runtime_error(
                "Native ReID input blob does not match the ONNX input shape.");
        }
        std::array<int64_t, 4> execution_shape = input_shape_;
        if (execution_shape[0] == 0) {
            execution_shape[0] = blob.size[0];
        } else if (execution_shape[0] != blob.size[0]) {
            throw std::runtime_error(
                "Native ReID input blob does not match the fixed ONNX batch size.");
        }
        const size_t element_count = static_cast<size_t>(execution_shape[0]) *
                                     static_cast<size_t>(execution_shape[1]) *
                                     static_cast<size_t>(execution_shape[2]) *
                                     static_cast<size_t>(execution_shape[3]);
        if (blob.total() != element_count) {
            throw std::runtime_error(
                "Native ReID input blob does not match the ONNX input element count.");
        }

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            reinterpret_cast<float*>(blob.data),
            element_count,
            execution_shape.data(),
            execution_shape.size()
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
    bool supports_dynamic_batch() const override { return true; }

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
    const cv::Size& input_size,
    int input_batch_size
) {
    std::unique_ptr<ReIdInferenceBackend> backend;
    ReIdBackend resolved_backend = requested_backend;

    if (resolved_backend == ReIdBackend::kAuto) {
#if defined(BOXMOT_HAS_ONNXRUNTIME)
        resolved_backend = ReIdBackend::kOnnxRuntime;
#else
        resolved_backend = ReIdBackend::kOpenCvDnn;
#endif
    }

    if (resolved_backend == ReIdBackend::kOnnxRuntime) {
#if defined(BOXMOT_HAS_ONNXRUNTIME)
        backend = std::make_unique<OnnxRuntimeInferenceBackend>(
            model_path, requested_device, input_size, input_batch_size);
#else
        (void)requested_device;
        (void)input_size;
        (void)input_batch_size;
        throw std::runtime_error(
            "ONNX Runtime was explicitly requested for native ReID but is unavailable in this build.");
#endif
    } else if (resolved_backend == ReIdBackend::kOpenCvDnn) {
        (void)input_size;
        (void)input_batch_size;
        if (requested_device != ReIdDevice::kAuto && requested_device != ReIdDevice::kCpu) {
            throw std::invalid_argument(
                "OpenCV DNN native ReID supports only device=cpu; the requested accelerator cannot be used.");
        }
        backend = std::make_unique<OpenCvDnnInferenceBackend>(model_path);
    } else {
        throw std::invalid_argument("Unsupported native ReID inference backend selection.");
    }

    if (backend) {
        std::cerr << "[boxmot] native ReID inference backend=" << BackendName(backend->kind())
                  << " device=" << DeviceName(backend->device())
                  << " model=" << model_path.filename().string() << '\n';
    }
    return backend;
}

}  // namespace boxmot::reid
