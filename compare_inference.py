"""
模型推理对比工具

对比 PyTorch、ONNX、OpenVINO 三种后端的推理结果和性能。

功能:
    - 加载并运行三种后端
    - 对比输出差异（绝对差异、相对差异、余弦相似度）
    - 性能基准测试（FPS、延迟）
    - 生成详细对比报告

使用:
    python compare_inference.py --model yolov8n.pt
    python compare_inference.py --model yolov8n.pt --onnx exports/yolov8n/yolov8n.onnx
    python compare_inference.py --model yolov8n.pt --openvino exports/yolov8n/yolov8n_openvino/model.xml
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch


class InferenceComparer:
    """
    多后端推理对比器。

    Attributes:
        results (dict): 存储推理结果
    """

    def __init__(self):
        self.results = {}

    def load_pytorch_model(self, model_path):
        """
        加载 PyTorch 模型。

        Args:
            model_path (str): 模型路径

        Returns:
            torch.nn.Module: 加载的模型，失败返回 None
        """
        print(f"\n🔷 Loading PyTorch model: {model_path}")
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            print("  ✓ PyTorch model loaded")
            return model.model.eval()
        except Exception as e:
            print(f"  ✗ Failed to load PyTorch model: {e}")
            return None

    def load_onnx_model(self, onnx_path):
        """Load ONNX model."""
        print(f"\n🔶 Loading ONNX model: {onnx_path}")
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(
                str(onnx_path),
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            print(f"  ✓ ONNX model loaded")
            print(f"  Provider: {session.get_providers()[0]}")
            return session
        except Exception as e:
            print(f"  ✗ Failed to load ONNX model: {e}")
            return None

    def load_openvino_model(self, xml_path, device="CPU"):
        """Load OpenVINO model."""
        print(f"\n🔷 Loading OpenVINO model: {xml_path}")
        try:
            import openvino as ov
            core = ov.Core()
            model = core.read_model(xml_path)
            compiled_model = core.compile_model(model, device)
            print("  ✓ OpenVINO model loaded")
            print(f"  Device: {device}")
            return compiled_model
        except Exception as e:
            print(f"  ✗ Failed to load OpenVINO model: {e}")
            return None

    def preprocess_image(self, image_path, input_size=(640, 640)):
        """Preprocess image for inference."""
        if isinstance(image_path, str) or isinstance(image_path, Path):
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"  Warning: Could not load image {image_path}, using random input")
                img = np.random.randint(0, 255, (*input_size, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, input_size)
        else:
            img = image_path

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] and convert to CHW format
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_normalized, (2, 0, 1))

        # Add batch dimension
        img_batch = np.expand_dims(img_chw, axis=0)

        return img_batch

    def run_pytorch_inference(self, model, input_tensor, num_runs=100):
        """Run PyTorch inference."""
        print(f"\n🔷 Running PyTorch inference ({num_runs} iterations)...")

        # Convert to torch tensor
        if isinstance(input_tensor, np.ndarray):
            input_torch = torch.from_numpy(input_tensor)
        else:
            input_torch = input_tensor

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_torch)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                output = model(input_torch)
                times.append((time.perf_counter() - start) * 1000)

        # Get final output for comparison
        with torch.no_grad():
            final_output = model(input_torch)

        self._print_stats("PyTorch", times)
        return final_output, times

    def run_onnx_inference(self, session, input_tensor, num_runs=100):
        """Run ONNX inference."""
        print(f"\n🔶 Running ONNX inference ({num_runs} iterations)...")

        # Ensure numpy array
        if isinstance(input_tensor, torch.Tensor):
            input_np = input_tensor.cpu().numpy()
        else:
            input_np = input_tensor.astype(np.float32)

        input_name = session.get_inputs()[0].name

        # Warmup
        for _ in range(10):
            _ = session.run(None, {input_name: input_np})

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            output = session.run(None, {input_name: input_np})
            times.append((time.perf_counter() - start) * 1000)

        # Get final output for comparison
        final_output = session.run(None, {input_name: input_np})

        self._print_stats("ONNX", times)
        return final_output, times

    def run_openvino_inference(self, compiled_model, input_tensor, num_runs=100):
        """Run OpenVINO inference."""
        print(f"\n🔷 Running OpenVINO inference ({num_runs} iterations)...")

        # Ensure numpy array
        if isinstance(input_tensor, torch.Tensor):
            input_np = input_tensor.cpu().numpy()
        else:
            input_np = input_tensor.astype(np.float32)

        # Warmup
        for _ in range(10):
            _ = compiled_model(input_np)

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            output = compiled_model(input_np)
            times.append((time.perf_counter() - start) * 1000)

        # Get final output for comparison
        final_output = compiled_model(input_np)

        # Convert OpenVINO output to numpy array
        if hasattr(final_output, 'values'):
            final_output_np = list(final_output.values())[0]
        elif isinstance(final_output, dict):
            final_output_np = list(final_output.values())[0]
        else:
            final_output_np = final_output

        self._print_stats("OpenVINO", times)
        return final_output_np, times

    def _print_stats(self, backend, times):
        """Print inference statistics."""
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000 / avg_time

        print(f"  Average: {avg_time:.2f} ms ± {std_time:.2f} ms")
        print(f"  Min/Max: {min_time:.2f} / {max_time:.2f} ms")
        print(f"  FPS: {fps:.2f}")

        return {
            "backend": backend,
            "avg_ms": avg_time,
            "std_ms": std_time,
            "min_ms": min_time,
            "max_ms": max_time,
            "fps": fps
        }

    def compare_outputs(self, output1, output2, name1="Reference", name2="Target", rtol=1e-3, atol=1e-5):
        """Compare two model outputs."""
        print(f"\n📊 Comparing {name1} vs {name2}...")

        # Convert to numpy
        def to_numpy(x):
            # Handle numpy arrays first (most common case)
            if isinstance(x, np.ndarray):
                return x
            # Handle torch tensors
            elif isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            # Handle OpenVINO outputs (has .values() method)
            elif hasattr(x, 'values') and callable(getattr(x, 'values')):
                values = list(x.values())
                return values[0] if values else np.array([])
            # Handle dict-like outputs
            elif isinstance(x, dict):
                return list(x.values())[0] if x else np.array([])
            # Handle multi-output models (list/tuple)
            elif isinstance(x, (list, tuple)):
                if len(x) > 0:
                    first = x[0]
                    if isinstance(first, torch.Tensor):
                        return first.detach().cpu().numpy()
                    elif isinstance(first, np.ndarray):
                        return first
                    else:
                        return np.array(first)
                return np.array([])
            # Fallback: try to convert to array
            else:
                try:
                    return np.array(x)
                except Exception as e:
                    print(f"  ⚠ Warning: Cannot convert to numpy: {type(x)}, error: {e}")
                    # Try to extract first item if iterable
                    try:
                        return np.array(list(x)[0])
                    except:
                        return np.array([])

        arr1 = to_numpy(output1)
        arr2 = to_numpy(output2)

        # Handle different output shapes
        if arr1.shape != arr2.shape:
            print(f"  ⚠ Shape mismatch: {arr1.shape} vs {arr2.shape}")
            # Try to flatten and compare if possible
            arr1_flat = arr1.flatten()
            arr2_flat = arr2.flatten()
            min_len = min(len(arr1_flat), len(arr2_flat))
            arr1 = arr1_flat[:min_len]
            arr2 = arr2_flat[:min_len]
            print(f"  Using first {min_len} elements for comparison")

        # Calculate differences
        abs_diff = np.abs(arr1 - arr2)
        rel_diff = abs_diff / (np.abs(arr1) + 1e-8)

        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)

        # Check tolerance
        is_close = np.allclose(arr1, arr2, rtol=rtol, atol=atol)

        # Calculate cosine similarity
        arr1_flat = arr1.flatten()
        arr2_flat = arr2.flatten()
        cosine_sim = np.dot(arr1_flat, arr2_flat) / (
            np.linalg.norm(arr1_flat) * np.linalg.norm(arr2_flat) + 1e-8
        )

        print(f"  Max absolute diff: {max_abs_diff:.6e}")
        print(f"  Mean absolute diff: {mean_abs_diff:.6e}")
        print(f"  Max relative diff: {max_rel_diff:.6e}")
        print(f"  Mean relative diff: {mean_rel_diff:.6e}")
        print(f"  Cosine similarity: {cosine_sim:.6f}")
        print(f"  Within tolerance: {'✓ PASS' if is_close else '✗ FAIL'}")

        return {
            "max_abs_diff": float(max_abs_diff),
            "mean_abs_diff": float(mean_abs_diff),
            "max_rel_diff": float(max_rel_diff),
            "mean_rel_diff": float(mean_rel_diff),
            "cosine_similarity": float(cosine_sim),
            "is_close": is_close
        }

    def print_summary(self, results):
        """Print comprehensive summary."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Performance table
        print("\n📈 Performance Comparison:")
        print(f"{'Backend':<15} {'Avg (ms)':<12} {'Std (ms)':<12} {'FPS':<10} {'Speedup':<10}")
        print("-" * 70)

        pytorch_fps = results.get("pytorch_stats", {}).get("fps", 1.0)

        for backend in ["pytorch", "onnx", "openvino"]:
            key = f"{backend}_stats"
            if key in results:
                stats = results[key]
                speedup = stats["fps"] / pytorch_fps if pytorch_fps > 0 else 0
                print(f"{stats['backend']:<15} "
                      f"{stats['avg_ms']:<12.2f} "
                      f"{stats['std_ms']:<12.2f} "
                      f"{stats['fps']:<10.2f} "
                      f"{speedup:<10.2f}x")

        # Accuracy table
        print("\n🎯 Accuracy Comparison (vs PyTorch):")
        print(f"{'Backend':<15} {'Max Abs Diff':<15} {'Mean Abs Diff':<15} {'Cosine Sim':<12} {'Status':<10}")
        print("-" * 75)

        for backend in ["onnx", "openvino"]:
            key = f"{backend}_comparison"
            if key in results:
                comp = results[key]
                status = "✓ PASS" if comp["is_close"] else "✗ FAIL"
                print(f"{backend.upper():<15} "
                      f"{comp['max_abs_diff']:<15.6e} "
                      f"{comp['mean_abs_diff']:<15.6e} "
                      f"{comp['cosine_similarity']:<12.6f} "
                      f"{status:<10}")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare inference across backends")
    parser.add_argument("--model", type=str, help="PyTorch model path (.pt)")
    parser.add_argument("--onnx", type=str, help="ONNX model path (.onnx)")
    parser.add_argument("--openvino", type=str, help="OpenVINO model path (.xml)")
    parser.add_argument("--image", type=str, help="Test image path (optional)")
    parser.add_argument("--input-size", type=int, nargs=2, default=[640, 640], help="Input size (H W)")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--ov-device", type=str, default="CPU", choices=["CPU", "GPU", "NPU"],
                        help="OpenVINO target device (default: CPU)")
    parser.add_argument("--save-outputs", type=str, help="Directory to save model outputs for manual comparison")

    args = parser.parse_args()

    comparer = InferenceComparer()
    results = {}

    # Create output directory if needed
    if args.save_outputs:
        output_dir = Path(args.save_outputs)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Saving outputs to: {output_dir}")

    # Prepare input
    if args.image and Path(args.image).exists():
        print(f"\nUsing test image: {args.image}")
        input_tensor = comparer.preprocess_image(args.image, tuple(args.input_size))
    else:
        print(f"\nUsing random input: {args.input_size}")
        input_tensor = np.random.randn(1, 3, args.input_size[0], args.input_size[1]).astype(np.float32)

    pytorch_output = None
    onnx_output = None
    ov_output = None

    # Run PyTorch inference
    if args.model:
        model = comparer.load_pytorch_model(args.model)
        if model is not None:
            pytorch_output, pytorch_times = comparer.run_pytorch_inference(model, input_tensor, args.num_runs)
            results["pytorch_stats"] = comparer._print_stats("PyTorch", pytorch_times)

            # Save PyTorch output
            if args.save_outputs:
                if isinstance(pytorch_output, torch.Tensor):
                    output_np = pytorch_output.detach().cpu().numpy()
                elif isinstance(pytorch_output, (list, tuple)):
                    # Handle multi-output models (e.g., YOLO returns list of tensors)
                    if len(pytorch_output) > 0:
                        # Save first output (main detection output)
                        first_output = pytorch_output[0]
                        if isinstance(first_output, torch.Tensor):
                            output_np = first_output.detach().cpu().numpy()
                        else:
                            output_np = np.array(first_output)
                    else:
                        output_np = np.array([])
                else:
                    output_np = np.array(pytorch_output)

                np.save(output_dir / "pytorch_output.npy", output_np)
                print(f"  Saved: {output_dir / 'pytorch_output.npy'}")
                print(f"    Shape: {output_np.shape}, dtype: {output_np.dtype}")

    # Run ONNX inference
    if args.onnx:
        onnx_session = comparer.load_onnx_model(args.onnx)
        if onnx_session is not None:
            onnx_output, onnx_times = comparer.run_onnx_inference(onnx_session, input_tensor, args.num_runs)
            results["onnx_stats"] = comparer._print_stats("ONNX", onnx_times)

            # Save ONNX output
            if args.save_outputs:
                if isinstance(onnx_output, list):
                    # ONNX typically returns list of arrays
                    output_np = onnx_output[0] if len(onnx_output) > 0 else np.array([])
                elif isinstance(onnx_output, np.ndarray):
                    output_np = onnx_output
                else:
                    output_np = np.array(onnx_output)

                np.save(output_dir / "onnx_output.npy", output_np)
                print(f"  Saved: {output_dir / 'onnx_output.npy'}")
                print(f"    Shape: {output_np.shape}, dtype: {output_np.dtype}")

            if pytorch_output is not None:
                results["onnx_comparison"] = comparer.compare_outputs(
                    pytorch_output, onnx_output, "PyTorch", "ONNX"
                )

    # Run OpenVINO inference
    if args.openvino:
        ov_model = comparer.load_openvino_model(args.openvino, device=args.ov_device)
        if ov_model is not None:
            ov_output, ov_times = comparer.run_openvino_inference(ov_model, input_tensor, args.num_runs)
            results["openvino_stats"] = comparer._print_stats("OpenVINO", ov_times)

            # Save OpenVINO output
            if args.save_outputs:
                if hasattr(ov_output, 'values'):
                    # OpenVINO returns dict-like object
                    output_np = list(ov_output.values())[0]
                elif isinstance(ov_output, dict):
                    output_np = list(ov_output.values())[0]
                elif isinstance(ov_output, np.ndarray):
                    output_np = ov_output
                else:
                    output_np = np.array(ov_output)

                np.save(output_dir / "openvino_output.npy", output_np)
                print(f"  Saved: {output_dir / 'openvino_output.npy'}")
                print(f"    Shape: {output_np.shape}, dtype: {output_np.dtype}")

            if pytorch_output is not None:
                results["openvino_comparison"] = comparer.compare_outputs(
                    pytorch_output, ov_output, "PyTorch", "OpenVINO"
                )

    # Print summary
    comparer.print_summary(results)

    if args.save_outputs:
        print(f"\n💾 Model outputs saved to: {output_dir}")
        print("  You can manually compare them using:")
        print(f"  import numpy as np")
        print(f"  pytorch = np.load('{output_dir}/pytorch_output.npy')")
        print(f"  onnx = np.load('{output_dir}/onnx_output.npy')")
        print(f"  openvino = np.load('{output_dir}/openvino_output.npy')")

    print("\n✓ Comparison complete!")

    return results


if __name__ == "__main__":
    main()
