import os

from boxmot.appearance.exporters.base_exporter import BaseExporter


class TFLiteExporter(BaseExporter):
    group = "tflite"
    cmds = "--extra-index-url https://pypi.ngc.nvidia.com"

    def export(self):

        import onnx2tf

        input_onnx_file_path = str(self.file.with_suffix(".onnx"))
        output_folder_path = input_onnx_file_path.replace(
            ".onnx", f"_saved_model{os.sep}"
        )
        onnx2tf.convert(
            input_onnx_file_path=input_onnx_file_path,
            output_folder_path=output_folder_path,
            not_use_onnxsim=True,
            verbosity=True,
            # output_integer_quantized_tflite=self.args.int8,
            # quant_type="per-tensor",  # "per-tensor" (faster) or "per-channel" (slower but more accurate)
            # custom_input_op_name_np_data_path=np_data,
        )
        return output_folder_path
