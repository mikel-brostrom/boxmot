"""Static checks for the model-free native tracker link boundary."""

from __future__ import annotations

import re
from pathlib import Path

NATIVE_CPP = Path(__file__).resolve().parents[3] / "boxmot" / "native" / "cpp"
BASE_CMAKE = NATIVE_CPP / "trackers" / "base" / "CMakeLists.txt"
REID_CMAKE = NATIVE_CPP / "reid" / "CMakeLists.txt"
TRACKER_NAMES = ("botsort", "bytetrack", "occluboost", "ocsort", "sfsort")


def _command_body(source: str, command: str, target: str) -> str:
    match = re.search(
        rf"{re.escape(command)}\(\s*{re.escape(target)}(?=\s|\))(?P<body>.*?)\)",
        source,
        flags=re.DOTALL,
    )
    assert match is not None, f"Missing {command}({target} ...)"
    return match.group("body")


def test_tracker_base_contains_no_reid_or_model_runtime_sources() -> None:
    source = BASE_CMAKE.read_text(encoding="utf-8")
    tracker_sources = _command_body(source, "add_library", "boxmot_tracker_base")
    tracker_links = _command_body(source, "target_link_libraries", "boxmot_tracker_base")

    assert "src/association.cpp" in tracker_sources
    assert "src/assignment.cpp" in tracker_sources
    assert "reid" not in tracker_sources.lower()
    assert "onnx" not in tracker_sources.lower()
    assert "opencv" not in tracker_links.lower()
    assert "reid" not in tracker_links.lower()
    assert "onnx" not in tracker_links.lower()
    assert "target_compile_definitions(boxmot_tracker_base" not in source
    assert "reid" not in source.lower()
    assert "onnx" not in source.lower()
    assert "dnn" not in source.lower()


def test_reid_sources_and_dependencies_are_scoped_to_the_reid_target() -> None:
    source = REID_CMAKE.read_text(encoding="utf-8")
    reid_sources = _command_body(source, "add_library", "boxmot_native_reid")
    reid_links = _command_body(source, "target_link_libraries", "boxmot_native_reid")
    capi_links = _command_body(source, "target_link_libraries", "reid_capi")

    assert "src/reid_inference_backend.cpp" in reid_sources
    assert "src/reid_onnx.cpp" in reid_sources
    assert "${OpenCV_LIBS}" in reid_links
    assert "boxmot_native_reid" in capi_links
    assert "boxmot_tracker_base" not in capi_links
    assert "find_package(OpenCV 4 REQUIRED COMPONENTS core dnn imgproc)" in source
    assert "target_compile_definitions(boxmot_native_reid PRIVATE BOXMOT_HAS_ONNXRUNTIME)" in source
    assert "target_link_libraries(boxmot_native_reid PUBLIC ${BOXMOT_ORT_TARGET})" in source
    assert "boxmot_tracker_base" not in source


def test_reid_cpp_tree_has_no_tracker_namespace_or_header_dependency() -> None:
    reid_root = NATIVE_CPP / "reid"
    expected_files = {
        "CMakeLists.txt",
        "include/boxmot/reid/reid_capi.h",
        "include/boxmot/reid/reid_inference_backend.hpp",
        "include/boxmot/reid/reid_onnx.hpp",
        "src/reid_capi.cpp",
        "src/reid_inference_backend.cpp",
        "src/reid_onnx.cpp",
    }
    actual_files = {
        path.relative_to(reid_root).as_posix() for path in reid_root.rglob("*") if path.is_file()
    }

    assert actual_files == expected_files
    for relative_path in sorted(expected_files - {"CMakeLists.txt"}):
        source = (reid_root / relative_path).read_text(encoding="utf-8")
        assert "boxmot/trackers/" not in source, relative_path
        assert "boxmot::trackers" not in source, relative_path


def test_tracker_base_tree_contains_no_reid_implementation_files() -> None:
    base_root = NATIVE_CPP / "trackers" / "base"

    assert not [path for path in base_root.rglob("*") if "reid" in path.name.lower()]


def test_tracker_projects_do_not_request_reid_or_dnn_dependencies() -> None:
    for tracker_name in TRACKER_NAMES:
        source = (NATIVE_CPP / "trackers" / tracker_name / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        lowered = source.lower()
        assert "reid" not in lowered, tracker_name
        assert "onnx" not in lowered, tracker_name
        assert " dnn" not in lowered, tracker_name


def test_tracker_cmake_helper_links_only_the_model_free_base() -> None:
    helper = (NATIVE_CPP / "cmake" / "BoxMOTNative.cmake").read_text(encoding="utf-8")
    core_links = _command_body(helper, "target_link_libraries", "${_core}")

    assert "boxmot_tracker_base" in core_links
    assert "boxmot_native_reid" not in helper
    assert "onnx" not in helper.lower()
    assert " dnn" not in helper.lower()


def test_aggregate_build_opts_into_reid_without_global_model_discovery() -> None:
    source = (NATIVE_CPP / "CMakeLists.txt").read_text(encoding="utf-8")

    assert 'option(BOXMOT_BUILD_NATIVE_REID "Build the independent native ReID runtime and C ABI" ON)' in source
    assert source.index("if(BOXMOT_BUILD_NATIVE_REID)") < source.index("add_subdirectory(reid)")
    lowered = source.lower()
    assert "find_package(opencv" not in lowered
    assert "dnn" not in lowered
    assert "onnx" not in lowered
