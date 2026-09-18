"""Verify the installed-release guard catches missing autocomplete metadata."""

from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass
from pathlib import Path

import pytest
from typing_extensions import TypedDict

import boxmot
from tests.ci import release_contract
from tests.ci.release_contract import check_typing_metadata


def test_release_typing_metadata() -> None:
    check_typing_metadata()


def test_release_rejects_missing_typing_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(boxmot, "__file__", str(tmp_path / "__init__.py"))

    with pytest.raises(AssertionError, match="Missing packaged typing marker"):
        check_typing_metadata()


@pytest.mark.parametrize(
    "module_name,alias_name",
    (
        ("boxmot.detectors._model_names", "DetectorName"),
        ("boxmot.reid._model_names", "ReIDName"),
        ("boxmot.trackers.common._model_names", "TrackerName"),
    ),
)
def test_release_rejects_untyped_factory_names(
    module_name: str, alias_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(importlib.import_module(module_name), alias_name, str)

    with pytest.raises(AssertionError, match=rf"{alias_name} must include Literal"):
        check_typing_metadata()


def test_release_rejects_missing_constructor_keyword_types(monkeypatch: pytest.MonkeyPatch) -> None:
    constructor = importlib.import_module("boxmot.trackers.common.constructor")
    monkeypatch.setattr(constructor, "BoxTrackerOptions", str)

    with pytest.raises(AssertionError, match="BoxTrackerOptions must be a TypedDict"):
        check_typing_metadata()


def test_release_rejects_incomplete_constructor_keyword_types(monkeypatch: pytest.MonkeyPatch) -> None:
    class IncompleteOptions(TypedDict, total=False):
        max_obs: int

    constructor = importlib.import_module("boxmot.trackers.common.constructor")
    monkeypatch.setattr(constructor, "BoxTrackerOptions", IncompleteOptions)

    with pytest.raises(AssertionError, match="BoxTrackerOptions must include 'is_obb'"):
        check_typing_metadata()


def test_release_rejects_required_inherited_constructor_options(monkeypatch: pytest.MonkeyPatch) -> None:
    class RequiredOptions(TypedDict):
        is_obb: bool

    constructor = importlib.import_module("boxmot.trackers.common.constructor")
    monkeypatch.setattr(constructor, "BoxTrackerOptions", RequiredOptions)

    with pytest.raises(AssertionError, match="BoxTrackerOptions constructor keywords must remain optional"):
        check_typing_metadata()


def test_release_rejects_mutable_reid_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    @dataclass
    class MutableReIDConfig:
        model: str = "osnet-x0-25-msmt17"

    monkeypatch.setattr(boxmot, "ReIDConfig", MutableReIDConfig)
    with pytest.raises(AssertionError, match="ReIDConfig must remain immutable"):
        check_typing_metadata()


def test_release_rejects_missing_reid_model_autocomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(release_contract, "get_overloads", lambda function: [])
    with pytest.raises(AssertionError, match="ReIDConfig.model must retain Literal autocomplete"):
        check_typing_metadata()


def test_release_rejects_missing_reid_batch_configuration_type(monkeypatch: pytest.MonkeyPatch) -> None:
    get_type_hints = release_contract.get_type_hints

    def without_batch_size(value):
        hints = get_type_hints(value)
        if value is boxmot.ReIDConfig:
            hints.pop("batch_size")
        return hints

    monkeypatch.setattr(release_contract, "get_type_hints", without_batch_size)
    with pytest.raises(AssertionError, match="ReIDConfig must retain its typed inference fields"):
        check_typing_metadata()


@pytest.mark.parametrize("missing_doc", (None, "A summary without constructor arguments."))
def test_release_rejects_missing_constructor_tooltip_docs(
    monkeypatch: pytest.MonkeyPatch, missing_doc: str | None
) -> None:
    get_docstring = ast.get_docstring

    def constructor_without_docs(node: ast.AST, clean: bool = True) -> str | None:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            return missing_doc
        return get_docstring(node, clean=clean)

    monkeypatch.setattr(release_contract.ast, "get_docstring", constructor_without_docs)
    with pytest.raises(AssertionError, match=r"__init__ must own Args documentation"):
        release_contract.check_tracker_constructor_docs()
