"""Keep editor-facing factory signatures aligned with their runtime APIs."""

from __future__ import annotations

import ast
import copy
import importlib
import inspect
from typing import Literal, get_args, get_origin

import pytest


@pytest.mark.parametrize(
    "namespace,name,representative",
    (
        ("boxmot", "create_tracker", "occluboost"),
        ("boxmot.trackers", "create_tracker", "eagermot"),
        ("boxmot.detectors", "create_detector", "yolo26n"),
        ("boxmot.reid", "create_reid_encoder", "osnet-x0-25-msmt17"),
    ),
)
def test_public_factories_expose_catalog_overloads_and_preserve_runtime_signature(
    namespace: str, name: str, representative: str
) -> None:
    """A new runtime keyword must also be usable through either typed overload."""
    package = importlib.import_module(namespace)
    factory = getattr(package, name)
    module = importlib.import_module(factory.__module__)
    tree = ast.parse(inspect.getsource(module))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    overloads = [
        node
        for node in definitions
        if any(isinstance(decorator, ast.Name) and decorator.id == "overload" for decorator in node.decorator_list)
    ]
    assert len(overloads) == 2
    implementation = next(node for node in definitions if node not in overloads)

    catalog_annotation = overloads[0].args.args[0].annotation
    assert isinstance(catalog_annotation, ast.Name)
    catalog = vars(module)[catalog_annotation.id]
    assert get_origin(catalog) is Literal
    assert representative in get_args(catalog)

    # The open-ended overload retains custom strings, paths, mappings and specs
    # wherever the runtime accepts them, without turning those calls into Any.
    assert ast.dump(overloads[1].args) == ast.dump(implementation.args)
    for definition in overloads:
        assert ast.dump(definition.returns) == ast.dump(implementation.returns)
        arguments = copy.deepcopy(definition.args)
        arguments.args[0].annotation = implementation.args.args[0].annotation
        assert ast.dump(arguments) == ast.dump(implementation.args)

    package_tree = ast.parse(inspect.getsource(package))
    type_only_imports = [
        imported
        for node in package_tree.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
        for statement in node.body
        if isinstance(statement, ast.ImportFrom)
        for imported in statement.names
    ]
    assert any(imported.name == name and imported.asname in (None, name) for imported in type_only_imports)


def test_reid_config_constructor_exposes_model_name_overload_and_all_settings() -> None:
    """Complete known model names while retaining typed custom artifact paths."""
    from boxmot.reid import ReIDConfig

    module = importlib.import_module(ReIDConfig.__module__)
    tree = ast.parse(inspect.getsource(module))
    config_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ReIDConfig")
    constructors = [
        node for node in config_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    assert len(constructors) == 3
    catalog, flexible, implementation = constructors
    annotation = catalog.args.args[1].annotation
    assert isinstance(annotation, ast.Name)
    names = vars(module)[annotation.id]
    assert get_origin(names) is Literal
    assert "osnet-x0-25-msmt17" in get_args(names)
    assert ast.dump(flexible.args) == ast.dump(implementation.args)
    arguments = copy.deepcopy(catalog.args)
    arguments.args[1].annotation = implementation.args.args[1].annotation
    assert ast.dump(arguments) == ast.dump(implementation.args)
    assert {parameter.name for parameter in inspect.signature(ReIDConfig).parameters.values()} == {
        "model", "device", "precision", "preprocessing", "batch_size", "image_size", "embedding_dim", "allow_download"
    }
