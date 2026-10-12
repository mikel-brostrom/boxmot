"""Resolve categorical tuning cases without changing runtime tracker defaults."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import fields
from typing import Any

from boxmot.trackers.common.algorithm_config import TrackerConfig

SCOPED_PARAMETER_PREFIX = "__boxmot_case__"
ASSOCIATION_METRICS = next(field.metadata["choices"] for field in fields(TrackerConfig) if field.name == "asso_func")


def scoped_parameter_name(name: str, entry: dict) -> str:
    """Give incompatible optimizer distributions distinct, deterministic identities."""
    distribution = {key: entry[key] for key in ("type", "range", "options", "values") if key in entry}
    digest = hashlib.sha256(json.dumps(distribution, sort_keys=True).encode()).hexdigest()[:16]
    return f"{SCOPED_PARAMETER_PREFIX}{name}__{digest}"


def map_entries(schema: Mapping[str, Any], transform: Callable[[str, dict], dict]) -> dict:
    """Copy a flattened option tree, transforming entries and activation children."""
    result = {}
    for name, entry in schema.items():
        if not isinstance(entry, dict):
            result[name] = entry
            continue
        entry = dict(entry)
        if isinstance(entry.get("activates"), dict):
            entry["activates"] = map_entries(entry["activates"], transform)
        result[name] = transform(name, entry)
    return result


def matching_case(cases: Mapping, value: Any) -> dict | None:
    """Compare categorical values without equating booleans with integer IDs."""
    return next((entry for key, entry in cases.items() if type(key) is type(value) and key == value), None)


def resolve_entry(entry: dict, parent_value: Any) -> dict:
    """Overlay a matching case; a default-only case disables its distribution."""
    result = {key: value for key, value in entry.items() if key not in {"depends_on", "cases"}}
    case = matching_case(entry["cases"], parent_value)
    if case is None:
        return result
    case = dict(case)
    inactive_children = case.pop("_inactive_children", None)
    if set(case) == {"default"}:
        result = {key: value for key, value in result.items() if key in {"geometry", "activates"}}
    result.update(case)
    if inactive_children is not None:
        result["activates"] = inactive_children
    return result


def _association_values(name: str, entry: dict, flat: Mapping[str, dict]) -> tuple[list[str], list[str]]:
    """Validate applicability metadata and return allowed and selectable metrics."""
    applicable = entry["association"]
    if not isinstance(applicable, list) or not applicable:
        raise ValueError(f"Parameter {name!r} requires a nonempty association list.")
    if any(not isinstance(value, str) or value not in ASSOCIATION_METRICS for value in applicable):
        raise ValueError(f"Parameter {name!r} has an unknown association metric; choose from {ASSOCIATION_METRICS}.")
    if len(set(applicable)) != len(applicable):
        raise ValueError(f"Parameter {name!r} has duplicate association metrics.")
    if name == "asso_func":
        raise ValueError("The asso_func selector cannot declare association applicability for itself.")
    if entry.get("depends_on", "asso_func") != "asso_func":
        raise ValueError(f"Parameter {name!r} combines association applicability with a different depends_on parent.")
    if "default" not in entry:
        raise ValueError(f"Parameter {name!r} needs a runtime default for inactive association metrics.")
    selector = flat.get("asso_func")
    if not isinstance(selector, dict):
        raise ValueError(f"Parameter {name!r} declares association applicability without an asso_func selector.")
    if selector.get("type") in {"choice", "grid_search"}:
        values = selector.get("options") or selector.get("values") or []
    elif selector.get("type") is None and "default" in selector:
        values = [selector["default"]]
    else:
        raise ValueError("Association applicability requires a categorical or fixed asso_func selector.")
    if not values or any(not isinstance(value, str) or value not in ASSOCIATION_METRICS for value in values):
        raise ValueError("Association applicability requires recognized asso_func metric choices.")
    return applicable, list(values)


def compile_association_schema(schema: dict) -> dict:
    """Represent inactive metrics as runtime constants in the shared case tree."""
    from boxmot.engine.tuning.search_space import expand_yaml_groups, flatten_yaml_config

    schema = expand_yaml_groups(schema)
    flat = flatten_yaml_config(schema)

    def compile_entry(name: str, entry: dict) -> dict:
        if "association" not in entry:
            return entry
        applicable, values = _association_values(name, entry, flat)
        result = {key: value for key, value in entry.items() if key != "association"}
        inactive = {value: {"default": entry["default"]} for value in values if value not in applicable}
        if inactive and isinstance(entry.get("activates"), dict):
            # Snapshot runtime defaults before descendant cases can override them.
            children = map_entries(entry["activates"], lambda _name, child: {
                "default": child["default"],
                **({"activates": child["activates"]} if "activates" in child else {}),
            })
            for case in inactive.values():
                case["_inactive_children"] = children
        if inactive:
            result.update(depends_on="asso_func", cases={**result.get("cases", {}), **inactive})
        return result

    return map_entries(schema, compile_entry)


def validate_dependencies(flat: Mapping[str, dict], activation_parents: Mapping[str, str]) -> None:
    """Reject undefined parents, impossible cases, and cycles before sampling."""
    graph = {name: [activation_parents[name]] if name in activation_parents else [] for name in flat}
    for name, entry in flat.items():
        if not isinstance(entry, dict):
            continue
        if "association" in entry:
            _association_values(name, entry, flat)
            graph[name].append("asso_func")
        if "depends_on" not in entry and "cases" not in entry:
            continue
        parent, cases = entry.get("depends_on"), entry.get("cases")
        if not isinstance(parent, str) or parent not in flat:
            raise ValueError(f"Conditional parameter {name!r} references an unknown depends_on parameter: {parent!r}.")
        if not isinstance(cases, Mapping) or not cases:
            raise ValueError(f"Conditional parameter {name!r} requires a nonempty cases mapping.")
        parent_entry = flat[parent]
        if not isinstance(parent_entry, dict):
            raise ValueError(f"Conditional parent {parent!r} must define a categorical choice or a fixed default.")
        if parent_entry.get("type") in {"choice", "grid_search"}:
            values = parent_entry.get("options") or parent_entry.get("values") or ()
        elif parent_entry.get("type") is None and "default" in parent_entry:
            values = (parent_entry["default"],)
        else:
            raise ValueError(f"Conditional parent {parent!r} must be a categorical choice or a fixed default.")
        for value, case in cases.items():
            if not any(type(value) is type(option) and value == option for option in values):
                raise ValueError(f"Conditional parameter {name!r} has an unknown case {value!r} for {parent!r}.")
            if not isinstance(case, dict) or not case:
                raise ValueError(f"Conditional parameter {name!r} case {value!r} must be a nonempty mapping.")
            unsupported = set(case) - {"default", "type", "range", "options", "values"}
            if unsupported:
                raise ValueError(
                    f"Conditional parameter {name!r} has unsupported case metadata: {sorted(unsupported)}."
                )
        graph[name].append(parent)
    active, complete = set(), set()

    def visit(name: str) -> None:
        if name in active:
            raise ValueError(f"Cyclic tuning dependency involving {name!r}.")
        if name in complete:
            return
        active.add(name)
        for parent in graph[name]:
            visit(parent)
        active.remove(name)
        complete.add(name)

    for name in graph:
        visit(name)


def branch_parameters(schema: dict) -> set[str]:
    """Identify case-controlled parameters and selectors recorded in each trial."""
    from boxmot.engine.tuning.search_space import flatten_yaml_config

    flat = flatten_yaml_config(schema)
    names = {name for name, entry in flat.items() if isinstance(entry, dict) and "depends_on" in entry}
    names.update(flat[name]["depends_on"] for name in tuple(names))
    for entry in flat.values():
        if isinstance(entry, dict):
            for case in entry.get("cases", {}).values():
                names.update(flatten_yaml_config(case.get("_inactive_children", {})))
    return names


def resolve_fixed_cases(schema: dict) -> dict:
    """Resolve default-only selectors for callers that already filtered a schema."""
    from boxmot.engine.tuning.search_space import flatten_yaml_config

    schema = compile_association_schema(schema)
    while True:
        flat = flatten_yaml_config(schema)
        fixed = {
            name: entry["default"] for name, entry in flat.items()
            if isinstance(entry, dict) and set(entry) == {"default"}
        }
        ready = {name for name, entry in flat.items() if isinstance(entry, dict) and entry.get("depends_on") in fixed}
        if not ready:
            return schema
        schema = map_entries(
            schema,
            lambda name, entry: resolve_entry(entry, fixed[entry["depends_on"]]) if name in ready else entry,
        )


def _fold_fixed_activations(schema: dict, *, inactive: bool = False) -> dict:
    """Flatten selected toggles without making assumptions about caller geometry."""
    result = {}
    for name, entry in schema.items():
        if not isinstance(entry, dict):
            result[name] = entry
            continue
        children = entry.get("activates")
        details = {key: value for key, value in entry.items() if key != "activates"}
        fixed = inactive or not {"type", "depends_on"}.intersection(details)
        if fixed:
            result[name] = {"default": entry["default"]}
            if isinstance(children, dict):
                result.update(_fold_fixed_activations(children, inactive=inactive or not bool(entry["default"])))
        else:
            if isinstance(children, dict):
                details["activates"] = _fold_fixed_activations(children)
            result[name] = details
    return result


def categorical_branch(schema: dict, controlled: set[str] | None = None) -> tuple[str, list[tuple[Any, dict]]] | None:
    """Choose a dependency selector and return independently resolved branches."""
    from boxmot.engine.tuning.search_space import (
        conditional_yaml_tree,
        flatten_yaml_config,
    )

    flat = flatten_yaml_config(schema)
    _, _, activation_parents = conditional_yaml_tree(schema)
    dependent = next((name for name, entry in flat.items() if isinstance(entry, dict) and "depends_on" in entry), None)
    if dependent is None:
        parent = next(
            (activation_parents[name] for name in sorted(controlled or ()) if name in activation_parents), None
        )
        if parent is None:
            return None
    else:
        parent = flat[dependent]["depends_on"]
    visited = set()
    while True:
        if parent in visited:
            raise ValueError(f"Cyclic tuning dependency involving {parent!r}.")
        visited.add(parent)
        if "depends_on" in flat[parent]:
            parent = flat[parent]["depends_on"]
        elif parent in activation_parents:
            parent = activation_parents[parent]
        else:
            break
    entry = flat[parent]
    values = entry.get("options") or entry.get("values")
    if entry.get("type") not in {"choice", "grid_search"} or not values:
        raise ValueError(f"Unresolved conditional parent {parent!r} must be categorical.")
    branches = []
    for value in values:
        selected = map_entries(
            schema,
            lambda name, item: (
                {"default": value, **({"activates": item["activates"]} if "activates" in item else {})}
                if name == parent else item
            ),
        )
        branches.append((value, _fold_fixed_activations(resolve_fixed_cases(selected))))
    return parent, branches
