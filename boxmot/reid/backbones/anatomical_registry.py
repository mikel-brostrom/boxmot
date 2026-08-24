"""Canonical anatomical-teacher mode specifications.

Keep mode capabilities here so model construction and training validation cannot
silently disagree about what a target type requires.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_ANATOMICAL_TARGET_TYPE = "deterministic_scale_aware_geometry"
V8_ANATOMICAL_TARGET_TYPE = "learned_pose_concat_ema"


@dataclass(frozen=True, slots=True)
class AnatomicalTargetSpec:
    """Capabilities owned by one anatomical-supervision target."""

    name: str
    uses_ema_teacher: bool = False
    uses_semantic_teacher: bool = False
    uses_decoupled_queries: bool = False
    uses_privileged_attention: bool = False
    uses_body_slots: bool = False


ANATOMICAL_TARGET_SPECS = {
    spec.name: spec
    for spec in (
        AnatomicalTargetSpec(DEFAULT_ANATOMICAL_TARGET_TYPE),
        AnatomicalTargetSpec(
            V8_ANATOMICAL_TARGET_TYPE,
            uses_ema_teacher=True,
        ),
        AnatomicalTargetSpec(
            "learned_pose_semantic_ema",
            uses_ema_teacher=True,
            uses_semantic_teacher=True,
        ),
        AnatomicalTargetSpec(
            "learned_pose_semantic_fused_ema",
            uses_ema_teacher=True,
            uses_semantic_teacher=True,
        ),
        AnatomicalTargetSpec(
            "privileged_mask_pose_attention",
            uses_privileged_attention=True,
        ),
        AnatomicalTargetSpec(
            "decoupled_pose_parsing_teacher",
            uses_ema_teacher=True,
            uses_decoupled_queries=True,
        ),
        AnatomicalTargetSpec(
            "body_slot_privileged_ema",
            uses_body_slots=True,
        ),
    )
}

ANATOMICAL_TARGET_TYPES = frozenset(ANATOMICAL_TARGET_SPECS)
EMA_ANATOMICAL_TARGET_TYPES = frozenset(
    name
    for name, spec in ANATOMICAL_TARGET_SPECS.items()
    if spec.uses_ema_teacher
)
SEMANTIC_ANATOMICAL_TARGET_TYPES = frozenset(
    name
    for name, spec in ANATOMICAL_TARGET_SPECS.items()
    if spec.uses_semantic_teacher
)


def get_anatomical_target_spec(name: str) -> AnatomicalTargetSpec:
    """Return the normalized target specification or raise a useful error."""
    normalized = str(name).lower()
    try:
        return ANATOMICAL_TARGET_SPECS[normalized]
    except KeyError as error:
        raise ValueError(
            f"Unsupported anatomical target type: {name!r}; "
            f"expected one of {sorted(ANATOMICAL_TARGET_TYPES)}"
        ) from error
