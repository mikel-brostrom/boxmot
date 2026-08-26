"""Declarative ReID head specifications shared by CLIs, trainers, and models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HeadImplementation(str, Enum):
    """Concrete head implementation selected by a training configuration."""

    MULTI_BRANCH = "multi_branch"
    BODY_SLOT = "body_slot"
    GPC_LITE = "gpc_lite"
    LMBN = "lmbn"

    def __str__(self) -> str:
        """Return the string value, matching ``StrEnum`` semantics."""
        return self.value


@dataclass(frozen=True)
class ReIDHeadSpec:
    """Static capabilities of one ReID head treatment."""

    name: str
    implementation: HeadImplementation
    families: frozenset[str]
    train_selectable: bool = True
    specialist: bool = False
    channel_control: bool = False
    supports_scale_balance: bool = False

    def supports_family(self, family: str) -> bool:
        """Return whether this head is implemented for a backbone family."""
        return family in self.families


_CSL = frozenset({"csl_tinyvit"})
_CSL_AND_MOBILENET = frozenset({"csl_tinyvit", "mobilenetv4"})

REID_HEAD_SPECS: dict[str, ReIDHeadSpec] = {
    "standard": ReIDHeadSpec(
        name="standard",
        implementation=HeadImplementation.MULTI_BRANCH,
        families=_CSL_AND_MOBILENET,
        supports_scale_balance=True,
    ),
    "gpc_lite": ReIDHeadSpec(
        name="gpc_lite",
        implementation=HeadImplementation.GPC_LITE,
        families=_CSL_AND_MOBILENET,
    ),
    "stage2_channel2": ReIDHeadSpec(
        name="stage2_channel2",
        implementation=HeadImplementation.MULTI_BRANCH,
        families=_CSL,
        specialist=True,
        channel_control=True,
        supports_scale_balance=True,
    ),
    "multiscale_channel2": ReIDHeadSpec(
        name="multiscale_channel2",
        implementation=HeadImplementation.MULTI_BRANCH,
        families=_CSL,
        specialist=True,
        channel_control=True,
        supports_scale_balance=True,
    ),
    "stage2_pg": ReIDHeadSpec(
        name="stage2_pg",
        implementation=HeadImplementation.MULTI_BRANCH,
        families=_CSL,
        specialist=True,
        supports_scale_balance=True,
    ),
    "stage2_gpc_lite": ReIDHeadSpec(
        name="stage2_gpc_lite",
        implementation=HeadImplementation.MULTI_BRANCH,
        families=_CSL,
        specialist=True,
        supports_scale_balance=True,
    ),
    "stage2_gpc_lite_gate": ReIDHeadSpec(
        name="stage2_gpc_lite_gate",
        implementation=HeadImplementation.MULTI_BRANCH,
        families=_CSL,
        specialist=True,
        supports_scale_balance=True,
    ),
    "stage2_pg_gate": ReIDHeadSpec(
        name="stage2_pg_gate",
        implementation=HeadImplementation.MULTI_BRANCH,
        families=_CSL,
        specialist=True,
        supports_scale_balance=True,
    ),
    "suppressed_global": ReIDHeadSpec(
        name="suppressed_global",
        implementation=HeadImplementation.MULTI_BRANCH,
        families=_CSL,
        specialist=True,
        supports_scale_balance=True,
    ),
    "body_slot": ReIDHeadSpec(
        name="body_slot",
        implementation=HeadImplementation.BODY_SLOT,
        families=_CSL,
        supports_scale_balance=True,
    ),
    "lmbn": ReIDHeadSpec(
        name="lmbn",
        implementation=HeadImplementation.LMBN,
        families=_CSL,
        train_selectable=False,
    ),
}

TRAIN_HEAD_TYPES = tuple(
    name for name, spec in REID_HEAD_SPECS.items() if spec.train_selectable
)
MULTI_BRANCH_HEAD_TYPES = frozenset(
    name
    for name, spec in REID_HEAD_SPECS.items()
    if spec.implementation == HeadImplementation.MULTI_BRANCH
)
CSL_SPECIALIST_HEAD_TYPES = frozenset(
    name for name, spec in REID_HEAD_SPECS.items() if spec.specialist
)
CHANNEL_CONTROL_HEAD_TYPES = frozenset(
    name for name, spec in REID_HEAD_SPECS.items() if spec.channel_control
)


def get_reid_head_spec(name: str, *, family: str | None = None) -> ReIDHeadSpec:
    """Resolve one canonical head spec and optionally validate its family."""
    normalized = str(name).lower()
    try:
        spec = REID_HEAD_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported ReID head_type={name!r}; expected one of "
            f"{sorted(REID_HEAD_SPECS)}"
        ) from exc
    if family is not None and not spec.supports_family(family):
        raise ValueError(
            f"head_type={normalized!r} is not implemented for {family}; "
            f"supported families are {sorted(spec.families)}"
        )
    return spec


__all__ = [
    "CHANNEL_CONTROL_HEAD_TYPES",
    "CSL_SPECIALIST_HEAD_TYPES",
    "HeadImplementation",
    "MULTI_BRANCH_HEAD_TYPES",
    "REID_HEAD_SPECS",
    "ReIDHeadSpec",
    "TRAIN_HEAD_TYPES",
    "get_reid_head_spec",
]
