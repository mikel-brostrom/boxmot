import json

import pytest

from boxmot.reid.backbones.head_registry import HeadImplementation
from boxmot.reid.training.ablation import ActivationKind, AddonCategory
from boxmot.reid.training.model_options import OptionTransform


@pytest.mark.parametrize(
    ("member", "value"),
    [
        (HeadImplementation.MULTI_BRANCH, "multi_branch"),
        (OptionTransform.POSITIVE, "positive"),
        (AddonCategory.ARCHITECTURE, "architecture"),
        (ActivationKind.NOT_EQUALS, "not_equals"),
    ],
)
def test_declarative_enums_preserve_string_semantics(member, value):
    assert isinstance(member, str)
    assert member == value
    assert str(member) == value
    assert f"{member}" == value
    assert json.dumps(member) == f'"{value}"'
