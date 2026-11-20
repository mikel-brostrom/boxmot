"""Rendered stream settings must pass the worker's authored config boundary."""

import pytest
import yaml

from boxmot.utils.config import ConfigurationError
from tests.ci.helm_contract import check_stream_configs


@pytest.mark.parametrize(
    "override,match",
    [
        ({"capture_timeout_s": -1}, "capture_timeout_s"),
        ({"timestamp_source": "fps"}, "timestamp_source"),
        ({"tracking": {"show": True}}, "Unsupported streaming tracking options"),
        ({"typo_in_generated_config": True}, "Unknown stream configuration fields"),
        ({"attributes": {"model": "fake", "prompt": "test", "schema": {"type": "array"}}}, "object root"),
    ],
)
def test_render_contract_rejects_yaml_that_runtime_cannot_load(override, match):
    config = {"camera_id": "door", "attributes_host": "http://vllm:8000", **override}
    objects = [{"kind": "ConfigMap", "data": {"stream.yaml": yaml.safe_dump(config)}}]
    with pytest.raises(ConfigurationError, match=match):
        check_stream_configs(objects)
