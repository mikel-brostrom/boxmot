import pytest

from boxmot.core.box_schema import (
    AABB_SCHEMA,
    OBB_SCHEMA,
    schema_from_cache_columns,
    schema_from_detection_columns,
    schema_from_frame_tagged_track_columns,
    schema_from_mot_columns,
    schema_from_track_columns,
)


@pytest.mark.parametrize(
    ("resolver", "aabb_columns", "obb_columns"),
    [
        (schema_from_detection_columns, 6, 7),
        (schema_from_track_columns, 8, 9),
        (schema_from_cache_columns, 7, 8),
        (schema_from_frame_tagged_track_columns, 9, 10),
        (schema_from_mot_columns, 9, 13),
    ],
)
def test_schema_resolvers_share_one_aabb_obb_contract(resolver, aabb_columns, obb_columns):
    assert resolver(aabb_columns) is AABB_SCHEMA
    assert resolver(obb_columns) is OBB_SCHEMA


@pytest.mark.parametrize(
    "resolver",
    [
        schema_from_detection_columns,
        schema_from_track_columns,
        schema_from_cache_columns,
        schema_from_frame_tagged_track_columns,
        schema_from_mot_columns,
    ],
)
def test_schema_resolvers_reject_noncanonical_widths(resolver):
    with pytest.raises(ValueError, match="column count"):
        resolver(12)
