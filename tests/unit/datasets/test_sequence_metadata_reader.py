from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from boxmot.datasets import CachedVisionDataset


@pytest.mark.parametrize("write_statistics", [False, True])
@pytest.mark.parametrize(
    ("split", "expected"),
    [(None, ("sample-a", "sample-b")), ("train", ("sample-a",)), ("validation", ("sample-b",))],
)
def test_sequence_metadata_reader_preserves_exact_split_keys_and_order(
    materialized_build, split, expected, write_statistics
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = materialized_build["root"]
    # A sequence can exist in different splits. Reverse its physical samples
    # and detections independently, including when statistics are unavailable.
    for artifact in ("samples", "instances"):
        for path in sorted((root / artifact).glob("*.parquet")):
            table = pq.ParquetFile(path).read()
            rows = list(reversed(table.to_pylist()))
            if artifact == "samples":
                for row in rows:
                    row["sequence_id"] = "shared-sequence"
            pq.write_table(
                pa.Table.from_pylist(rows, schema=table.schema),
                path,
                compression="zstd",
                row_group_size=1,
                write_statistics=write_statistics,
            )

    dataset = CachedVisionDataset._for_sequence(root, sequence_id="shared-sequence", split=split)

    assert dataset.sample_ids == expected
    for sample in dataset:
        assert sample.sequence_id == "shared-sequence"
        if split is not None:
            assert sample.split == split
        expected_count = 2 if sample.sample_id == "sample-a" else 1
        assert sample.detections.instance_ids == tuple(
            f"{dataset.manifest.build_id}:{sample.sample_id}:{index}" for index in range(expected_count)
        )


@pytest.mark.parametrize("artifact", ["samples", "instances"])
def test_sequence_metadata_reader_checks_empty_shard_schemas(materialized_build, artifact) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = materialized_build["root"]
    pq.write_table(
        pa.table({"unexpected": pa.array([], type=pa.int64())}),
        root / artifact / "part-empty.parquet",
        compression="zstd",
    )

    with pytest.raises(ValueError, match="Parquet schema mismatch"):
        CachedVisionDataset._for_sequence(root, sequence_id="seq-a", split="train")


def test_sequence_metadata_reader_does_not_initialize_dataset_or_pandas(materialized_boxes_only_build) -> None:
    code = textwrap.dedent(
        """
        import sys

        from boxmot.datasets import CachedVisionDataset

        dataset = CachedVisionDataset._for_sequence(
            sys.argv[1], sequence_id="seq-a", split="train"
        )
        assert dataset.sample_ids == ("sample-a",)
        assert len(dataset[0].detections) == 2
        assert "pyarrow.dataset" not in sys.modules
        assert "pandas" not in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(materialized_boxes_only_build)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
