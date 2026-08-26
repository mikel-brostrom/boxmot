import pytest
import torch

from boxmot.reid.datasets.transforms import (
    apply_independent_random_erasing,
    cross_camera_same_id_part_mosaic,
)


def _constant_pk_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = torch.stack(
        [torch.full((3, 40, 20), float(index)) for index in range(8)]
    )
    pids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    camera_ids = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    return images, pids, camera_ids


def test_same_id_part_mosaic_preserves_identity_and_unaltered_fraction():
    images, pids, camera_ids = _constant_pk_batch()
    torch.manual_seed(7)

    output = cross_camera_same_id_part_mosaic(
        images,
        pids,
        camera_ids,
        probability=1.0,
        cross_camera_rate=1.0,
        min_unaltered_fraction=0.5,
    )

    changed = (output != images).flatten(1).any(dim=1)
    assert 0 < int(changed.sum()) <= 4
    assert int((~changed).sum()) >= 4

    for anchor_index in torch.nonzero(changed).flatten().tolist():
        donor_values = torch.unique(output[anchor_index][output[anchor_index] != anchor_index])
        for donor_value in donor_values.tolist():
            donor_index = int(donor_value)
            assert pids[donor_index] == pids[anchor_index]
            assert camera_ids[donor_index] != camera_ids[anchor_index]

        replaced_fraction = float((output[anchor_index] != anchor_index).float().mean())
        assert 0.13 <= replaced_fraction <= 0.42


def test_same_id_part_mosaic_is_torch_seed_reproducible():
    images, pids, camera_ids = _constant_pk_batch()

    torch.manual_seed(19)
    first = cross_camera_same_id_part_mosaic(
        images,
        pids,
        camera_ids,
        probability=1.0,
    )
    torch.manual_seed(19)
    second = cross_camera_same_id_part_mosaic(
        images,
        pids,
        camera_ids,
        probability=1.0,
    )

    assert torch.equal(first, second)


def test_same_id_part_mosaic_rejects_invalid_labels():
    images, pids, camera_ids = _constant_pk_batch()

    with pytest.raises(ValueError, match="one-dimensional"):
        cross_camera_same_id_part_mosaic(images, pids[:, None], camera_ids)


def test_batch_random_erasing_is_independent_and_seed_reproducible():
    images = torch.ones(4, 3, 40, 20)

    torch.manual_seed(31)
    first = apply_independent_random_erasing(images, probability=1.0)
    torch.manual_seed(31)
    second = apply_independent_random_erasing(images, probability=1.0)

    assert torch.equal(first, second)
    erased = first == 0
    assert erased.flatten(1).any(dim=1).all()
    assert not torch.equal(erased[0], erased[1])
