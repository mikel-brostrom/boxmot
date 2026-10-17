"""Sequence workers must not multiply the tuning actor's native thread budget."""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
import pytest
import torch
from scipy.linalg.blas import dgemm
from threadpoolctl import threadpool_info

from boxmot.engine.eval import replay
from boxmot.engine.eval.session import ReplaySession

_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _initialize_with_loaded_libraries(progress_queue: Any) -> None:
    """Import this module's native libraries before the real worker initializer."""
    replay._initialize_replay_worker(progress_queue)


def _thread_state(_: object) -> dict[str, Any]:
    """Exercise each native library and report its effective thread limits."""
    matrix = np.arange(256, dtype=np.float64).reshape(16, 16)
    matrix @ matrix
    dgemm(alpha=1.0, a=matrix, b=matrix)
    torch.mm(torch.from_numpy(matrix), torch.from_numpy(matrix))
    cv2.GaussianBlur(matrix, (3, 3), 0)
    return {
        "pid": os.getpid(),
        "environment": {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT},
        "native_pools": tuple(
            sorted((pool["user_api"], pool["prefix"], pool["num_threads"]) for pool in threadpool_info())
        ),
        "torch_intraop": torch.get_num_threads(),
        "torch_interop": torch.get_num_interop_threads(),
        "opencv": cv2.getNumThreads(),
    }


@pytest.mark.parametrize("preload_libraries", [False, True], ids=["normal-startup", "already-loaded-libraries"])
def test_replay_worker_caps_native_threads_without_changing_parent(
    monkeypatch: pytest.MonkeyPatch, preload_libraries: bool
) -> None:
    """A retained worker gets one native thread even when its actor owns seven."""
    for name in _THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "7")
    if preload_libraries:
        monkeypatch.setattr(replay, "_initialize_replay_worker", _initialize_with_loaded_libraries)
    parent_before = _thread_state(None)

    with ReplaySession(1) as session:
        first = session.map(_thread_state, [None])[0]
        second = session.map(_thread_state, [None])[0]

    assert first == second
    assert first["pid"] != os.getpid()
    assert first["environment"] == dict.fromkeys(_THREAD_ENVIRONMENT, "1")
    assert first["torch_intraop"] == first["torch_interop"] == first["opencv"] == 1
    assert any(api == "blas" for api, _prefix, _threads in first["native_pools"])
    assert all(threads == 1 for _api, _prefix, threads in first["native_pools"]), first["native_pools"]
    assert _thread_state(None) == parent_before
