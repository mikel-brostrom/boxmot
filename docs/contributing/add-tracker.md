# Add a Tracker

To integrate a new tracker cleanly:

1. Add `boxmot/trackers/<name>/tracker.py` and keep each `__init__.py` free of
   tracker-class re-exports. Declare `TrackerFamily` from the tracker's primary
   state: `box` for AABB/OBB, `mask` for mask state, or `multimodal` when several
   representations or model memory are fundamental. This family is capability
   metadata; all algorithm packages live directly under `boxmot/trackers/`.
2. Box-state implementations subclass `BoxTracker` from
   `boxmot.trackers.common.box.base`; other trackers subclass `BaseTracker`
   from `boxmot.trackers.common.base`. Declare immutable
   capabilities and the configuration-dependent `use_embeddings`,
   `_requires_frame`, and `_requires_masks` settings, then implement
   `_track_detections()`. The inherited
   `TrackerRequirements` property exposes those requirements to pipelines.
   Keep the inherited public
   overloaded `update()` entry point: `Detections` input returns `Tracks`, and
   packed NumPy input returns packed NumPy track rows. `BaseTracker` owns
   validation and conversion of both representations; concrete trackers
   continue to implement only their private NumPy kernel. A ReID-enabled
   high-level tracker adapter must expose and forward the shared `reid_model`,
   `reid_weights`, `device`, `half`, and `reid_preprocess` constructor options
   rather than implementing its own extraction path. The shared tracker-domain
   appearance helper owns lazy extraction for missing embeddings, the
   precomputed bypass, and empty-batch behavior for Python and native adapters.
3. Add the tracker key and canonical implementation path to `_TRACKER_MANIFEST`
   in `boxmot/trackers/common/manifest.py`, then add its static capability declaration
   to `boxmot/trackers/common/registry.py`. Public exports and exact class
   identities derive from the manifest; tests require registry and
   implementation capabilities to agree.
4. Import the class in application examples with
   `from boxmot import <TrackerClass>`; implementation packages do not provide
   parallel class aliases.
5. Add `boxmot/configs/trackers/<name>.yaml` with each parameter's runtime default and tuning metadata.
6. Add a tracker doc page and wire it into `mkdocs.yml`.
7. Extend registry/package tests under `tests/unit/trackers/`, add focused
   algorithm tests under `tests/unit/trackers/<family>/` when useful, and update
   the tracker lists in `tests/test_config.py` where applicable.
8. Update the tracker, ReID, mask/OBB, and benchmark lists in
   `.github/workflows/` when the new tracker should run in those jobs.

Reuse shared motion adapters from `boxmot.trackers.common.motion.models`
and filters from `boxmot.trackers.common.motion.kalman_filters`. Construct
camera-motion estimators through `boxmot.trackers.common.motion.cmc.registry`
and apply or reset them through `cmc.integration`. Keep shared motion tests
under `tests/unit/trackers/common/motion/`.

## Optional native C++ backend

If the tracker also gets a native backend:

The backend does not change the tracker's representation family. Keep C++
implementation and ABI code under `boxmot/native/cpp`; only the canonical
Python adapter lives beside the algorithm's tracker implementation.

1. Add native sources under `boxmot/native/cpp/trackers/<name>/`.
2. Add the tracker subdirectory and wheel-install entries to
   `boxmot/native/cpp/CMakeLists.txt`.
3. Add the low-level ctypes binding under
   `boxmot/native/trackers/<name>.py`. It must accept and return typed,
   contiguous NumPy buffers and must not import structures or tracker code.
4. Add the public adapter under `boxmot/trackers/<name>/native.py`.
   The shared adapter converts `Detections` or packed detection rows plus an
   optional `Frame`, returns `Tracks` or packed rows to match the input,
   resolves tracker configuration, and declares requirements.
5. Set `native_class_path` on the algorithm's `_TRACKER_MANIFEST` entry. Native
   validation and construction are owned by `boxmot/trackers/common/factory.py`.
6. Document `--tracker-backend cpp` support on the tracker page.
7. Add low-level ABI and domain-adapter tests under
   `tests/unit/native/trackers/test_native_<name>.py`.

Native tracker sources should follow the existing CMake layout: a
`<name>_capi` shared library exposing typed ABI v2 buffers and a `<name>_core`
target for reusable C++ code. A state-mutating update is called exactly once;
the library allocates its result and the caller releases it with the matching
free function. Do not add positional-cache or replay executables. Evaluation
and tuning stream keyed records through the same live ABI.

Do not place model inference code under `native/cpp/trackers` or link it into a
tracker target. The optional native appearance runtime is independently owned
by `native/cpp/reid`; trackers receive embeddings through their typed input
buffers. A high-level native adapter may implement the shared tracker-owned
live-extraction fallback, but model settings remain separate from
`TrackerSpec` and the low-level tracker ABI.

## Minimum checklist

- tracker implementation
- tracker registration
- tracker YAML
- docs page
- tests
- workflow matrices if benchmarked in CI
- native C++ registration and tests if a native backend is added
