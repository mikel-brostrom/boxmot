"""Measure moving-video throughput for multiple cameras without recovery injections.

Run separately from the image-promotion recovery gate. Measurements describe the
selected fixture, resources, image, and duration; they are not production sizing.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from tests.e2e.support import CHART, RELEASE, ROOT, HelmHarness, wait_for


def summarize(samples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute rates only across continuous worker processes, using observed wall time."""
    if len(samples) < 2:
        raise ValueError("Capacity measurement requires at least two samples")
    first, last = samples[0], samples[-1]
    if not first["cameras"] or any(sample["cameras"].keys() != first["cameras"].keys() for sample in samples):
        raise ValueError("Capacity samples must contain the same nonempty camera set")
    summary = {}
    for camera, initial in first["cameras"].items():
        observations = [sample["cameras"][camera] for sample in samples]
        if any(
            item["pod_uid"] != initial["pod_uid"]
            or item["restarts"] != initial["restarts"]
            or item["reconnects"] != initial["reconnects"]
            for item in observations
        ):
            raise AssertionError(f"{camera} restarted or reconnected during measurement; use the recovery scenario")
        for previous, current in zip(observations, observations[1:]):
            if any(current[key] < previous[key] for key in ("frames", "dropped_frames", "completed_events")):
                raise AssertionError(f"{camera} counters regressed")
            if any(
                not math.isfinite(current[key] - previous[key]) or current[key] <= previous[key]
                for key in ("observed_at_s", "attributes_observed_at_s")
            ):
                raise ValueError("Capacity samples need positive finite observation intervals")
        final = last["cameras"][camera]
        seconds = final["observed_at_s"] - initial["observed_at_s"]
        attribute_seconds = final["attributes_observed_at_s"] - initial["attributes_observed_at_s"]
        processed = final["frames"] - initial["frames"]
        dropped = final["dropped_frames"] - initial["dropped_frames"]
        completed = final["completed_events"] - initial["completed_events"]
        summary[camera] = {
            "seconds": seconds,
            "processed_frames": processed,
            "processed_fps": processed / seconds,
            "dropped_frames": dropped,
            "dropped_fraction": dropped / max(1, processed + dropped),
            "completed_attribute_events": completed,
            "attribute_seconds": attribute_seconds,
            "attribute_events_per_second": completed / attribute_seconds,
            "peak_pending_batches": max(item["pending_batches"] for item in observations),
            "peak_queued_crop_bytes": max(item["queued_crop_bytes"] for item in observations),
            "peak_oldest_batch_age_s": max(item["oldest_batch_age_s"] for item in observations),
            "peak_mean_frame_ms": max(item["mean_frame_ms"] for item in observations),
        }
    return summary


class CapacityTest(HelmHarness):
    """Run simultaneous independent camera workers against a looping moving fixture."""

    scenario = "capacity"

    def prepare(self) -> None:
        super().prepare()
        source = self.args.fixture_frames or ROOT / "assets/MOT17-mini/train/MOT17-04-FRCNN/img1"
        frames = sorted(source.glob("*.jpg"))
        if len(frames) < 2:
            raise ValueError("A moving fixture requires at least two JPEG frames")
        directory = self.directory / "frames"
        directory.mkdir()
        for index, frame in enumerate(frames):
            self.command(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(frame),
                    "-vf",
                    "scale=640:-2",
                    "-frames:v",
                    "1",
                    "-threads",
                    "1",
                    str(directory / f"{index:06d}.jpg"),
                ]
            )
            # Keep fixture payload below the Kubernetes ConfigMap object limit.
            if sum(path.stat().st_size for path in directory.glob("*.jpg")) > 650_000:
                raise ValueError("Fixture exceeds 650 KB; provide a shorter JPEG sequence")
        self.values["cameras"] = {
            f"camera-{number:02d}": {"sourceSecret": {"name": "test-camera", "key": f"camera-{number:02d}"}}
            for number in range(1, self.args.cameras + 1)
        }
        path = self.directory / "values.json"
        path.write_text(json.dumps(self.values, indent=2) + "\n")
        self.command(["helm", "template", RELEASE, str(CHART), "--namespace", self.namespace, "--values", str(path)])
        fixture = {
            "fps": 5,
            "loop": True,
            "frames": [
                {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in self.fixture_images()
            ],
        }
        if len({frame["sha256"] for frame in fixture["frames"]}) < 2:
            raise ValueError("A moving fixture requires at least two distinct rendered frames")
        (self.directory / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
        self.mark(f"Moving fixture and {self.args.cameras} independent camera configurations rendered")

    def fixture_images(self) -> list[Path]:
        return sorted((self.directory / "frames").glob("*.jpg"))

    def snapshot(self, started: float) -> dict[str, Any]:
        """Read health, immutable pod identity, and persisted nonempty outcomes for each camera."""
        events = {
            row["camera_id"]: row["completed"]
            for row in self.sql(
                "SELECT r.camera_id, COUNT(*) AS completed FROM boxmot_attributes.events e "
                "JOIN boxmot_attributes.runs r ON r.run_id=e.run_id "
                "WHERE e.payload->>'status'='completed' AND e.delivered_at IS NOT NULL "
                "AND jsonb_typeof(e.payload->'values')='object' AND e.payload->'values' <> '{}'::jsonb "
                "GROUP BY r.camera_id"
            )
        }
        attributes_observed_at = time.monotonic() - started
        cameras = {}
        for camera in self.values["cameras"]:
            deployment = json.loads(self.kubectl("get", "deployment", f"{RELEASE}-{camera}", "-o", "json").stdout)
            selector = ",".join(
                f"{key}={value}" for key, value in deployment["spec"]["selector"]["matchLabels"].items()
            )
            pods = json.loads(self.kubectl("get", "pods", "-l", selector, "-o", "json").stdout)["items"]
            assert len(pods) == 1, f"{camera} must have exactly one worker"
            script = ("import urllib.request; "
                      "print(urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=5).read().decode())")
            state = json.loads(self.kubectl("exec", pods[0]["metadata"]["name"], "--", "python", "-c", script).stdout)
            assert state["ready"], f"{camera} lost readiness during measurement"
            state.update(
                pod_uid=pods[0]["metadata"]["uid"],
                restarts=sum(item["restartCount"] for item in pods[0]["status"]["containerStatuses"]),
                completed_events=events.get(camera, 0),
                observed_at_s=time.monotonic() - started,
                attributes_observed_at_s=attributes_observed_at,
            )
            cameras[camera] = state
        return {"elapsed_s": time.monotonic() - started, "cameras": cameras}

    def run(self) -> None:
        self.start()

        def warmed() -> bool:
            rows = self.sql(
                "SELECT DISTINCT r.camera_id FROM boxmot_attributes.events e "
                "JOIN boxmot_attributes.runs r ON r.run_id=e.run_id "
                "WHERE e.payload->>'status'='completed' AND e.delivered_at IS NOT NULL "
                "AND jsonb_typeof(e.payload->'values')='object' AND e.payload->'values' <> '{}'::jsonb"
            )
            return set(self.values["cameras"]) <= {row["camera_id"] for row in rows}

        wait_for(warmed, "Every camera must complete real attributes before measurement", self.args.warmup_timeout)
        self.mark("Every camera completed and exported real attributes during warmup")
        started = time.monotonic()
        samples = [self.snapshot(started)]
        deadline = time.monotonic() + self.args.duration
        path = self.directory / "samples.jsonl"
        with path.open("w") as output:
            output.write(json.dumps(samples[0]) + "\n")
            output.flush()
            while time.monotonic() < deadline:
                time.sleep(min(self.args.interval, max(0, deadline - time.monotonic())))
                samples.append(self.snapshot(started))
                output.write(json.dumps(samples[-1]) + "\n")
                output.flush()
                print(f"Measured {samples[-1]['elapsed_s']:.0f}s across {self.args.cameras} cameras", flush=True)
        summary = summarize(samples)
        report = {
            "camera_image": self.args.image,
            "external_vllm": bool(self.args.external_vllm_url),
            "vllm_image": None if self.args.external_vllm_url else self.values["vllm"]["image"],
            "model_revision": None if self.args.external_vllm_url else self.defaults["vllm"]["revision"],
            "cameras": summary,
            "thresholds": {"min_fps": self.args.min_fps, "max_queue_age_s": self.args.max_queue_age},
        }
        (self.directory / "capacity.json").write_text(json.dumps(report, indent=2) + "\n")
        # Resource evidence is optional when metrics-server is absent.
        self.kubectl("top", "pods", check=False)
        for camera, values in summary.items():
            assert values["processed_frames"] > 0, f"{camera} made no frame progress"
            assert values["completed_attribute_events"] > 0, f"{camera} completed no attributes during measurement"
            assert values["processed_fps"] >= self.args.min_fps, f"{camera} is below --min-fps"
            assert values["peak_oldest_batch_age_s"] <= self.args.max_queue_age, f"{camera} exceeded --max-queue-age"
        self.mark("All cameras progressed within the selected capacity thresholds")


def main() -> None:
    from tests.e2e.support import argument_parser, run_scenario

    parser = argument_parser(__doc__)
    parser.add_argument("--cameras", type=int, default=2)
    parser.add_argument("--duration", type=float, default=300, help="Measurement seconds after all cameras warm up.")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--warmup-timeout", type=float, default=1800)
    parser.add_argument(
        "--fixture-frames", type=Path, help="Bounded JPEG clip; default is bundled moving MOT17 frames."
    )
    parser.add_argument("--min-fps", type=float, default=0)
    parser.add_argument("--max-queue-age", type=float, default=600)
    args = parser.parse_args()
    if not all(math.isfinite(value) for value in (
        args.duration, args.interval, args.warmup_timeout, args.min_fps, args.max_queue_age,
    )):
        parser.error("Timing and capacity thresholds must be finite")
    if not 1 <= args.cameras <= 32 or not 0 < args.interval <= 60 or args.duration <= args.interval:
        parser.error("Use 1–32 cameras, a 0–60 second sampling interval, and a longer duration")
    if args.warmup_timeout <= 0 or args.min_fps < 0 or args.max_queue_age <= 0:
        parser.error("Warmup/queue timeouts must be positive and minimum FPS cannot be negative")
    run_scenario(CapacityTest(args))


if __name__ == "__main__":
    main()
