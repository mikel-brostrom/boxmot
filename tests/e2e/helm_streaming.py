"""Install, upgrade, and recover a real Helm deployment in an isolated namespace.

Run ``python -m tests.e2e.helm_streaming --help`` from the repository root.
"""

from __future__ import annotations

import json
import re
from typing import Any

from tests.e2e.support import CAMERA_ID, RELEASE, HelmHarness, wait_for


class HelmTest(HelmHarness):
    """Prove recovery and export correctness before promoting image artifacts."""

    promotable = True
    scenario = "recovery"

    def run_id(self) -> str | None:
        rows = self.sql(f"SELECT run_id FROM boxmot_attributes.runs WHERE camera_id='{CAMERA_ID}' AND state='active'")
        assert len(rows) <= 1, "More than one active camera session"
        return rows[0]["run_id"] if rows else None

    def completed(self, run_id: str) -> bool:
        assert re.fullmatch(r"[a-f0-9]{32}", run_id)
        rows = self.sql(
            f"SELECT payload FROM boxmot_attributes.events WHERE run_id='{run_id}' "
            "AND payload->>'status'='completed' AND delivered_at IS NOT NULL"
        )
        return bool(rows) and all(
            isinstance(row["payload"].get("values"), dict) and row["payload"]["values"] for row in rows
        )

    def ready(self) -> bool:
        result = self.kubectl("get", "deployment", f"{RELEASE}-{CAMERA_ID}", "-o", "json")
        return json.loads(result.stdout).get("status", {}).get("readyReplicas", 0) == 1

    def camera_pods(self) -> list[dict[str, Any]]:
        """Select the camera through its deployment selector, not name substrings."""
        deployment = json.loads(self.kubectl("get", "deployment", f"{RELEASE}-{CAMERA_ID}", "-o", "json").stdout)
        selector = ",".join(f"{key}={value}" for key, value in deployment["spec"]["selector"]["matchLabels"].items())
        return json.loads(self.kubectl("get", "pods", "-l", selector, "-o", "json").stdout)["items"]

    def run(self) -> None:
        """Require attributes, data continuity, source recovery, and pod/DB recovery."""
        self.start()
        run_id = wait_for(self.run_id, "No active camera session")
        wait_for(lambda: self.completed(run_id), "No completed real attributes in PostgreSQL and export", 1800)
        self.mark("Helm install → network fixture → detector/tracker → real VLM → PostgreSQL/export")
        claims = self.pvc_ids()
        before = self.sql(
            "SELECT id,payload,delivered_at FROM boxmot_attributes.events WHERE payload->>'status'='completed'"
        )
        pod_uid = self.camera_pods()[0]["metadata"]["uid"]

        self.values["cameraDefaults"]["config"]["ready_timeout_s"] = 31
        self.install()
        upgraded_run = wait_for(
            lambda: current if (current := self.run_id()) and current != run_id else None,
            "Config upgrade did not start a new session",
        )
        assert self.camera_pods()[0]["metadata"]["uid"] != pod_uid
        assert self.pvc_ids() == claims, "Upgrade replaced persistent storage"
        after = self.sql(
            "SELECT id,payload,delivered_at FROM boxmot_attributes.events WHERE payload->>'status'='completed'"
        )
        assert all(row in after for row in before), "Upgrade changed existing attribute events"
        wait_for(lambda: self.completed(upgraded_run), "No real attributes after Helm upgrade", 1800)
        self.mark("Helm config upgrade restarts camera, preserves PVCs/history, and completes new attributes")

        self.kubectl("scale", "deployment/fixture", "--replicas=0")
        wait_for(lambda: not self.ready(), "Source outage did not withdraw readiness", 90)
        self.kubectl(
            "exec",
            f"deployment/{RELEASE}-{CAMERA_ID}",
            "--",
            "python",
            "-c",
            "import urllib.request; "
            "assert urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=5).status == 200",
        )
        self.kubectl("scale", "deployment/fixture", "--replicas=1")
        wait_for(self.ready, "Camera failed to reconnect", 180)
        reconnected_run = wait_for(
            lambda: current if (current := self.run_id()) and current != upgraded_run else None,
            "Reconnect did not create a fresh session",
        )
        self.mark("Camera disconnect/reconnect changes session without killing supervision")

        pods = self.camera_pods()
        assert len(pods) == 1
        self.kubectl("delete", "pod", pods[0]["metadata"]["name"], "--wait=true")
        wait_for(self.ready, "Camera pod replacement failed", 180)
        replacement_run = wait_for(
            lambda: current if (current := self.run_id()) and current != reconnected_run else None,
            "Pod replacement reused old session",
        )
        self.mark("Camera pod replacement recovers ownership with a fresh session")

        self.kubectl("scale", f"deployment/{RELEASE}-postgres", "--replicas=0")
        wait_for(lambda: not self.ready(), "Database loss did not withdraw camera readiness", 90)
        self.kubectl("scale", f"deployment/{RELEASE}-postgres", "--replicas=1")
        self.kubectl("rollout", "status", f"deployment/{RELEASE}-postgres", "--timeout=180s", timeout=200)
        wait_for(self.ready, "Camera did not recover after database restart", 240)
        recovered_run = wait_for(
            lambda: current if (current := self.run_id()) and current != replacement_run else None,
            "Database recovery did not create a new session",
        )
        wait_for(lambda: self.completed(recovered_run), "No completed attributes after recovery", 1800)
        self.expected_sessions = {run_id, upgraded_run, recovered_run}
        assert self.pvc_ids() == claims
        self.mark("Database restart preserves storage and resumes real attribute generation")

        self.values["cameraDefaults"]["suspend"] = True
        self.install()
        wait_for(lambda: not self.camera_pods(), "Camera did not stop before output verification", 90)
        for source, name in (
            (f"{RELEASE}-{CAMERA_ID}-out-gc", "output-maintenance"),
            (f"{RELEASE}-db-gc", "database-maintenance"),
        ):
            self.kubectl("create", "job", name, f"--from=cronjob/{source}")
            self.kubectl("wait", "--for=condition=complete", f"job/{name}", "--timeout=300s", timeout=320)
        self.verify_outputs()
        self.mark("Maintenance jobs complete; exported tracks/attributes match persisted outcomes")
        self.command(
            [
                "helm",
                "uninstall",
                RELEASE,
                "--kube-context",
                self.args.context,
                "--namespace",
                self.namespace,
                "--wait",
                "--timeout",
                "3m",
            ],
            timeout=200,
        )
        assert self.pvc_ids() == claims, "Uninstall deleted retained PVCs"
        self.mark("Helm uninstall retains every data/model PVC")

    def verify_outputs(self) -> None:
        """Require exact JSONL/database correspondence, allowing consistent replay IDs."""
        events = self.sql("SELECT run_id,id,payload,delivered_at FROM boxmot_attributes.events ORDER BY id")
        self.volume_pod("output-reader", f"{RELEASE}-{CAMERA_ID}-outputs", "/outputs")
        script = (
            "from pathlib import Path; import json; print(json.dumps([json.loads(line) "
            "for path in Path('/outputs').rglob('*.jsonl*') "
            "for line in path.read_text().splitlines() if line.strip()]))"
        )
        rows = json.loads(self.kubectl("exec", "output-reader", "--", "python", "-c", script).stdout)
        self.kubectl("delete", "pod", "output-reader", "--wait=true")
        tracks = [row for row in rows if "tracks" in row]
        assert any(row["tracks"] for row in tracks), "No detector/tracker output"
        assert self.expected_sessions <= {row["session_id"] for row in tracks if row["tracks"]}, (
            "Missing tracks for a successful attribute session"
        )
        exported: dict[str, dict[str, Any]] = {}
        for row in rows:
            if "event_id" not in row:
                continue
            event_id = row["event_id"]
            assert event_id not in exported or exported[event_id] == row, "Conflicting event replay"
            exported[event_id] = row
        completed = [row for row in events if row["payload"].get("status") == "completed"]
        assert completed, "No completed persisted attribute events"
        for event in events:
            assert event["delivered_at"] is not None, "Stopped fixture retained unexported events"
            event_id = f"{event['run_id']}:{event['id']}"
            assert event_id in exported, f"Missing event {event_id}"
            row = exported[event_id]
            assert row["camera_id"] == CAMERA_ID and row["session_id"] == event["run_id"]
            payload = {
                key: value
                for key, value in row.items()
                if key not in {"event_id", "camera_id", "session_id", "created_at"}
            }
            assert payload == event["payload"], f"Changed event {event_id}"
        assert set(exported) == {f"{event['run_id']}:{event['id']}" for event in events}
        summary = {
            "track_frames": len(tracks),
            "persisted_events": len(events),
            "completed_events": len(completed),
            "verified_events": len(exported),
            "session_ids": sorted({row["session_id"] for row in tracks}),
        }
        (self.directory / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    from tests.e2e.support import argument_parser, run_scenario

    args = argument_parser(__doc__).parse_args()
    run_scenario(HelmTest(args))


if __name__ == "__main__":
    main()
