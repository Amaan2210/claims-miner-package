"""Exercise the built sample wheel inside the Claims disposable worker harness.

Usage: python tests/worker_harness_smoke.py --wheel PATH --wheelhouse DIR
The wheelhouse must contain offline wheels for pydantic, pypdf and their
transitive dependencies for the worker Python version and platform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parent / "Claims"))

from competition.worker_bundle import build_worker_bundle
from competition.worker_contract import build_worker_contract
from competition.worker_guest import run_bundle
from test_package import valid_artifact, write_pdf


class Gateway(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append({"path": self.path, "authorization": self.headers.get("Authorization"), "body": body})
        payload = json.dumps(valid_artifact())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in (payload[:50], payload[50:]):
            event = {"choices": [{"delta": {"content": piece}}]}
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *_args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--wheelhouse", required=True, type=Path)
    args = parser.parse_args()
    Gateway.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_pdf(root / "paper.pdf")
            manifest = json.loads((REPO / "examples/competition_manifest.json").read_text())
            package = args.wheel.read_bytes()
            pdf = (root / "paper.pdf").read_bytes()
            job = {
                "job_id": "a" * 64, "round_id": "pilot", "hotkey": "sample-miner",
                "package_sha256": hashlib.sha256(package).hexdigest(),
                "paper_id": "example-001", "lease_token": "local-smoke",
                "paper": {"paper_id": "example-001", "title": "Example paper"},
                "manifest": manifest,
            }
            contract = build_worker_contract(
                job, allowed_models=("example/allowed-model",),
                max_output_tokens=8192, paper_token_limit=1000, dollar_limit_usd=0.10,
            )
            bundle = build_worker_bundle(
                job={**job, "worker_contract": contract,
                     "expected_pdf_sha256": hashlib.sha256(pdf).hexdigest()},
                package=package, pdf=pdf,
            )
            input_dir = root / "input"
            input_dir.mkdir()
            for filename, content in bundle.items():
                (input_dir / filename).write_bytes(content)
            key_file = root / "paper-key"
            key_file.write_text("paper-scoped-test-token")
            result_dir = root / "result"
            result = run_bundle(
                input_dir=input_dir, result_dir=result_dir,
                wheelhouse=args.wheelhouse,
                inference_base_url=f"http://127.0.0.1:{server.server_port}/v1",
                paper_key_file=key_file,
            )
            assert result["status"] == "completed", (result, (result_dir / "worker.log").read_text())
            output = result["artifact"]
            assert output["logic"]["claims"][0]["statement"] == (
                "The treatment group had a higher response rate than the control group."
            )
            assert len(Gateway.requests) == 1
            assert Gateway.requests[0]["path"] == "/v1/chat/completions"
            assert Gateway.requests[0]["authorization"] == "Bearer paper-scoped-test-token"
            assert Gateway.requests[0]["body"]["stream"] is True
            assert Gateway.requests[0]["body"]["model"] == "example/allowed-model"
            logs = (result_dir / "worker.log").read_text()
            assert "[install] exit=0" in logs
            assert "[validate] agent_v1 output accepted" in logs
            assert "paper-scoped-test-token" not in logs
            print("worker package smoke passed: offline install, PDF extraction, gateway stream, agent_v1 validation")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
