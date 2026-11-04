import json
import io
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from claims_sample_miner.cli import InferenceError, completion, ingest_pdf, run
from claims_sample_miner.smoke_baseline import run as run_smoke_baseline

PAPER_TEXT = "The treatment group had a higher response rate than the control group."


def write_pdf(path: Path) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)}),
    })
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 50 700 Td ({PAPER_TEXT}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as handle:
        writer.write(handle)


def valid_artifact() -> dict:
    source = {
        "source_id": "example-001-span-0001", "source_type": "span",
        "span_ids": ["example-001-span-0001"], "quote": PAPER_TEXT, "role": "result",
    }
    return {
        "ara_version": "1.0",
        "paper": {"paper_id": "example-001", "title": "Example paper"},
        "logic": {
            "claims": [{
                "claim_id": "C01", "statement": PAPER_TEXT,
                "conditions": "In the study population", "status": "supported",
                "falsification_criteria": "A lower response rate in the treatment group",
                "proof": ["E01"], "evidence_ids": ["EV01"], "sources": [source],
            }],
            "experiments": [{
                "experiment_id": "E01", "title": "Response comparison", "verifies": ["C01"],
                "setup": "Two groups", "procedure": "Compare responses",
                "expected_outcome": "Higher treatment response", "evidence_ids": ["EV01"],
                "source_refs": [source],
            }],
        },
        "evidence": {"records": [{
            "evidence_id": "EV01", "title": "Reported response comparison",
            "role": "support", "summary": PAPER_TEXT, "source_refs": [source],
            "linked_claim_ids": ["C01"],
        }]},
        "trace": {"node_id": "root", "node_type": "extraction", "summary": "Paper result", "source_refs": [source]},
        "src": {"environment": ["claims-sample-miner"], "artifacts": []},
    }


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def __iter__(self):
        payload = json.dumps(valid_artifact())
        for piece in (payload[:40], payload[40:]):
            event = {"choices": [{"delta": {"content": piece}}]}
            yield ("data: " + json.dumps(event) + "\n").encode()
        yield b"data: [DONE]\n"


def fake_urlopen(request, timeout):
    body = json.loads(request.data)
    assert request.full_url == "http://gateway.internal:8080/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer paper-scoped-test-token"
    assert body["stream"] is True
    assert body["model"] == "openrouter/example/allowed-model"
    assert timeout == 1800
    return FakeResponse()


class PackageTests(unittest.TestCase):
    def test_wrapped_upstream_errors_are_classified_without_logging_response_content(self):
        for upstream, expected_retryable in [(504, True), (429, True), (400, False), (401, False)]:
            with self.subTest(upstream=upstream), patch.dict(os.environ, {
                "CLAIMS_INFERENCE_BASE_URL": "https://gateway.example/v1",
                "CLAIMS_INFERENCE_TOKEN": "private-paper-key",
            }):
                error = HTTPError("https://gateway.example/v1/chat/completions", 400, "Bad Request", {},
                    io.BytesIO(json.dumps({"error": {"code": str(upstream),
                              "message": "sensitive request private-paper-key"}}).encode()))
                with patch("claims_sample_miner.cli.urlopen", side_effect=error):
                    with self.assertRaises(InferenceError) as raised:
                        completion(model="qwen/qwen3-32b", messages=[], stream=True, max_tokens=8192)
                self.assertEqual(raised.exception.retryable, expected_retryable)
                self.assertIn(f"upstream {upstream}", str(raised.exception))
                self.assertNotIn("sensitive", str(raised.exception))
                self.assertNotIn("private-paper-key", str(raised.exception))

    def test_testnet_smoke_baseline_emits_one_exact_pdf_claim_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_pdf(root / "paper.pdf")
            request = root / "request.json"
            output = root / "output.json"
            request.write_text(json.dumps({
                "schema_version": "claims_competition_request_v1",
                "paper": {"paper_id": "example-001", "title": "Example paper"},
                "pdf_path": "paper.pdf",
                "inference": {"model": "example/allowed-model"},
            }))
            run_smoke_baseline(request, output)
            artifact = json.loads(output.read_text())
            self.assertEqual(len(artifact["logic"]["claims"]), 1)
            self.assertEqual(artifact["logic"]["claims"][0]["sources"][0]["quote"], PAPER_TEXT)

    def test_cli_extracts_pdf_through_openai_compatible_http_endpoint(self):
        observed = []

        class Gateway(BaseHTTPRequestHandler):
            def do_POST(self):
                observed.append({
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                })
                payload = json.dumps(valid_artifact())
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for piece in (payload[:50], payload[50:]):
                    event = {"choices": [{"delta": {"content": piece}}]}
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_pdf(root / "paper.pdf")
                (root / "request.json").write_text(json.dumps({
                    "schema_version": "claims_competition_request_v1",
                    "paper": {"paper_id": "example-001", "title": "Example paper"},
                    "pdf_path": "paper.pdf",
                    "inference": {"model": "example/allowed-model", "stream": True},
                }))
                env = {
                    **os.environ,
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
                    "CLAIMS_INFERENCE_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                    "CLAIMS_INFERENCE_TOKEN": "paper-scoped-test-token",
                }
                result = subprocess.run(
                    [sys.executable, "-m", "claims_sample_miner.cli",
                     "--request", str(root / "request.json"),
                     "--output", str(root / "output.json")],
                    env=env, capture_output=True, text=True, timeout=15,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads((root / "output.json").read_text())["logic"]["claims"][0]["claim_id"], "C01")
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0]["path"], "/v1/chat/completions")
            self.assertEqual(observed[0]["authorization"], "Bearer paper-scoped-test-token")
            self.assertEqual(observed[0]["body"]["model"], "openrouter/example/allowed-model")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_real_pdf_extraction_and_gateway_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_pdf(root / "paper.pdf")
            self.assertIn(PAPER_TEXT, PdfReader(str(root / "paper.pdf")).pages[0].extract_text())
            request = {
                "schema_version": "claims_competition_request_v1",
                "paper": {"paper_id": "example-001", "title": "Example paper"},
                "pdf_path": "paper.pdf",
                "inference": {"model": "example/allowed-model", "stream": True},
            }
            (root / "request.json").write_text(json.dumps(request))
            output = root / "agent_output.json"
            with patch.dict(os.environ, {
                "CLAIMS_INFERENCE_BASE_URL": "http://gateway.internal:8080/v1",
                "CLAIMS_INFERENCE_TOKEN": "paper-scoped-test-token",
            }), patch("claims_sample_miner.cli.urlopen", side_effect=fake_urlopen):
                run(root / "request.json", output)
            artifact = json.loads(output.read_text())
            self.assertEqual(artifact["logic"]["claims"][0]["statement"], PAPER_TEXT)
            self.assertEqual(artifact["logic"]["claims"][0]["proof"], ["E01"])
            source = ingest_pdf(root / "paper.pdf", {"paper_id": "example-001", "title": "Example paper"})
            self.assertEqual(source["source_metadata"], {"pdf_reader": "pypdf"})
            self.assertEqual(source["paper"]["authors"], [])
            self.assertEqual(source["spans"][0]["metadata"]["reader_span_id"], "example-001-p001-001")
            self.assertIsNone(source["spans"][0]["char_start"])

    def test_repair_invalid_grounding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_pdf(root / "paper.pdf")
            (root / "request.json").write_text(json.dumps({
                "schema_version": "claims_competition_request_v1",
                "paper": {"paper_id": "example-001", "title": "Example paper"},
                "pdf_path": "paper.pdf",
                "inference": {"model": "allowed"},
            }))
            bad = valid_artifact()
            bad["logic"]["claims"][0]["sources"][0]["quote"] = "Invented quote"
            with patch("claims_sample_miner.cli.completion", side_effect=[
                json.dumps(bad), json.dumps(valid_artifact()),
            ]) as model:
                run(root / "request.json", root / "result.json")
            self.assertEqual(model.call_count, 2)
            self.assertIn("Validation errors", model.call_args.kwargs["messages"][-1]["content"])

    def test_exact_source_fallback_after_two_invalid_model_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_pdf(root / "paper.pdf")
            (root / "request.json").write_text(json.dumps({
                "schema_version": "claims_competition_request_v1",
                "paper": {"paper_id": "example-001", "title": "Example paper"},
                "pdf_path": "paper.pdf",
                "inference": {"model": "allowed"},
            }))
            bad = valid_artifact()
            bad["logic"]["claims"][0]["sources"][0]["quote"] = "Invented quote"
            bad["logic"]["claims"][0]["proof"] = ["E99"]
            bad["logic"]["claims"][0]["evidence_ids"] = ["EV99"]
            with patch("claims_sample_miner.cli.completion", return_value=json.dumps(bad)) as model:
                run(root / "request.json", root / "result.json")
            self.assertEqual(model.call_count, 2)
            artifact = json.loads((root / "result.json").read_text())
            claim = artifact["logic"]["claims"][0]
            self.assertEqual(claim["sources"][0]["quote"], PAPER_TEXT)
            self.assertEqual(claim["proof"], [])
            self.assertEqual(claim["evidence_ids"], ["EV01"])
            self.assertEqual(artifact["metadata"]["repair_strategy"], "exact_pdf_quote_alignment_v1")

    def test_scanned_pdf_fails_without_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            PdfWriter().write(root / "paper.pdf")
            with self.assertRaises(ValueError):
                ingest_pdf(root / "paper.pdf", {"paper_id": "blank"})


if __name__ == "__main__":
    unittest.main()
