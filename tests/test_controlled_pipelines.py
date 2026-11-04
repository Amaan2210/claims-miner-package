import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from claims_sample_miner.artifact_models import Artifact
from claims_sample_miner.artifact_validator import validate_agent_artifact
from claims_sample_miner.cli import InferenceError
from test_package import PAPER_TEXT, write_pdf


ROOT = Path(__file__).resolve().parents[1]
NAMES = ("full_document", "section_map", "evidence_first", "verify_findings", "results_focus")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ControlledPipelinesTests(unittest.TestCase):
    def run_responses(self, strategy, replies, *, source=None, expect_failure=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_pdf(root / "paper.pdf")
            request = root / "request.json"
            request.write_text(json.dumps({
                "schema_version": "claims_competition_request_v1",
                "paper": {"paper_id": "example-001", "title": "Example paper"},
                "pdf_path": "paper.pdf",
                "inference": {"model": "allowed/model", "allowed_models": ["allowed/model"]},
            }))
            namespace = "response_test_" + strategy
            package = types.ModuleType(namespace)
            package.__path__ = []
            sys.modules[namespace] = package
            load(namespace + ".strategy", ROOT / f"examples/pipelines/{strategy}.py")
            runner = load(namespace + ".runner", ROOT / "examples/pipelines/runner.py")
            output = root / "output.json"
            if source is None:
                source = {"spans": [{"span_id": "real", "text": PAPER_TEXT}], "source_integrity": {}}
            source = {**source, "paper": {"paper_id": "example-001", "title": "Example paper"}}
            with patch.object(runner, "ingest_pdf", return_value=source), \
                 patch.object(runner, "completion", side_effect=replies) as completion, \
                 patch.object(sys, "argv", ["pipeline", "--request", str(request), "--output", str(output)]):
                if expect_failure:
                    with self.assertRaisesRegex(ValueError, "no grounded findings"):
                        runner.main()
                    self.assertFalse(output.exists())
                    return None, completion.call_args_list
                runner.main()
            artifact = Artifact.model_validate_json(output.read_text())
            self.assertEqual(validate_agent_artifact(artifact), [])
            return artifact, completion.call_args_list

    def test_evidence_schema_conflict_is_removed_and_missing_passages_gets_repaired(self):
        artifact, calls = self.run_responses("evidence_first", [
            json.dumps({"claims": []}),
            json.dumps({"passages": [None, {"span_id": "real", "quote": []},
                                      {"span_id": "real", "quote": PAPER_TEXT}]}),
            json.dumps({"claims": [{"statement": PAPER_TEXT, "span_id": "real", "quote": PAPER_TEXT}]}),
        ])
        self.assertEqual(len(calls), 3)
        self.assertEqual(artifact.metadata["invalid_model_responses"], 1)
        first_prompt = calls[0].kwargs["messages"][0]["content"]
        self.assertIn('Return JSON {"passages":', first_prompt)
        self.assertNotIn('Return JSON {"claims":', first_prompt)
        self.assertIn('Return JSON {"claims":', calls[2].kwargs["messages"][0]["content"])

    def test_malformed_json_is_repaired_and_invalid_rows_do_not_destroy_valid_findings(self):
        artifact, calls = self.run_responses("section_map", ["not JSON", json.dumps({"claims": [
            None,
            {"statement": {"bad": "shape"}, "span_id": "real", "quote": PAPER_TEXT},
            {"statement": "Invented", "span_id": "real", "quote": "Fabricated evidence"},
            {"statement": PAPER_TEXT, "span_id": "real", "quote": PAPER_TEXT},
        ]})])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(artifact.logic.claims), 1)
        self.assertEqual(artifact.metadata["discarded_ungrounded_claims"], 3)
        self.assertEqual(artifact.logic.claims[0].sources[0].quote, PAPER_TEXT)

    def test_failed_section_response_does_not_discard_other_sections(self):
        source = {"spans": [{"span_id": name, "text": PAPER_TEXT * 300}
                            for name in ["first", "second"]], "source_integrity": {}}
        artifact, calls = self.run_responses("section_map", ["bad", "bad", json.dumps({"claims": [
            {"statement": PAPER_TEXT, "span_id": "second", "quote": PAPER_TEXT},
        ]})], source=source)
        self.assertEqual(len(calls), 3)
        self.assertEqual(artifact.logic.claims[0].sources[0].span_ids, ["second"])
        self.assertEqual(artifact.metadata["invalid_model_responses"], 2)

    def test_response_repairs_share_the_six_call_limit_and_never_fabricate_fallbacks(self):
        source = {"spans": [{"span_id": str(i), "text": PAPER_TEXT * 300} for i in range(4)],
                  "source_integrity": {}}
        _, calls = self.run_responses("section_map", ["bad"] * 6,
                                      source=source, expect_failure=True)
        self.assertEqual(len(calls), 6)

    def test_transient_upstream_failure_gets_one_bounded_retry(self):
        artifact, calls = self.run_responses("full_document", [
            InferenceError("inference gateway HTTP 400 (upstream 504)", retryable=True),
            json.dumps({"claims": [{"statement": PAPER_TEXT, "span_id": "real", "quote": PAPER_TEXT}]}),
        ])
        self.assertEqual(len(calls), 2)
        self.assertEqual(artifact.metadata["transient_inference_errors"], 1)

    def test_bad_requests_are_not_retried_as_model_format_errors(self):
        with self.assertRaisesRegex(InferenceError, "upstream 400"):
            self.run_responses("full_document", [
                InferenceError("inference gateway HTTP 400 (upstream 400)", retryable=False),
                AssertionError("bad requests must not be retried"),
            ])

    def test_empty_grounded_output_gets_one_repair_and_never_fabricates_evidence(self):
        for repair_succeeds in (True, False):
            with self.subTest(repair_succeeds=repair_succeeds), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                request = root / "request.json"
                request.write_text(json.dumps({
                    "schema_version": "claims_competition_request_v1",
                    "paper": {"paper_id": "example-001", "title": "Example"},
                    "pdf_path": "paper.pdf",
                    "inference": {"model": "allowed/model", "allowed_models": ["allowed/model"]},
                }))
                namespace = "controlled_repair"
                package = types.ModuleType(namespace)
                package.__path__ = []
                sys.modules[namespace] = package
                load(namespace + ".strategy", ROOT / "examples/pipelines/full_document.py")
                runner = load(namespace + ".runner", ROOT / "examples/pipelines/runner.py")
                source = {"spans": [{"span_id": "real", "text": PAPER_TEXT}], "source_integrity": {}}
                bad = {"statement": "Result", "span_id": "real", "quote": "fabricated quotation"}
                good = {"statement": PAPER_TEXT, "span_id": "real", "quote": PAPER_TEXT}
                replies = [json.dumps({"claims": [bad]}),
                           json.dumps({"claims": [good if repair_succeeds else bad]})]
                output = root / "output.json"
                with patch.object(runner, "ingest_pdf", return_value=source), \
                     patch.object(runner, "completion", side_effect=replies) as completion, \
                     patch.object(sys, "argv", ["pipeline", "--request", str(request), "--output", str(output)]):
                    if repair_succeeds:
                        runner.main()
                        artifact = Artifact.model_validate_json(output.read_text())
                        self.assertEqual(artifact.logic.claims[0].sources[0].quote, PAPER_TEXT)
                        self.assertEqual(artifact.metadata["model_calls"], 2)
                    else:
                        with self.assertRaisesRegex(ValueError, "no grounded findings"):
                            runner.main()
                        self.assertFalse(output.exists())
                    self.assertEqual(completion.call_count, 2)

    def test_all_strategies_write_grounded_artifacts_and_use_the_expected_calls(self):
        for strategy, expected_calls in zip(NAMES, (1, 1, 2, 2, 1), strict=True):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_pdf(root / "paper.pdf")
                request = root / "request.json"
                request.write_text(json.dumps({
                    "schema_version": "claims_competition_request_v1",
                    "paper": {"paper_id": "example-001", "title": "Example paper"},
                    "pdf_path": "paper.pdf",
                    "inference": {"model": "allowed/model", "allowed_models": ["allowed/model"]},
                }))
                namespace = "controlled_" + strategy
                package = types.ModuleType(namespace)
                package.__path__ = []
                sys.modules[namespace] = package
                load(namespace + ".strategy", ROOT / f"examples/pipelines/{strategy}.py")
                runner = load(namespace + ".runner", ROOT / "examples/pipelines/runner.py")
                claim = {"statement": PAPER_TEXT, "span_id": "example-001-span-0001", "quote": PAPER_TEXT}

                def completion(**kwargs):
                    if 'Return {"passages":' in kwargs["messages"][0]["content"]:
                        return json.dumps({"passages": [{"span_id": claim["span_id"], "quote": PAPER_TEXT}]})
                    return json.dumps({"claims": [claim, {
                        "statement": "Invented result", "span_id": claim["span_id"],
                        "quote": "This evidence does not exist in the source.",
                    }, None]})

                output = root / "output.json"
                with patch.object(runner, "completion", side_effect=completion), patch.object(sys, "argv", ["pipeline", "--request", str(request), "--output", str(output)]):
                    runner.main()
                artifact = Artifact.model_validate_json(output.read_text())
                self.assertEqual(validate_agent_artifact(artifact), [])
                self.assertEqual(artifact.metadata["model_calls"], expected_calls)
                self.assertEqual(artifact.metadata["discarded_ungrounded_claims"], 2)
                self.assertEqual(artifact.logic.claims[0].sources[0].quote, PAPER_TEXT)

    def test_evidence_first_rejects_fabricated_passages_before_the_second_call(self):
        module = load("controlled_evidence_rejection", ROOT / "examples/pipelines/evidence_first.py")
        calls = []

        def ask(*args, **kwargs):
            calls.append(args)
            self.assertEqual(kwargs, {"response_key": "passages"})
            return {"passages": [{"span_id": "real", "quote": "invented"}]}

        with self.assertRaisesRegex(ValueError, "no grounded passages"):
            module.extract({"spans": [{"span_id": "real", "text": PAPER_TEXT}]}, ask)
        self.assertEqual(len(calls), 1)
