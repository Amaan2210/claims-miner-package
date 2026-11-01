# Claims sample miner package

This repository is a complete Python extraction package for the [Claims weekly competition](https://github.com/DeSciClaims/Claims). It reads a supplied PDF, extracts stable source spans, calls a permitted model through the Claims inference gateway, and writes a source-grounded `agent_v1` artifact. It validates the artifact and retries once with validation feedback.

Use it as a working starting point for your own pipeline. Its [manifest](./examples/competition_manifest.json), [request](./examples/request.json), and [package entrypoint](./src/claims_sample_miner/cli.py) show the competition contract.

## Package contract

Build a wheel containing source and declare a `module:function` entrypoint in the manifest. For this package the entrypoint is `claims_sample_miner.cli:main`. The worker invokes it with:

```text
--request /path/to/request.json --output /path/to/output.json
```

The request has schema `claims_competition_request_v1`. It contains paper metadata, a read-only local `pdf_path`, and inference settings: the default model, allowed models, streaming preference, and maximum output tokens. `inference.budget` supplies the actual paper `token_limit` (input plus output across all calls), `dollar_limit_usd`, `remaining_tokens`, and `remaining_dollar_limit_usd`. The remaining amounts match the paper-scoped gateway key at attempt start; a retry receives only the unused allowance. Track usage during execution; these fields are not live counters. The entrypoint must write one JSON `agent_v1` artifact for that paper. It may make multiple allowed inference calls within this budget. The gateway enforces the allowance independently of the package.

The worker sets `CLAIMS_INFERENCE_BASE_URL` to the gateway's OpenAI-compatible `/v1` endpoint and `CLAIMS_INFERENCE_TOKEN` to a short-lived, paper-scoped virtual key. Your code can read this key to call `/chat/completions`; it never receives the upstream OpenRouter key or gateway administration credential. Do not write the scoped key into your output or logs.

## Build and verify

```sh
uv build --wheel --out-dir dist
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests
```

The tests use a generated PDF and a local gateway fixture. They require no provider credentials. This package reads selectable PDF text; add OCR if your pipeline must handle image-only papers.

For a local extraction run, use your own OpenAI-compatible gateway and set the two inference environment variables before invoking the installed entrypoint:

```sh
CLAIMS_INFERENCE_BASE_URL=https://your-gateway.example/v1 \
CLAIMS_INFERENCE_TOKEN=your-local-key \
claims-sample-miner --request examples/request.json --output /tmp/claims-output.json
```

Set the request's model to one permitted by that gateway. The PDF path in the request resolves relative to the request file. For competition jobs, the worker supplies the request, PDF, gateway URL, and scoped key automatically.

## Submit

Update [`examples/competition_manifest.json`](./examples/competition_manifest.json) with your team ID, entrypoint, allowed model IDs, licence declaration, and the exact wheel filename from `dist/`. Upload the package through the Claims hotkey-signed upload service, then finalize it so its bytes and SHA-256 are sealed. Publish the matching on-chain package commitment and manifest ID before the round deadline. The [submission guide](https://github.com/DeSciClaims/Claims/blob/main/docs/submission.md) defines the commitment format.

Declare the same `team_id` for pipelines controlled by the same operator or team. This is a declaration, not proof that entries are independent. The backend determines effective teams using declared affiliations, shared coldkeys, resolved funding lineage and reviewed code links. Different team IDs do not override detected links; advancement and reward limits apply to the resolved teams.

The worker installs packages from an offline wheelhouse. Include or declare dependencies compatible with that environment; package installation and execution logs are available to the submitting hotkey after evaluation.
