# Controlled competition pipelines

These five pipelines exercise the competition using operator-controlled
accounts. They share the public sample's PDF ingestion and artifact schema,
with different extraction algorithms:

| Pipeline | Method |
| --- | --- |
| `full_document` | One complete-document extraction call |
| `section_map` | Independent extraction from document portions, followed by exact deduplication |
| `evidence_first` | Select and verify quotations, then express findings |
| `verify_findings` | Draft findings, then audit them against the full source |
| `results_focus` | Retrieve result-bearing spans before extraction |

All calls use the worker's scoped gateway key. Each step requests its own JSON
schema; malformed responses or confirmed transient gateway errors get at most
one retry within a shared six-call limit. Ordinary bad requests are not retried.
Invalid candidates are discarded without losing other grounded
findings. If no grounded finding remains after one evidence repair, the paper fails.
Quotes must map back to exact PDF substrings. Whitespace and ligatures can be
normalized for matching, but paraphrases and ambiguous source matches are rejected.
No evaluation paper IDs, expected answers or private provider keys are bundled.

Build using a private participant manifest with five entries containing
`wallet`, `coldkey`, `hotkey` and `team_id`:

```sh
python scripts/build_testnet_pipelines.py \
  --participants /private/path/participants.json \
  --output /private/path/packages
```

The build creates one wheel and competition manifest per account, plus
`packages.json` with hashes. Each wheel contains only its selected strategy.
Shared public source can be supplied to the normal duplicate candidate scanner;
duplicate checks and manual linkage review remain enabled. These accounts test
the workflow and do not demonstrate five independent operators.
