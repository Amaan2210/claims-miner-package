from claims_sample_miner.grounding import grounded_passage


def extract(source, ask):
    """Select verbatim result evidence before expressing the scientific findings."""
    evidence = ask(
        'First identify original result passages. Return {"passages": [{"span_id": "...", "quote": "..."}]}; do not yet write claims.',
        source,
        response_key="passages",
    )["passages"]
    spans = {row["span_id"]: row for row in source["spans"]}
    checked = []
    for row in evidence:
        if not isinstance(row, dict):
            continue
        passage = grounded_passage(source["spans"], row.get("span_id"), row.get("quote"))
        if passage is not None:
            checked.append({**passage, "context": spans[passage["span_id"]]["text"]})
    if not checked:
        raise ValueError("evidence selection returned no grounded passages")
    return ask("Express source-faithful findings from these verified passages. Return the claims schema.",
               {"paper": source["paper"], "evidence": checked})["claims"]
