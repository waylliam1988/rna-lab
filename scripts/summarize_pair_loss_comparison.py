"""Summarize the six frozen runs and apply the predeclared development gate."""
import hashlib
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="data/manifests/data_first_loss_protocol_v2.json")
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol
    version = protocol_path.stem.removeprefix("data_first_loss_protocol_")
    if not version.startswith("v"):
        raise ValueError("Protocol filename must include an explicit version")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    verification = json.loads((ROOT / f"data/manifests/data_first_loss_verification_{version}.json").read_text(encoding="utf-8"))
    if not verification.get("all_checks_passed") or verification.get("reloaded_checkpoints") != 6:
        raise ValueError("Independent checkpoint verification must pass before summarizing")
    results = {}
    for condition in protocol["conditions"]:
        results[condition] = []
        for seed in protocol["seeds"]:
            result_path = ROOT / "runs" / f"data_first_loss_{version}" / condition / f"seed{seed}" / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result["test_accessed"] or result["updates"] != protocol["optimizer_updates"]:
                raise ValueError(f"Run violates frozen protocol: {condition}/{seed}")
            results[condition].append(result)
    seeds = protocol["seeds"]
    bce_by_seed = {r["seed"]: r for r in results["bce"]}
    comp_by_seed = {r["seed"]: r for r in results["bce_competition"]}
    for seed in seeds:
        bce_run, comp_run = bce_by_seed[seed], comp_by_seed[seed]
        expected_initial = protocol["initial_state_sha256"][str(seed)]
        expected_order = protocol["sample_order_sha256"][str(seed)]
        if bce_run["initial_state_sha256"] != expected_initial or comp_run["initial_state_sha256"] != expected_initial:
            raise ValueError(f"Initial state differs from protocol for seed {seed}")
        if bce_run["sample_order_sha256"] != comp_run["sample_order_sha256"] or bce_run["sample_order_sha256"] != expected_order:
            raise ValueError(f"Paired sample order mismatch for seed {seed}")
    seed_deltas = {str(seed): comp_by_seed[seed]["validation_group_macro_f1"] - bce_by_seed[seed]["validation_group_macro_f1"] for seed in seeds}
    mean_delta = statistics.mean(seed_deltas.values())
    groups = sorted(results["bce"][0]["validation_per_group_f1"])
    per_group_delta = {}
    for group in groups:
        bce = statistics.mean(r["validation_per_group_f1"][group] for r in results["bce"])
        comp = statistics.mean(r["validation_per_group_f1"][group] for r in results["bce_competition"])
        per_group_delta[group] = comp - bce
    conflict_rate_by_group_delta = {}
    for group in sorted(results["bce"][0]["validation_per_group_multi_partner_conflict_endpoint_rate"]):
        bce = statistics.mean(
            r["validation_per_group_multi_partner_conflict_endpoint_rate"][group]
            for r in results["bce"]
        )
        comp = statistics.mean(
            r["validation_per_group_multi_partner_conflict_endpoint_rate"][group]
            for r in results["bce_competition"]
        )
        conflict_rate_by_group_delta[group] = comp - bce
    improved_seeds = sum(delta > 0 for delta in seed_deltas.values())
    improved_groups = sum(delta > 0 for delta in per_group_delta.values())
    severe_group_regressions = {g: d for g, d in per_group_delta.items() if d < -protocol["gate"]["severe_group_regression_max"]}
    gate_passed = (
        improved_seeds == protocol["gate"]["required_seed_improvements"]
        and mean_delta >= protocol["gate"]["delta_loss"]
        and improved_groups >= protocol["gate"]["required_group_improvements"]
        and not severe_group_regressions
    )
    result = {
        "summary": f"data-first-loss-results-{version}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(protocol_path),
        "data_sha256": protocol["data_sha256"],
        "conditions": {
            condition: {
                "validation_group_macro_f1_mean": statistics.mean(r["validation_group_macro_f1"] for r in rows),
                "validation_group_macro_f1_stdev": statistics.stdev(r["validation_group_macro_f1"] for r in rows),
                "validation_macro_f1_mean": statistics.mean(r["validation"]["macro_f1"] for r in rows),
                "validation_group_macro_multi_partner_conflict_endpoint_rate_mean": statistics.mean(
                    r["validation_group_macro_multi_partner_conflict_endpoint_rate"] for r in rows
                ),
                "validation_group_macro_multi_partner_conflict_endpoint_rate_stdev": statistics.stdev(
                    r["validation_group_macro_multi_partner_conflict_endpoint_rate"] for r in rows
                ),
                "validation_macro_multi_partner_conflict_endpoint_rate_mean": statistics.mean(
                    r["validation"]["macro_multi_partner_conflict_endpoint_rate"] for r in rows
                ),
                "parameters": rows[0]["parameters"],
                "total_updates_per_seed": sorted({r["updates"] for r in rows}),
                "sample_order_hashes": {str(r["seed"]): r["sample_order_sha256"] for r in rows},
            } for condition, rows in results.items()
        },
        "seed_deltas_group_macro_f1": seed_deltas,
        "mean_seed_delta": mean_delta,
        "per_group_delta_competition_minus_bce": per_group_delta,
        "per_group_multi_partner_conflict_endpoint_rate_delta_competition_minus_bce":
            conflict_rate_by_group_delta,
        "improved_seeds": improved_seeds,
        "improved_groups": improved_groups,
        "severe_group_regressions": severe_group_regressions,
        "predeclared_development_gate_passed": gate_passed,
        "optional_feature_status": "eligible_for_optional_integration_only" if gate_passed else "stop_or_retain_candidate_without_integration",
        "default_status": "unchanged_balanced_bce",
        "independent_test_available": False,
        "test_accessed": False,
        "limits": ["Only 10 validation RNAs in seven groups; no independent test was available.", "A passed development gate is an engineering continuation rule, not statistical evidence of broad generalization."],
    }
    path = ROOT / f"data/manifests/data_first_loss_summary_{version}.json"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(result, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_group_delta_competition_minus_bce"}, indent=2))


if __name__ == "__main__":
    main()
