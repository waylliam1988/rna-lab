"""Audit pairwise BCE/competition-label coverage on train and validation only."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rna2d.data import CANONICAL
from rna2d.evidence import pairing_evidence_issues


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_split(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    group_records = Counter()
    group_applicable = Counter()
    all_negative_counts = []
    total_positive_edges = 0
    total_positive_endpoints = 0
    total_applicable_endpoints = 0
    records_with_positive_and_negative_bce = 0
    records_without_positive = 0
    records_without_reliable_negative = 0
    lengths = []
    for row in rows:
        issues = pairing_evidence_issues(row)
        if issues:
            raise ValueError(f"Evidence contract failed for {row['id']}: {issues}")
        seq = row["sequence"]
        observed = row["observed_mask"]
        positives = {tuple(pair) for pair in row["pairs"]}
        partners = {}
        for i, j in positives:
            if i in partners or j in partners:
                raise ValueError(f"Multiple partners in {row['id']}")
            partners[i] = j
            partners[j] = i
        candidates_by_endpoint = {}
        for i in partners:
            candidates = [
                k for k, base in enumerate(seq)
                if k != i and abs(k - i) >= 4 and observed[i] and observed[k]
                and seq[min(i, k)] + seq[max(i, k)] in CANONICAL
                and tuple(sorted((i, k))) not in positives
            ]
            candidates_by_endpoint[i] = candidates
        positive_endpoints = len(partners)
        applicable_counts = [len(candidates_by_endpoint[i]) for i in partners if candidates_by_endpoint[i]]
        total_positive_edges += len(positives)
        total_positive_endpoints += positive_endpoints
        total_applicable_endpoints += len(applicable_counts)
        all_negative_counts.extend(applicable_counts)
        if positives and any(applicable_counts):
            records_with_positive_and_negative_bce += 1
        if not positives:
            records_without_positive += 1
        if not applicable_counts:
            records_without_reliable_negative += 1
        lengths.append(len(seq))
        group = row.get("split_group_id")
        if not group:
            raise ValueError(f"Missing split_group_id in {row['id']}")
        group_records[group] += 1
        if applicable_counts:
            group_applicable[group] += 1
    summary = {
        "records": len(rows),
        "groups": len(group_records),
        "groups_with_competition_applicable_record": len(group_applicable),
        "records_with_positive_and_negative_bce": records_with_positive_and_negative_bce,
        "records_without_positive": records_without_positive,
        "records_without_reliable_negative_endpoint": records_without_reliable_negative,
        "positive_edges": total_positive_edges,
        "positive_endpoints": total_positive_endpoints,
        "competition_applicable_endpoints": total_applicable_endpoints,
        "competition_endpoint_coverage": total_applicable_endpoints / total_positive_endpoints if total_positive_endpoints else 0.0,
        "reliable_negative_candidates_per_applicable_endpoint": {
            "min": min(all_negative_counts) if all_negative_counts else 0,
            "median": statistics.median(all_negative_counts) if all_negative_counts else 0,
            "mean": statistics.mean(all_negative_counts) if all_negative_counts else 0,
            "max": max(all_negative_counts) if all_negative_counts else 0,
        },
        "mean_log1p_reliable_negative_candidates": statistics.mean(math.log1p(x) for x in all_negative_counts) if all_negative_counts else 0.0,
        "lambda_candidate_aux_to_bce_at_zero_logit_ratio": 0.05 * (statistics.mean(math.log1p(x) for x in all_negative_counts) if all_negative_counts else 0.0) / math.log(2),
        "length": {"min": min(lengths), "median": statistics.median(lengths), "max": max(lengths)},
        "_group_ids": set(group_records),
    }
    return summary


def main() -> None:
    dataset = ROOT / "data/experimental/joint_split_v2"
    train_path = dataset / "train.jsonl"
    validation_path = dataset / "validation.jsonl"
    train = audit_split(train_path)
    validation = audit_split(validation_path)
    if train["_group_ids"] & validation["_group_ids"]:
        raise ValueError("Train and validation groups overlap")
    lambda_candidate = 0.05
    report = {
        "audit": "pair-loss-coverage-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "joint_split_v2",
        "files": {
            "train_sha256": sha256(train_path),
            "validation_sha256": sha256(validation_path),
        },
        "splits": {
            "train": {k: v for k, v in train.items() if k != "_group_ids"},
            "validation": {k: v for k, v in validation.items() if k != "_group_ids"},
        },
        "test_accessed": False,
        "lambda_candidate": lambda_candidate,
        "lambda_rationale": "At equal logits the per-endpoint ranking term is log(1 + reliable_negative_count). lambda=0.05 is fixed as a modest initial auxiliary scale (roughly a few tenths of the 0.693 balanced-BCE-at-zero scale for the observed candidate counts); it is not searched or selected on validation performance.",
        "status": "coverage_audit_only_no_training",
    }
    out = ROOT / "data/manifests/data_first_loss_coverage_v1.json"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
