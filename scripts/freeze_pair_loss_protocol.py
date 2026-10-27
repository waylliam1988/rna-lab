"""Freeze the single-factor BCE vs BCE+competition protocol before training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from rna2d.sequence_encoders import SequencePairModel
from rna2d.sampling import group_balanced_sample_indices, sample_schedule_sha256
from rna2d.training import seed_all


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_sha256(state) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    output = ROOT / "data/manifests/data_first_loss_protocol_v1.json"
    if output.exists():
        raise FileExistsError("Protocol already exists; create a new version instead of overwriting")
    coverage_path = ROOT / "data/manifests/data_first_loss_coverage_v1.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if coverage["status"] != "coverage_audit_only_no_training" or coverage["test_accessed"]:
        raise ValueError("Unexpected coverage audit state")
    seeds = [42, 43, 44]
    model_config = {"class": "rna2d.sequence_encoders.SequencePairModel", "kind": "cnn", "width": 113, "hidden": 64, "depth": 2, "parameters": 150019, "precision": "float32"}
    initial_hashes = {}
    for seed in seeds:
        seed_all(seed)
        model = SequencePairModel(model_config["kind"], model_config["width"], model_config["hidden"], model_config["depth"])
        initial_hashes[str(seed)] = state_sha256(model.state_dict())
    train = ROOT / "data/experimental/joint_split_v2/train.jsonl"
    validation = ROOT / "data/experimental/joint_split_v2/validation.jsonl"
    train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines() if line]
    sample_order_hashes = {}
    for seed in seeds:
        schedule = group_balanced_sample_indices(
            [row["split_group_id"] for row in train_rows], updates=550, batch_size=4, seed=seed,
        )
        sample_order_hashes[str(seed)] = sample_schedule_sha256(schedule, [row["id"] for row in train_rows])
    source_paths = {
        "model": ROOT / "src/rna2d/sequence_encoders.py",
        "loss": ROOT / "src/rna2d/losses.py",
        "training": ROOT / "src/rna2d/training.py",
        "sampling": ROOT / "src/rna2d/sampling.py",
        "data": ROOT / "src/rna2d/data.py",
        "protocol_freezer": ROOT / "scripts/freeze_pair_loss_protocol.py",
        "runner": ROOT / "scripts/run_pair_loss_comparison.py",
        "verifier": ROOT / "scripts/verify_pair_loss_comparison.py",
        "summarizer": ROOT / "scripts/summarize_pair_loss_comparison.py",
        "coverage_auditor": ROOT / "scripts/audit_pair_loss_coverage.py",
    }
    group_deltas = coverage["splits"]["validation"]["groups"]
    protocol = {
        "protocol": "data-first-pair-loss-v1",
        "status": "frozen_before_training",
        "coverage_audit_sha256": sha256(coverage_path),
        "dataset": "data/experimental/joint_split_v2",
        "data_sha256": {"train": sha256(train), "validation": sha256(validation)},
        "split_counts": {"train": 88, "validation": 10},
        "split_groups": {"train": 8, "validation": 7},
        "isolation_manifest_sha256": sha256(ROOT / "data/manifests/joint_split_v2.json"),
        "split_validation_manifest_sha256": sha256(ROOT / "data/manifests/joint_split_v2_validation.json"),
        "test_access": "prohibited; runner and verifier open only train.jsonl and validation.jsonl",
        "model": model_config,
        "model_source_sha256": sha256(source_paths["model"]),
        "initial_state_sha256": initial_hashes,
        "conditions": {
            "bce": {"loss": "balanced_pair_bce", "lambda": 0.0},
            "bce_competition": {"loss": "balanced_pair_bce + lambda * partner_competition_loss", "lambda": coverage["lambda_candidate"]},
        },
        "lambda": coverage["lambda_candidate"],
        "lambda_rationale": coverage["lambda_rationale"],
        "loss_source_sha256": sha256(source_paths["loss"]),
        "label_semantics": "Only upper-triangular pairs selected by the existing canonical AU/GC/GU, minimum separation, padding and observed-mask candidate set are supervised. Unknown pairs are masked. Competition negatives are only explicitly supervised compatible pairs with label 0; positive-only endpoints without known negatives are skipped.",
        "seeds": seeds,
        "initialization": "Fresh random initialization for all runs; exact per-seed state hashes above; paired conditions start from identical state.",
        "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.01, "gradient_clip": 1.0, "schedule": "constant", "precision": "float32", "autocast": False},
        "batch_size": 4,
        "epochs": 25,
        "optimizer_updates": 550,
        "updates_per_epoch": 22,
        "sampling": "Uniformly sample split_group_id, then uniformly sample an RNA within that group, with replacement; paired conditions share the exact precomputed 550-update schedule for each seed.",
        "sample_order_sha256": sample_order_hashes,
        "checkpoint": "Final optimizer update only; no validation checkpoint selection or early stopping.",
        "threshold": 0.5,
        "primary_metric": "validation group-macro F1 (RNA F1 averaged within split_group_id, then equally across groups)",
        "secondary_metrics": ["validation RNA-macro precision/recall/F1", "per-group F1", "per-RNA F1", "length strata", "training time", "peak allocated CUDA memory", "loss component and competition coverage"],
        "validation_groups": group_deltas,
        "gate": {
            "delta_loss": 0.02,
            "delta_rationale": "Predeclared minimum practical gain of two absolute F1 points for continuation; set as an engineering utility threshold before any formal run, not estimated from the outcomes.",
            "required_seed_improvements": 3,
            "required_group_improvements": 4,
            "severe_group_regression_max": 0.10,
            "group_rule": "Compare per-group F1 averaged across the three paired seeds; at least 4 of 7 groups must improve and no group may decline by more than 0.10 absolute F1.",
            "pass_rule": "All three seed-level group-macro deltas are positive, their mean is at least delta_loss, at least 4/7 group-averaged deltas are positive, and no severe group regression occurs.",
            "failure_rule": "Technical failure is retained and reported; no metric-driven retries, extra seeds, lambda search, threshold changes, or protocol edits. Missing any one of six runs stops the conclusion.",
        },
        "run_limit": 6,
        "attempts_per_condition_seed": 1,
        "device": "cuda:0",
        "torch_num_threads": 1,
        "evaluation_batch_size": 4,
        "reload_tolerance": 1e-9,
        "independent_test_available": False,
        "adoption_rules": {
            "candidate": "partner competition function as implemented and tested",
            "development_gate_pass": "eligible for optional integration after independent checkpoint verification; no automatic default change",
            "development_gate_fail": "stop or retain as unadopted candidate; do not retune",
            "optional_integration": "only after gate pass and reload verification; this experiment itself does not change training CLI",
            "default_enabled": False,
        },
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths.values()},
        "environment": {"expected_device": "NVIDIA GeForce RTX 2080 Ti", "torch": "2.7.1+cu128", "cuda_runtime": "12.8"},
        "limitations": ["Only 88 training RNAs and 10 validation RNAs in 8/7 groups.", "The validation set is development evidence and has been seen previously.", "No new independent test exists; results cannot establish broad cross-family generalization.", "The stage-2 auxiliary source pilot failed admission; this is an isolated loss comparison on approved experimental labels, not an auxiliary-data A/B/C result."],
    }
    with output.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(protocol, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps({"path": str(output.relative_to(ROOT)), "sha256": sha256(output), "seeds": seeds, "updates_per_run": protocol["optimizer_updates"], "runs": protocol["run_limit"], "lambda": protocol["lambda"], "status": protocol["status"]}, indent=2))


if __name__ == "__main__":
    main()
