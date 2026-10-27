"""Build the first data-first freeze and audit manifests from existing evidence.

This script only reads existing records. It does not approve the clean-release
pool for training and never reads the held-out test records as model inputs.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, ensure_ascii=True, indent=2) + "\n")


def main() -> None:
    joint = ROOT / "data/experimental/joint_split_v2"
    old = ROOT / "data/experimental/paired_v0_2"
    clean = ROOT / "data/experimental/clean_release_v1"
    annotation_review = ROOT / "data/experimental/annotation_review_v1/paired.jsonl"
    annotation_review = ROOT / "data/experimental/annotation_review_v1/paired.jsonl"
    manifests = ROOT / "data/manifests"

    protocol = read_json(manifests / "encoder_comparison_protocol_v1.json")
    joint_manifest = read_json(manifests / "joint_split_v2.json")
    joint_validation = read_json(manifests / "joint_split_v2_validation.json")
    clean_manifest = read_json(manifests / "clean_release_v1.json")
    policy = read_json(ROOT / "data/experimental/POLICY.json")

    baseline_files = {
        "model_source": ROOT / "src/rna2d/sequence_encoders.py",
        "comparison_runner": ROOT / "scripts/run_encoder_comparison.py",
        "loss_source": ROOT / "scripts/run_controlled_experiment.py",
        "training_source": ROOT / "src/rna2d/training.py",
        "data_source": ROOT / "src/rna2d/data.py",
        "protocol": manifests / "encoder_comparison_protocol_v1.json",
        "joint_manifest": manifests / "joint_split_v2.json",
        "joint_validation": manifests / "joint_split_v2_validation.json",
        "train": joint / "train.jsonl",
        "validation": joint / "validation.jsonl",
        "test": joint / "test.jsonl",
    }
    checkpoints = {}
    for seed in (42, 43, 44):
        path = ROOT / f"runs/encoder_comparison_v1_seed{seed}/cnn.pt"
        report = read_json(path.parent / "results.json")
        row = next(x for x in report["results"] if x["encoder"] == "cnn")
        checkpoints[str(seed)] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "report_sha256": sha256(path.parent / "results.json"),
            "updates": row["updates"],
            "parameters": row["parameters"],
            "validation_macro_f1": row["validation"]["macro_f1"],
            "reload_verified": row["reload_verified"],
            "test_evaluated": report["test_evaluated"],
        }

    baseline = {
        "manifest": "data-first-baseline-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_engineering_reference",
        "model": {
            "class": "rna2d.sequence_encoders.SequencePairModel",
            "kind": "cnn", "width": 113, "hidden": 64, "depth": 2,
            "parameters": protocol["parameters"]["cnn"], "precision": protocol["precision"],
            "head": protocol["head"], "loss": "balanced_masked_bce_from_run_controlled_experiment",
        },
        "training": {k: protocol[k] for k in ("seeds", "epochs", "batch_size", "lr", "weight_decay", "gradient_clip", "checkpoint")},
        "dataset": {
            "path": protocol["dataset"], "train_sha256": protocol["data_sha256"]["train"],
            "joint_source_path": str(annotation_review.relative_to(ROOT)),
            "joint_source_sha256": sha256(annotation_review),
            "validation_sha256": protocol["data_sha256"]["validation"],
            "test_sha256": sha256(joint / "test.jsonl"), "joint_manifest_sha256": sha256(manifests / "joint_split_v2.json"),
            "joint_validation_sha256": sha256(manifests / "joint_split_v2_validation.json"),
            "split_counts": joint_manifest["splits"], "groups": joint_manifest["groups"],
            "one_d_pretraining_enabled": joint_manifest["one_d_pretraining_enabled"],
        },
        "files": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in baseline_files.items()},
        "checkpoints": checkpoints,
        "historical_boundaries": {
            "paired_v0_2": {"path": "data/experimental/paired_v0_2", "status": "historical_pipeline_validation_only", "training_default": False},
            "clean_release_v1": {"path": "data/experimental/clean_release_v1", "status": "evidence_pool_pending_joint_audit", "training_split_ready": clean_manifest["training_split_ready"]},
            "joint_split_v2": {"status": "approved_for_frozen_exploration_only", "full_pool_not_approved": True},
        },
        "limitations": ["The old test is already exposed and cannot serve as a new blind test.", "The 88/10/8 split is exploratory and not evidence of broad cross-family generalization.", "This freeze records a reference; it does not start a new training run."],
    }
    write_json(manifests / "data_first_baseline_v1.json", baseline)

    relations = []
    assigned_2d = {}
    for split in ("train", "validation", "test"):
        for row in read_jsonl(joint / f"{split}.jsonl"):
            assigned_2d[row["sequence_sha256"]] = (split, row)
    for split in ("train", "validation", "test"):
        for row in read_jsonl(joint / f"{split}.jsonl"):
            relations.append({"record_id": row["id"], "sequence_sha256": row["sequence_sha256"], "source_pool": "joint_split_v2", "split": split, "isolation_group_id": row.get("split_group_id"), "homology_components": row.get("homology_components", []), "study_ids": row.get("study_ids", []), "source_pdb_ids": row.get("source_pdb_ids", []), "context": row.get("context"), "relation_status": "reviewed_and_assigned", "evidence_class": row.get("evidence_class"), "training_allowed": split == "train"})
    for row in read_jsonl(annotation_review):
        assignment = assigned_2d.get(row["sequence_sha256"])
        assigned_row = assignment[1] if assignment else None
        label_match = bool(assigned_row and row.get("observed_mask") == assigned_row.get("observed_mask") and row.get("pairs") == assigned_row.get("pairs"))
        relations.append({"record_id": row["id"], "sequence_sha256": row["sequence_sha256"], "source_pool": "annotation_review_v1", "split": assignment[0] if assignment else None, "isolation_group_id": assigned_row.get("split_group_id") if assignment else None, "homology_components": row.get("homology_components", []), "study_ids": row.get("study_ids", []), "source_pdb_ids": row.get("source_pdb_ids", []), "context": row.get("context"), "relation_status": "exact_label_match_to_joint_split_v2" if label_match else ("same_sequence_label_mismatch" if assignment else "quarantined_or_not_assigned"), "evidence_class": row.get("evidence_class"), "label_match_to_joint_split_v2": label_match, "training_allowed": bool(label_match and assignment[0] == "train")})
    for row in read_jsonl(clean / "paired_deduplicated.jsonl"):
        assignment = assigned_2d.get(row["sequence_sha256"])
        assigned_row = assignment[1] if assignment else None
        label_match = bool(assigned_row and row.get("observed_mask") == assigned_row.get("observed_mask") and row.get("pairs") == assigned_row.get("pairs"))
        relations.append({"record_id": row["id"], "sequence_sha256": row["sequence_sha256"], "source_pool": "clean_release_v1", "split": assignment[0] if assignment else None, "isolation_group_id": assigned_row.get("split_group_id") if assignment else None, "homology_components": row.get("homology_components", []), "study_ids": row.get("study_ids", []), "source_pdb_ids": row.get("source_pdb_ids", []), "context": row.get("context"), "relation_status": "alias_same_split_exact_label" if label_match else ("alias_same_sequence_label_conflict" if assignment else "requires_joint_audit"), "evidence_class": row.get("evidence_class"), "label_match_to_joint_split_v2": label_match, "training_allowed": False})
    for row in read_jsonl(joint / "joint_1d_assignments.jsonl"):
        relations.append({"record_id": "sequence:" + row["sequence_sha256"], "sequence_sha256": row["sequence_sha256"], "source_pool": "joint_1d_assignments_v2", "split": row["split"], "isolation_group_id": row.get("group"), "relation_status": "reviewed_and_assigned" if row["split"] != "quarantine" else "quarantined_unresolved", "reason": row.get("reason"), "training_allowed": row["split"] == "train"})
    out_rel = manifests / "data_first_relations_v1.jsonl"
    with out_rel.open("w", encoding="utf-8", newline="\n") as f:
        f.write("".join(json.dumps(x, ensure_ascii=True) + "\n" for x in relations))

    admissions = []
    for row in relations:
        if row["source_pool"] == "joint_split_v2":
            admissions.append({"record_id": row["record_id"], "source_pool": row["source_pool"], "decision": "pass" if row["split"] == "train" else "held_out", "reason_code": "joint_isolation_v2_validated", "split": row["split"], "isolation_group_id": row["isolation_group_id"], "training_allowed": row["training_allowed"]})
        elif row["source_pool"] == "annotation_review_v1" and row["split"] and row.get("label_match_to_joint_split_v2"):
            admissions.append({"record_id": row["record_id"], "source_pool": row["source_pool"], "decision": "represented_in_joint_split_v2", "reason_code": "exact_sequence_mask_pair_match; no duplicate training row", "split": row["split"], "isolation_group_id": row["isolation_group_id"], "training_allowed": False})
        elif row["source_pool"] == "clean_release_v1" and row["split"]:
            admissions.append({"record_id": row["record_id"], "source_pool": row["source_pool"], "decision": "alias_excluded" if row.get("label_match_to_joint_split_v2") else "quarantine_label_conflict", "reason_code": "same_sequence_link_only; exact labels required for alias; clean_release not approved as training input", "split": row["split"], "isolation_group_id": row["isolation_group_id"], "training_allowed": False})
        elif row["source_pool"] == "joint_1d_assignments_v2":
            admissions.append({"record_id": row["record_id"], "source_pool": row["source_pool"], "decision": "quarantine" if row["split"] == "quarantine" else "excluded_from_pair_training", "reason_code": row.get("reason", "no_pair_label"), "split": row["split"], "isolation_group_id": row["isolation_group_id"], "training_allowed": False})
        else:
            admissions.append({"record_id": row["record_id"], "source_pool": row["source_pool"], "decision": "quarantine", "reason_code": "joint_homology_study_construct_audit_pending", "split": None, "isolation_group_id": None, "training_allowed": False})
    out_ad = manifests / "data_first_admission_v1.jsonl"
    with out_ad.open("w", encoding="utf-8", newline="\n") as f:
        f.write("".join(json.dumps(x, ensure_ascii=True) + "\n" for x in admissions))
    relation_counts = {}
    admission_counts = {}
    for row in relations:
        relation_counts[row["source_pool"]] = relation_counts.get(row["source_pool"], 0) + 1
    for row in admissions:
        key = row["source_pool"] + ":" + row["decision"]
        admission_counts[key] = admission_counts.get(key, 0) + 1
    clean_matches = sum(r["source_pool"] == "clean_release_v1" and r.get("label_match_to_joint_split_v2") for r in relations)
    clean_conflicts = sum(r["source_pool"] == "clean_release_v1" and r.get("relation_status") == "alias_same_sequence_label_conflict" for r in relations)
    clean_same_sequence = sum(r["source_pool"] == "clean_release_v1" and r.get("split") is not None for r in relations)
    joint_exact_matches = sum(r["source_pool"] == "annotation_review_v1" and r.get("label_match_to_joint_split_v2") for r in relations)

    candidates = list(read_jsonl(ROOT / "data/experimental/bprna_pdb_candidates_NOT_GOLD.jsonl"))[:20]
    candidate_pilot = {
        "pilot": "comparative-source-feasibility-v1",
        "source": "data/experimental/bprna_pdb_candidates_NOT_GOLD.jsonl",
        "selection": "first 20 records in fixed file order; feasibility only, never training",
        "candidate_count": len(candidates),
        "unique_sequence_sha256": len({r["sequence_sha256"] for r in candidates}),
        "total_positive_pairs": sum(len(r.get("pairs", [])) for r in candidates),
        "crossing_pair_records": sum(bool(r.get("crossing_pair_count", 0)) for r in candidates),
        "evidence_classes": sorted({r.get("evidence_class") for r in candidates}),
        "provenance_verified": sorted({r.get("provenance_verified") for r in candidates}),
        "quality_verified": sorted({r.get("quality_verified") for r in candidates}),
        "pending_reason_counts": {reason: sum(reason in r.get("reasons_pending", []) for r in candidates) for reason in sorted({x for r in candidates for x in r.get("reasons_pending", [])})},
        "decision": "stop_and_quarantine",
        "reason": "All sampled records lack verified provenance and quality; original PDB/chain, method/resolution/coverage and extractor policy are unresolved.",
        "training_allowed": False,
        "next_action": "Do not expand or train from this source until each admitted record has source receipt, mapping audit, quality review, and joint isolation evidence.",
    }
    write_json(manifests / "data_first_comparative_pilot_v1.json", candidate_pilot)

    summary = {
        "audit": "data-first-audit-v1", "created_at_utc": baseline["created_at_utc"], "status": "S0_and_S1_partial_complete",
        "baseline_manifest": "data/manifests/data_first_baseline_v1.json", "baseline_sha256": sha256(manifests / "data_first_baseline_v1.json"),
        "relations_path": "data/manifests/data_first_relations_v1.jsonl", "relations_sha256": sha256(out_rel),
        "admission_path": "data/manifests/data_first_admission_v1.jsonl", "admission_sha256": sha256(out_ad),
        "counts": {"joint_train": 88, "joint_validation": 10, "joint_test": 8, "joint_quarantine_2d": 70, "clean_release_unique_2d_sequences": clean_manifest["unique_2d_sequences"], "clean_release_same_sequence_as_split": clean_same_sequence, "clean_release_exact_mask_pair_alias": clean_matches, "clean_release_same_sequence_label_conflict": clean_conflicts, "joint_source_exact_label_matches": joint_exact_matches, "relation_rows_by_source": relation_counts, "admission_rows_by_decision": admission_counts, "joint_1d_assignments": joint_manifest["joint_1d_assignments"], "old_paired_v0_2_status": "historical_only"},
        "decisions": {"cnn_reference": "retain_as_optional_engineering_reference", "joint_split_v2": "retain_for_frozen_exploration", "clean_release_v1": "quarantine_until_joint_audit", "old_paired_v0_2": "retain_historical_pipeline_only", "one_d_pretraining": "do_not_enable"},
        "not_completed": ["Full relation coverage for probing profiles and historical aliases is not claimed; unverified material remains quarantined.", "Comparative auxiliary source trial is not admitted because sampled records lack reviewed source evidence.", "A/B/C is not frozen or run; no new effect conclusion is made."],
    }
    write_json(ROOT / "docs/DATA_FIRST_AUDIT_V1.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
