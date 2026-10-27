"""Audit exact profile/history links without changing data admission or splits."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def classify_profile_relation(assigned_split: str | None) -> tuple[str, bool]:
    """Never admit a probing-only record as pair supervision."""
    if assigned_split in {"train", "validation", "test"}:
        return f"exact_sequence_alias_to_joint_1d_{assigned_split}_assignment", False
    if assigned_split == "quarantine":
        return "quarantined_unresolved_1d_parent_study_or_homology", False
    return "unmapped_sequence_quarantined", False


def classify_history_relation(prior_split: str, current_split: str | None) -> tuple[str, bool]:
    """Record historical exposure while prohibiting reuse in new training."""
    if prior_split in {"test", "test_external"}:
        return "historically_exposed_holdout", False
    if current_split in {"train", "validation", "test"}:
        return f"exact_sequence_alias_to_joint_{current_split}", False
    if current_split == "quarantine":
        return "exact_sequence_alias_to_joint_quarantine", False
    return "not_exactly_mapped_to_joint_v2", False


def load_assignments(path: Path):
    assignments = {}
    for row in read_jsonl(path):
        key = row["sequence_sha256"]
        if key in assignments:
            raise ValueError(f"Duplicate joint 1D assignment hash: {key}")
        assignments[key] = row
    return assignments


def load_joint_pairs(dataset: Path):
    records = {}
    for split in ("train", "validation", "test"):
        for row in read_jsonl(dataset / f"{split}.jsonl"):
            key = row["sequence_sha256"]
            if key in records:
                raise ValueError(f"Duplicate joint paired sequence: {key}")
            records[key] = (split, row)
    return records


def audit_history_file(path: Path, pool: str, assignments, joint_pairs, output):
    count = 0
    hash_matches = Counter()
    paired_aliases = Counter()
    group_matches = set()
    seen = set()
    with output.open("w", encoding="utf-8", newline="\n") as ledger:
        for row in read_jsonl(path):
            sequence = row["sequence"]
            sequence_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
            if row.get("sequence_sha256") != sequence_hash:
                raise ValueError(f"Sequence hash mismatch in {pool}: {row.get('id')}")
            identity = (pool, sequence_hash)
            if identity in seen:
                raise ValueError(f"Duplicate sequence in historical split {pool}")
            seen.add(identity)
            assignment = assignments.get(sequence_hash)
            current_split = assignment["split"] if assignment else None
            status, training_allowed = classify_history_relation(pool.split(":", 1)[1], current_split)
            paired = joint_pairs.get(sequence_hash)
            label_match = None
            current_pair_split = None
            current_group = None
            if paired:
                current_pair_split, current = paired
                current_group = current.get("split_group_id")
                label_match = (
                    row.get("observed_mask") == current.get("observed_mask")
                    and row.get("pairs") == current.get("pairs")
                )
                paired_aliases["same_sequence"] += 1
                paired_aliases["exact_mask_pair_match" if label_match else "label_conflict"] += 1
            hash_matches[current_split or "no_exact_joint_assignment"] += 1
            if current_group:
                group_matches.add(current_group)
            ledger.write(json.dumps({
                "record_id": row["id"],
                "sequence_sha256": sequence_hash,
                "source_pool_split": pool,
                "historical_evaluation_exposed": pool.split(":", 1)[1] in {"test", "test_external"},
                "relation_to_joint_1d_assignment": current_split or "no_exact_assignment",
                "relation_to_joint_paired_record": current_pair_split or "no_exact_pair_alias",
                "paired_label_match": label_match,
                "joint_isolation_group_id": current_group,
                "decision": status,
                "training_allowed": training_allowed,
            }, ensure_ascii=True) + "\n")
            count += 1
    return {
        "records": count,
        "distinct_sequences": len(seen),
        "exact_joint_assignment_counts": dict(sorted(hash_matches.items())),
        "exact_joint_pair_aliases": dict(sorted(paired_aliases.items())),
        "exact_joint_pair_group_count": len(group_matches),
        "ledger_sha256": sha256(output),
        "ledger_rows_local_only": True,
        "training_allowed": False,
    }


def audit_profiles():
    manifests = ROOT / "data/manifests"
    profile_db_path = ROOT / "data/experimental/probing_clean_v2/profiles.sqlite3"
    profile_config_path = ROOT / "data/experimental/probing_clean_v2/config.json"
    profile_summary_path = ROOT / "data/experimental/probing_clean_v2/summary.json"
    header_review_path = ROOT / "data/experimental/probing_clean_v2/source_header_review.json"
    clean_db_path = ROOT / "data/experimental/clean_release_v1/sequences.sqlite3"
    output_path = manifests / "data_first_profile_relations_v1.jsonl"
    temp_output_path = output_path.with_suffix(output_path.suffix + ".writing")

    for path in (output_path, temp_output_path, manifests / "data_first_history_relations_v1.jsonl"):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite local relation ledger: {path}")

    assignments = load_assignments(ROOT / "data/experimental/joint_split_v2/joint_1d_assignments.jsonl")
    joint_pairs = load_joint_pairs(ROOT / "data/experimental/joint_split_v2")
    acquisition = json.loads((ROOT / "data/reports/experimental_acquisition_v1_all.json").read_text(encoding="utf-8"))
    receipts = {row["id"]: row for row in acquisition["results"]}
    raw_manifest = {}
    raw_manifest_path = manifests / "raw_artifacts_v1.jsonl"
    for row in read_jsonl(raw_manifest_path):
        raw_manifest[row["id"]] = row

    profile_config = json.loads(profile_config_path.read_text(encoding="utf-8"))
    header_review = json.loads(header_review_path.read_text(encoding="utf-8"))
    profile_summary = json.loads(profile_summary_path.read_text(encoding="utf-8"))
    if sha256(ROOT / "scripts/clean_probing_corpus.py") != profile_config["script_sha256"]:
        raise ValueError("Probing cleaning script no longer matches its recorded protocol")
    if sha256(ROOT / "src/rna2d/probing_qc.py") != profile_config["module_sha256"]:
        raise ValueError("Probing QC module no longer matches its recorded protocol")
    if sha256(raw_manifest_path) != json.loads((manifests / "raw_artifacts_v1_summary.json").read_text(encoding="utf-8"))["manifest_sha256"]:
        raise ValueError("Raw-artifact manifest fingerprint mismatch")

    db = sqlite3.connect(f"file:{profile_db_path.resolve().as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise ValueError("Probing profile SQLite quick_check failed")
    file_rows = {row["id"]: row for row in db.execute("SELECT id,sha256,annotations,counts FROM files")}
    sequence_rows = {row["sha256"]: row["sequence"] for row in db.execute("SELECT sha256,sequence FROM sequences")}
    for sequence_hash, sequence in sequence_rows.items():
        if hashlib.sha256(sequence.encode("utf-8")).hexdigest() != sequence_hash:
            raise ValueError(f"Sequence SHA256 mismatch: {sequence_hash}")
        if not sequence or not set(sequence) <= set("ACGU"):
            raise ValueError(f"Invalid normalized profile sequence: {sequence_hash}")

    source_files_with_profiles = {
        row[0] for row in db.execute("SELECT DISTINCT file FROM profiles")
    }
    if len(source_files_with_profiles) != header_review["accepted_source_files"]:
        raise ValueError("Profile-bearing source-file count differs from header review")
    raw_source_bytes = 0
    source_upstream_verified = Counter()
    verified_sources = 0
    for source_id in sorted(source_files_with_profiles):
        source = file_rows.get(source_id)
        receipt = receipts.get(source_id)
        artifact = raw_manifest.get(source_id)
        if not source or not receipt or not artifact:
            raise ValueError(f"Missing profile source metadata receipt: {source_id}")
        expected_hash = source["sha256"]
        if expected_hash != receipt["sha256"] or expected_hash != artifact["sha256"]:
            raise ValueError(f"Source checksum crosswalk mismatch: {source_id}")
        source_path = ROOT / receipt["path"]
        if not source_path.is_file() or source_path.stat().st_size != receipt["bytes"]:
            raise ValueError(f"Missing or size-mismatched raw profile source: {source_id}")
        actual_hash = sha256(source_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Raw profile source SHA256 mismatch: {source_id}")
        verified_sources += 1
        raw_source_bytes += receipt["bytes"]
        source_upstream_verified[str(bool(artifact.get("upstream_sha256_verified")))] += 1

    profile_counts_by_file = Counter()
    profile_relation_counts = Counter()
    assignment_counts = Counter()
    assignment_sequence_counts = defaultdict(set)
    position_counts = []
    position_coverages = []
    observed_position_counts = []
    profile_count = 0
    matched_sequences = set()
    coordinate_overlap = 0
    try:
        with temp_output_path.open("w", encoding="utf-8", newline="\n") as ledger:
            query = """SELECT file,profile,seqhash,seqpos,reactivity,uncertainty,mask,annotations
                       FROM profiles ORDER BY file,profile"""
            for row in db.execute(query):
                source_id = row["file"]
                source = file_rows[source_id]
                sequence_hash = row["seqhash"]
                sequence = sequence_rows.get(sequence_hash)
                if sequence is None:
                    raise ValueError(f"Profile references missing sequence: {source_id}")
                try:
                    positions = json.loads(row["seqpos"])
                    values = np.frombuffer(zlib.decompress(row["reactivity"]), dtype=np.float32)
                    uncertainty = np.frombuffer(zlib.decompress(row["uncertainty"]), dtype=np.float32)
                    mask = np.frombuffer(zlib.decompress(row["mask"]), dtype=np.bool_)
                    profile_annotations = json.loads(row["annotations"])
                except Exception as exc:
                    raise ValueError(f"Corrupt accepted profile {source_id}/{row['profile']}") from exc
                if not positions or len(set(positions)) != len(positions):
                    raise ValueError(f"Invalid source-position mapping: {source_id}/{row['profile']}")
                if any(not isinstance(i, int) or i < 0 or i >= len(sequence) for i in positions):
                    raise ValueError(f"Out-of-range source-position mapping: {source_id}/{row['profile']}")
                if not (len(values) == len(uncertainty) == len(mask) == len(positions)):
                    raise ValueError(f"Profile arrays do not align: {source_id}/{row['profile']}")
                observed = int(mask.sum())
                coverage = observed / len(mask) if len(mask) else 0.0
                if observed < profile_config["min_observed"] or coverage < profile_config["min_reported_position_coverage"]:
                    raise ValueError(f"Profile fails recorded measurement mask gate: {source_id}/{row['profile']}")
                if not np.isfinite(values[mask]).all() or not np.isfinite(uncertainty[mask]).all() or (uncertainty[mask] < 0).any():
                    raise ValueError(f"Invalid measured values or errors: {source_id}/{row['profile']}")
                assignment = assignments.get(sequence_hash)
                assigned_split = assignment["split"] if assignment else None
                decision, training_allowed = classify_profile_relation(assigned_split)
                if training_allowed:
                    raise AssertionError("A probing-only profile was marked as pair training data")
                assignment_counts[assigned_split or "unmapped"] += 1
                profile_relation_counts[decision] += 1
                profile_counts_by_file[source_id] += 1
                matched_sequences.add(sequence_hash)
                assignment_sequence_counts[assigned_split or "unmapped"].add(sequence_hash)
                position_counts.append(len(positions))
                position_coverages.append(coverage)
                observed_position_counts.append(observed)
                profile_count += 1
                key = hashlib.sha256((source_id + "\0" + row["profile"]).encode("utf-8")).hexdigest()
                ledger.write(json.dumps({
                    "profile_key_sha256": key,
                    "source_file_sha256": source["sha256"],
                    "sequence_sha256": sequence_hash,
                    "sequence_length": len(sequence),
                    "source_positions": len(positions),
                    "observed_positions": observed,
                    "observed_fraction": coverage,
                    "joint_1d_assignment": assigned_split or "unmapped_quarantine",
                    "isolation_group_id": assignment.get("group") if assignment else None,
                    "relation_status": decision,
                    "evidence_class": "directly_measured_reactivity_not_pair_truth",
                    "label_type": "no_pair_label",
                    "training_allowed": False,
                }, ensure_ascii=True) + "\n")
        if profile_count != profile_summary["measured_profiles"]:
            raise ValueError("Accepted profile count differs from published screening summary")
        if len(sequence_rows) != profile_summary["unique_sequences"] or len(matched_sequences) != len(sequence_rows):
            raise ValueError("Profile sequence cardinality differs from the screened unique-sequence set")
        if set(profile_counts_by_file) != source_files_with_profiles:
            raise ValueError("Not every header-reviewed source file has accepted profiles")
        expected_file_counts = {
            source_id: json.loads(file_rows[source_id]["counts"])["accepted_measured_profiles"]
            for source_id in source_files_with_profiles
        }
        if dict(profile_counts_by_file) != expected_file_counts:
            raise ValueError("Per-source accepted profile counts differ from source receipts")

        clean_db = sqlite3.connect(f"file:{clean_db_path.resolve().as_posix()}?mode=ro", uri=True)
        try:
            clean_sequences = {
                sequence_hash: sequence for sequence_hash, sequence in clean_db.execute(
                    "SELECT sha256,sequence FROM sequences"
                )
            }
            if any(hashlib.sha256(sequence.encode("utf-8")).hexdigest() != sequence_hash
                   for sequence_hash, sequence in clean_sequences.items()):
                raise ValueError("Clean-release sequence content does not match its recorded SHA256")
            clean_hashes = set(clean_sequences)
            if clean_hashes != set(assignments):
                raise ValueError("joint 1D assignments do not exactly cover clean-release sequence hashes")
            if any(clean_sequences[sequence_hash] != sequence for sequence_hash, sequence in sequence_rows.items()):
                raise ValueError("Probing and clean-release sequence content differs under identical SHA256")
            coordinate_hashes = {
                sequence_hash for (sequence_hash,) in clean_db.execute(
                    "SELECT sha256 FROM sequences WHERE coordinate_evidence=1"
                )
            }
            coordinate_overlap = len(matched_sequences & coordinate_hashes)
        finally:
            clean_db.close()
    finally:
        db.close()

    os.replace(temp_output_path, output_path)

    return {
        "profile_db_sha256": sha256(profile_db_path),
        "profile_config_sha256": sha256(profile_config_path),
        "profile_screen_summary_sha256": sha256(profile_summary_path),
        "source_header_review_sha256": sha256(header_review_path),
        "profile_sequence_records": len(sequence_rows),
        "profile_records": profile_count,
        "clean_release_sequence_records": len(clean_sequences),
        "joint_1d_assignment_records": len(assignments),
        "sqlite_file_records": len(file_rows),
        "profile_bearing_source_files": len(source_files_with_profiles),
        "raw_source_receipts_matched": verified_sources,
        "raw_source_local_sha256_verified": verified_sources,
        "raw_source_total_bytes_verified": raw_source_bytes,
        "upstream_sha256_verified_file_counts": dict(source_upstream_verified),
        "profile_count_by_joint_assignment": dict(sorted(assignment_counts.items())),
        "profile_unique_sequence_count_by_joint_assignment": {
            key: len(values) for key, values in sorted(assignment_sequence_counts.items())
        },
        "profile_count_by_relation_status": dict(sorted(profile_relation_counts.items())),
        "unique_profile_sequences_mapped": len(matched_sequences),
        "profile_sequence_coordinate_evidence_overlap": coordinate_overlap,
        "position_count": {
            "min": min(position_counts), "median": float(np.median(position_counts)), "max": max(position_counts),
        },
        "observed_position_count": {
            "min": min(observed_position_counts), "median": float(np.median(observed_position_counts)), "max": max(observed_position_counts),
        },
        "observed_fraction": {
            "min": float(min(position_coverages)), "median": float(np.median(position_coverages)), "max": float(max(position_coverages)),
        },
        "local_relation_ledger_sha256": sha256(output_path),
        "local_relation_ledger_rows": profile_count,
        "training_allowed": False,
        "pair_label_count": 0,
    }


def _run_audit() -> None:
    manifests = ROOT / "data/manifests"
    history_output = manifests / "data_first_history_relations_v1.jsonl"
    profile_output = manifests / "data_first_profile_relations_v1.jsonl"
    history_temp = history_output.with_suffix(history_output.suffix + ".writing")
    profile_temp = profile_output.with_suffix(profile_output.suffix + ".writing")
    report_path = manifests / "data_first_profile_history_audit_v1.json"
    report_temp = report_path.with_suffix(report_path.suffix + ".writing")
    history_part = history_output.with_name(history_output.name + ".part")
    if any(path.exists() for path in (history_output, profile_output, history_temp, profile_temp,
                                      report_path, report_temp, history_part)):
        raise FileExistsError("Refusing to overwrite local audit relation ledgers")

    assignments = load_assignments(ROOT / "data/experimental/joint_split_v2/joint_1d_assignments.jsonl")
    joint_pairs = load_joint_pairs(ROOT / "data/experimental/joint_split_v2")
    profile_report = audit_profiles()

    history_splits = {
        "legacy_processed:train": ROOT / "data/processed/train.jsonl",
        "legacy_processed:validation": ROOT / "data/processed/validation.jsonl",
        "legacy_processed:test": ROOT / "data/processed/test.jsonl",
        "legacy_processed:test_external": ROOT / "data/processed/test_external.jsonl",
        "legacy_processed:archiveII": ROOT / "data/processed/archiveII.jsonl",
        "paired_v0_2:train": ROOT / "data/experimental/paired_v0_2/train.jsonl",
        "paired_v0_2:validation": ROOT / "data/experimental/paired_v0_2/validation.jsonl",
        "paired_v0_2:test": ROOT / "data/experimental/paired_v0_2/test.jsonl",
    }
    history_summary = {}
    with history_temp.open("w", encoding="utf-8", newline="\n") as combined:
        for pool, path in history_splits.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            # Build a deterministic temporary ledger and stream it into one
            # local, ignored file to avoid holding all rows in memory.
            temp_path = history_output.with_name(history_output.name + ".part")
            if temp_path.exists():
                raise FileExistsError(f"Refusing to overwrite partial history ledger: {temp_path}")
            result = audit_history_file(path, pool, assignments, joint_pairs, temp_path)
            with temp_path.open(encoding="utf-8") as part:
                for line in part:
                    combined.write(line)
            temp_path.unlink()
            history_summary[pool] = {
                "records": result["records"],
                "distinct_sequences": result["distinct_sequences"],
                "exact_joint_assignment_counts": result["exact_joint_assignment_counts"],
                "exact_joint_pair_aliases": result["exact_joint_pair_aliases"],
                "exact_joint_pair_group_count": result["exact_joint_pair_group_count"],
                "source_jsonl_sha256": sha256(path),
                "ledger_rows_local_only": True,
                "training_allowed": False,
            }
    os.replace(history_temp, history_output)

    report = {
        "audit": "profile-and-history-exact-relations-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Read-only exact sequence/provenance crosswalk. It does not infer homology or construct ancestry from names and does not evaluate model predictions.",
        "current_joint_split": {
            "manifest_sha256": sha256(manifests / "joint_split_v2.json"),
            "validation_sha256": sha256(manifests / "joint_split_v2_validation.json"),
            "assignments_sha256": sha256(ROOT / "data/experimental/joint_split_v2/joint_1d_assignments.jsonl"),
            "assignment_records": len(assignments),
            "paired_records": sum(1 for _ in joint_pairs.values()),
            "paired_sequence_sha256": {
                split: sha256(ROOT / f"data/experimental/joint_split_v2/{split}.jsonl")
                for split in ("train", "validation", "test")
            },
        },
        "profile_pool": profile_report,
        "historical_sequence_crosswalk": history_summary,
        "decisions": {
            "measured_reactivity": "retain as measured-position evidence only; never promote to pair truth",
            "profile_only_records": "profile rows never add pair-training rows; pair supervision and eligibility come only from frozen paired records",
            "historical_test_aliases": "mark exposed and exclude from new training/selection; exact alias audit is not a re-evaluation",
            "homology_construct_study": "no inference beyond the existing frozen joint split; unresolved relationships remain quarantine",
            "new_independent_test": "not established by this exact-alias audit",
            "auxiliary_source": "the fixed comparison-source pilot remains not admitted; no source eligibility change",
        },
        "limitations": [
            "A verified local source checksum and measured array do not substitute for original-paper, biological-replicate, construct-parent or remote-homology review.",
            "Profile-bearing RMDB files have header metadata screening only; source header review is not a complete per-publication assay audit.",
            "Exact sequence hashes detect aliases only; this audit performs no alignment-based homology or construct ancestry analysis.",
            "Historical split membership is recorded as prior exposure context, not used for model selection or performance analysis.",
            "All row-level relation ledgers remain local and ignored by Git; only aggregate counts and fingerprints are publishable.",
        ],
        "row_level_ledgers_local_only": {
            "profile_relations": "data/manifests/data_first_profile_relations_v1.jsonl",
            "history_relations": "data/manifests/data_first_history_relations_v1.jsonl",
        },
        "test_predictions_read": False,
        "training_started": False,
    }
    with report_temp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, ensure_ascii=True, indent=2) + "\n")
    os.replace(report_temp, report_path)
    print(json.dumps({
        "path": str(report_path.relative_to(ROOT)),
        "sha256": sha256(report_path),
        "profiles": profile_report["profile_records"],
        "profile_sequences": profile_report["profile_sequence_records"],
        "source_files_sha256_verified": profile_report["raw_source_local_sha256_verified"],
        "history_splits": {name: row["exact_joint_assignment_counts"] for name, row in history_summary.items()},
        "test_predictions_read": False,
        "training_started": False,
    }, ensure_ascii=True, indent=2))


def main() -> None:
    manifests = ROOT / "data/manifests"
    outputs = (
        manifests / "data_first_profile_relations_v1.jsonl",
        manifests / "data_first_profile_relations_v1.jsonl.writing",
        manifests / "data_first_history_relations_v1.jsonl",
        manifests / "data_first_history_relations_v1.jsonl.writing",
        manifests / "data_first_history_relations_v1.jsonl.part",
        manifests / "data_first_profile_history_audit_v1.json",
        manifests / "data_first_profile_history_audit_v1.json.writing",
    )
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite audit outputs: {existing[0]}")
    try:
        _run_audit()
    except Exception:
        for path in outputs:
            path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
