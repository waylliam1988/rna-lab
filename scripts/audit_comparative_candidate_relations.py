"""Crosswalk unapproved bpRNA-PDB candidates to local data without admission."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rna2d.data import parse_bpseq


HISTORY_SPLITS = {
    "legacy_processed:train": "data/processed/train.jsonl",
    "legacy_processed:validation": "data/processed/validation.jsonl",
    "legacy_processed:test": "data/processed/test.jsonl",
    "legacy_processed:test_external": "data/processed/test_external.jsonl",
    "legacy_processed:archiveII": "data/processed/archiveII.jsonl",
    "paired_v0_2:train": "data/experimental/paired_v0_2/train.jsonl",
    "paired_v0_2:validation": "data/experimental/paired_v0_2/validation.jsonl",
    "paired_v0_2:test": "data/experimental/paired_v0_2/test.jsonl",
}


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


def pair_signature(pairs, sequence_length: int) -> tuple[tuple[int, int], ...]:
    normalized = []
    for pair in pairs:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(not isinstance(index, int) for index in pair)):
            raise ValueError("Malformed pair in comparative candidate")
        left, right = sorted(pair)
        if left < 0 or right >= sequence_length or left == right:
            raise ValueError("Out-of-range or self-pair in comparative candidate")
        normalized.append((left, right))
    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate pair in comparative candidate")
    return tuple(sorted(normalized))


def compare_pair_sets(candidate, reference) -> str:
    """Compare raw edge sets only; this is not a label-quality decision."""
    return "exact_pair_set_match" if candidate == reference else "different_pair_set"


def sequence_alphabet_class(sequence: str) -> str:
    if not sequence:
        raise ValueError("Empty comparative candidate sequence")
    return "canonical_ACGU" if set(sequence) <= set("ACGU") else "contains_non_ACGU"


def audit_archive(candidates: list[dict], archive_path: Path) -> dict:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    member_checks = 0
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        archive_entry_count = len(archive.infolist())
        for row in candidates:
            member = row.get("source_member")
            if member not in names:
                raise ValueError(f"Candidate source member absent from local archive: {member}")
            sequence, pairs, _ = parse_bpseq(archive.read(member).decode("utf-8-sig"))
            if Path(member).stem != row.get("id"):
                raise ValueError(f"Candidate ID does not match source member: {member}")
            if sequence != row.get("sequence") or pair_signature(pairs, len(sequence)) != pair_signature(row.get("pairs", []), len(sequence)):
                raise ValueError(f"Candidate sequence/pairs differ from local source member: {member}")
            member_checks += 1
        archive_entries = archive_entry_count
    return {
        "local_archive_sha256": sha256(archive_path),
        "local_archive_bytes": archive_path.stat().st_size,
        "archive_member_count": archive_entries,
        "candidate_members_reconstructed": member_checks,
        "upstream_url_and_download_time_receipt_available": False,
        "biological_pdb_chain_provenance_verified": False,
    }


def audit() -> dict:
    manifests = ROOT / "data/manifests"
    candidate_path = ROOT / "data/experimental/bprna_pdb_candidates_NOT_GOLD.jsonl"
    archive_path = Path("D:/E backup/下载/bpseqFiles.zip")
    source_report_path = ROOT / "data/reports/experimental_sources_audit.json"
    asset_inventory_path = ROOT / "data/reports/local_assets_inventory.json"
    output_path = manifests / "data_first_comparative_candidate_relations_v1.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit report: {output_path}")

    candidates = list(read_jsonl(candidate_path))
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    asset_inventory = json.loads(asset_inventory_path.read_text(encoding="utf-8"))
    if len(candidates) != source_report["bprna"]["reported_pdb_candidates_parsed"]:
        raise ValueError("Candidate record count differs from the source audit")
    if len({row.get("id") for row in candidates}) != len(candidates):
        raise ValueError("Duplicate comparative candidate ID")
    if len({row.get("sequence_sha256") for row in candidates}) != source_report["bprna"]["unique_pdb_candidate_sequences"]:
        raise ValueError("Candidate unique-sequence count differs from the source audit")

    sequence_rows: dict[str, list[dict]] = defaultdict(list)
    pair_variants: dict[str, set[tuple[tuple[int, int], ...]]] = defaultdict(set)
    alphabet_record_counts = Counter()
    lengths = []
    source_flag_counts = {
        key: Counter(str(row.get(key, False)).lower() for row in candidates)
        for key in ("provenance_verified", "quality_verified", "auxiliary_training_approved")
    }
    for row in candidates:
        sequence = row.get("sequence")
        sequence_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        if row.get("sequence_sha256") != sequence_hash:
            raise ValueError(f"Candidate sequence SHA256 mismatch: {row.get('id')}")
        alphabet_record_counts[sequence_alphabet_class(sequence)] += 1
        signature = pair_signature(row.get("pairs", []), len(sequence))
        sequence_rows[sequence_hash].append(row)
        pair_variants[sequence_hash].add(signature)
        lengths.append(len(sequence))

    assignments = {}
    assignment_path = ROOT / "data/experimental/joint_split_v2/joint_1d_assignments.jsonl"
    assignment_split_counts = Counter()
    for row in read_jsonl(assignment_path):
        key = row["sequence_sha256"]
        if key in assignments:
            raise ValueError(f"Duplicate current 1D assignment hash: {key}")
        assignments[key] = row
        assignment_split_counts[row["split"]] += 1
    joint_manifest_path = manifests / "joint_split_v2.json"
    joint_validation_path = manifests / "joint_split_v2_validation.json"
    joint_manifest = json.loads(joint_manifest_path.read_text(encoding="utf-8"))
    joint_validation = json.loads(joint_validation_path.read_text(encoding="utf-8"))
    if joint_manifest.get("validation_passed") is not True or joint_validation.get("passed") is not True:
        raise ValueError("Frozen joint split does not pass its published validation")
    if dict(assignment_split_counts) != joint_manifest["joint_1d_assignments"]:
        raise ValueError("Current 1D assignment counts differ from the frozen manifest")
    current_pairs = {}
    current_pair_paths = {}
    for split in ("train", "validation", "test"):
        path = ROOT / "data/experimental/joint_split_v2" / f"{split}.jsonl"
        if sha256(path) != joint_validation["files"][split]:
            raise ValueError(f"Current {split} file differs from its published validation fingerprint")
        for row in read_jsonl(path):
            if hashlib.sha256(row["sequence"].encode("utf-8")).hexdigest() != row["sequence_sha256"]:
                raise ValueError(f"Current {split} sequence SHA256 mismatch: {row.get('id')}")
            key = row["sequence_sha256"]
            if key in current_pairs:
                raise ValueError(f"Duplicate current paired sequence hash: {key}")
            current_pairs[key] = pair_signature(row["pairs"], len(row["sequence"]))
            current_pair_paths[key] = split

    assignment_counts = Counter()
    assignment_sequence_counts: dict[str, set[str]] = defaultdict(set)
    for sequence_hash, rows in sequence_rows.items():
        assignment = assignments.get(sequence_hash)
        split = assignment["split"] if assignment else "no_exact_assignment"
        assignment_counts[split] += len(rows)
        assignment_sequence_counts[split].add(sequence_hash)

    paired_record_counts = Counter()
    paired_sequence_counts: dict[str, set[str]] = defaultdict(set)
    paired_edge_relations = Counter()
    for sequence_hash, rows in sequence_rows.items():
        current_split = current_pair_paths.get(sequence_hash)
        if current_split is None:
            continue
        paired_sequence_counts[current_split].add(sequence_hash)
        for row in rows:
            paired_record_counts[current_split] += 1
            candidate_signature = pair_signature(row["pairs"], len(row["sequence"]))
            paired_edge_relations[f"{current_split}:{compare_pair_sets(candidate_signature, current_pairs[sequence_hash])}"] += 1

    history_summary = {}
    for pool, relative_path in HISTORY_SPLITS.items():
        path = ROOT / relative_path
        history = {}
        for row in read_jsonl(path):
            sequence = row["sequence"]
            sequence_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
            if row.get("sequence_sha256") != sequence_hash:
                raise ValueError(f"Historical sequence hash mismatch in {pool}: {row.get('id')}")
            if sequence_hash in history:
                raise ValueError(f"Duplicate historical sequence in {pool}: {sequence_hash}")
            history[sequence_hash] = row
        exact_hashes = set(sequence_rows) & set(history)
        relation_counts = Counter()
        for sequence_hash in exact_hashes:
            reference = history[sequence_hash]
            if "pairs" not in reference:
                relation_counts["reference_pair_set_unavailable"] += len(sequence_rows[sequence_hash])
                continue
            reference_signature = pair_signature(reference["pairs"], len(reference["sequence"]))
            for candidate in sequence_rows[sequence_hash]:
                candidate_signature = pair_signature(candidate["pairs"], len(candidate["sequence"]))
                relation_counts[compare_pair_sets(candidate_signature, reference_signature)] += 1
        history_summary[pool] = {
            "historical_records": len(history),
            "candidate_record_exact_aliases": sum(len(sequence_rows[key]) for key in exact_hashes),
            "candidate_unique_sequence_exact_aliases": len(exact_hashes),
            "candidate_pair_set_comparison": dict(sorted(relation_counts.items())),
            "historical_evaluation_split": pool.split(":", 1)[1] in {"test", "test_external"},
            "candidate_training_allowed": False,
        }

    archive = audit_archive(candidates, archive_path)
    local_inventory = asset_inventory.get("bpseqFiles.zip", {})
    if local_inventory.get("bytes") != archive["local_archive_bytes"]:
        raise ValueError("Local archive size differs from recorded asset inventory")
    if local_inventory.get("entries") != archive["archive_member_count"]:
        raise ValueError("Local archive member count differs from recorded asset inventory")

    conflict_sequence_count = sum(len(variants) > 1 for variants in pair_variants.values())
    repeated_sequence_count = sum(len(rows) > 1 for rows in sequence_rows.values())
    duplicate_sequence_record_count = sum(len(rows) - 1 for rows in sequence_rows.values() if len(rows) > 1)
    return {
        "audit": "comparative-candidate-exact-relations-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Read-only source reconstruction, exact sequence crosswalk, and raw pair-set comparison. No candidate is admitted.",
        "candidate_source": {
            "path": "data/experimental/bprna_pdb_candidates_NOT_GOLD.jsonl",
            "sha256": sha256(candidate_path),
            "record_count": len(candidates),
            "unique_id_count": len({row["id"] for row in candidates}),
            "unique_sequence_count": len(sequence_rows),
            "repeated_sequence_hash_count": repeated_sequence_count,
            "duplicate_sequence_records_beyond_first": duplicate_sequence_record_count,
            "sequences_with_different_candidate_pair_sets": conflict_sequence_count,
            "sequence_length": {"min": min(lengths), "median": sorted(lengths)[len(lengths) // 2], "max": max(lengths)},
            "sequence_alphabet_record_counts": dict(sorted(alphabet_record_counts.items())),
            "sequence_alphabet_unique_counts": {
                alphabet: sum(sequence_alphabet_class(rows[0]["sequence"]) == alphabet
                              for rows in sequence_rows.values())
                for alphabet in sorted(alphabet_record_counts)
            },
            "source_status_counts": {key: dict(sorted(value.items())) for key, value in source_flag_counts.items()},
            "candidate_pair_annotations_are_approved_truth": False,
            "eligible_training_records": 0,
            "eligible_new_test_records": 0,
        },
        "source_archive_reconstruction": archive,
        "joint_split_v2": {
            "manifest_sha256": sha256(joint_manifest_path),
            "validation_report_sha256": sha256(joint_validation_path),
            "assignments_sha256": sha256(assignment_path),
            "exact_1d_assignment_record_counts": dict(sorted(assignment_counts.items())),
            "exact_1d_assignment_unique_sequence_counts": {
                key: len(values) for key, values in sorted(assignment_sequence_counts.items())
            },
            "exact_paired_record_counts": dict(sorted(paired_record_counts.items())),
            "exact_paired_unique_sequence_counts": {
                key: len(values) for key, values in sorted(paired_sequence_counts.items())
            },
            "raw_pair_set_comparison_to_joint_pairs": dict(sorted(paired_edge_relations.items())),
            "candidate_training_allowed": False,
        },
        "historical_crosswalk": history_summary,
        "decisions": {
            "source_eligibility": "retain all records as unapproved candidates until publication, original PDB/chain, method, coverage, extraction policy, and mapping are independently reviewed per record",
            "exact_sequence_aliases": "do not count exact aliases to current or historical records as new independent evidence",
            "pair_set_comparison": "descriptive only; a match does not validate provenance or labels and a difference does not resolve whether BPSEQ semantics or current masks are correct",
            "new_independent_test": "not established by this audit",
        },
        "limitations": [
            "The source archive and each listed member reproduce the parsed local candidate data, but no upstream URL/download-time receipt was found in the local asset inventory.",
            "The archive member name does not establish an original PDB entry, biological chain, experimental method, coordinate coverage, or annotation procedure.",
            "Exact hashes detect aliases only; this audit performs no alignment-based homology, family, study, or construct-ancestry review.",
            "Raw pair-set equality is not an admission decision and does not compare full observation masks or experimental label scope.",
            "All candidates remain ineligible for training and evaluation; no model predictions were opened and no training was started.",
        ],
        "input_fingerprints": {
            "experimental_sources_audit_sha256": sha256(source_report_path),
            "local_asset_inventory_sha256": sha256(asset_inventory_path),
            "joint_split_validation_sha256": sha256(manifests / "joint_split_v2_validation.json"),
            "historical_source_sha256": {pool: sha256(ROOT / path) for pool, path in HISTORY_SPLITS.items()},
        },
        "training_started": False,
        "test_predictions_read": False,
    }


def main() -> None:
    report = audit()
    output = ROOT / "data/manifests/data_first_comparative_candidate_relations_v1.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit report: {output}")
    output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "path": str(output.relative_to(ROOT)),
        "sha256": sha256(output),
        "candidate_records": report["candidate_source"]["record_count"],
        "unique_sequences": report["candidate_source"]["unique_sequence_count"],
        "candidate_training_allowed": report["candidate_source"]["eligible_training_records"] > 0,
        "training_started": report["training_started"],
        "test_predictions_read": report["test_predictions_read"],
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
