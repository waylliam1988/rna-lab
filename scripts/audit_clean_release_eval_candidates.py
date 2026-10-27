"""Assess clean-release records as possible new independent evaluation groups."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_joint_splits import sequence_edges, study_map


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


def classify_candidate_group(
    current_splits: set[str],
    historical_test_exposed: bool,
    has_verified_homology_component: bool,
    has_primary_study: bool,
    has_construct_provenance: bool,
) -> str:
    """Classify relation evidence without ever approving a candidate for test."""
    if historical_test_exposed:
        return "historically_test_exposed"
    if len(current_splits) > 1:
        return "bridges_multiple_frozen_splits"
    if current_splits:
        return f"connected_to_current_{next(iter(current_splits))}"
    if not (has_verified_homology_component and has_primary_study and has_construct_provenance):
        return "no_known_current_link_but_independence_evidence_incomplete"
    return "unlinked_candidate_requires_manual_review"


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        self.parent[self.find(right)] = self.find(left)


def audit() -> dict:
    manifests = ROOT / "data/manifests"
    clean_directory = ROOT / "data/experimental/clean_release_v1"
    clean_pairs_path = clean_directory / "paired_deduplicated.jsonl"
    clean_summary_path = clean_directory / "summary.json"
    clean_validation_path = manifests / "clean_release_validation_v1.json"
    rna3db_path = ROOT / "data/experimental/rna3db_all_chain_metadata.jsonl"
    joint_directory = ROOT / "data/experimental/joint_split_v2"
    joint_manifest_path = manifests / "joint_split_v2.json"
    joint_validation_path = manifests / "joint_split_v2_validation.json"
    acquisition_path = ROOT / "data/reports/experimental_acquisition_v1_all.json"
    coordinate_receipts_path = ROOT / "data/experimental/verified_coordinate_v1/source_verification_report.json"
    output_path = manifests / "data_first_clean_release_eval_candidate_audit_v1.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit report: {output_path}")

    clean_summary = json.loads(clean_summary_path.read_text(encoding="utf-8"))
    clean_validation = json.loads(clean_validation_path.read_text(encoding="utf-8"))
    joint_manifest = json.loads(joint_manifest_path.read_text(encoding="utf-8"))
    joint_validation = json.loads(joint_validation_path.read_text(encoding="utf-8"))
    if clean_validation.get("passed") is not True or clean_validation.get("training_split_ready") is not False:
        raise ValueError("Clean-release evidence validation or admission state changed")
    if clean_summary.get("training_split_ready") is not False or clean_summary.get("fingerprints", {}).get("paired_deduplicated.jsonl") != sha256(clean_pairs_path):
        raise ValueError("Clean-release input differs from its published frozen fingerprint")
    if joint_manifest.get("validation_passed") is not True or joint_validation.get("passed") is not True:
        raise ValueError("Frozen joint split does not pass its published validation")

    clean_records = list(read_jsonl(clean_pairs_path))
    if len(clean_records) != clean_summary["unique_2d_sequences"]:
        raise ValueError("Clean-release paired record count differs from summary")
    if len({row["sequence_sha256"] for row in clean_records}) != len(clean_records):
        raise ValueError("Clean-release pair sequence hashes are not unique")

    current_records = []
    current_hashes = {}
    for split in ("train", "validation", "test"):
        path = joint_directory / f"{split}.jsonl"
        if sha256(path) != joint_validation["files"][split]:
            raise ValueError(f"Current {split} file differs from its frozen validation fingerprint")
        rows = list(read_jsonl(path))
        if len(rows) != joint_manifest["splits"][split]:
            raise ValueError(f"Current {split} record count differs from frozen manifest")
        for row in rows:
            if hashlib.sha256(row["sequence"].encode("utf-8")).hexdigest() != row["sequence_sha256"]:
                raise ValueError(f"Current sequence SHA256 mismatch: {row.get('id')}")
            if row["sequence_sha256"] in current_hashes:
                raise ValueError("Duplicate sequence across current joint splits")
            current_hashes[row["sequence_sha256"]] = split
            current_records.append({**row, "_graph_id": f"joint:{split}:{row['id']}", "_kind": "current", "_split": split})

    # Use the same RNA3DB exact-sequence component map and primary-CIF citation
    # parser as the frozen joint split builder. component_0 remains unresolved.
    components_by_sequence: dict[str, set[str]] = defaultdict(set)
    rna3db_rows = 0
    for row in read_jsonl(rna3db_path):
        rna3db_rows += 1
        component = row.get("homology_component")
        if component and component != "component_0":
            components_by_sequence[row["sequence"]].add(component)
    studies_by_pdb = study_map(clean_records)

    for row in clean_records:
        sequence_hash = hashlib.sha256(row["sequence"].encode("utf-8")).hexdigest()
        if sequence_hash != row["sequence_sha256"]:
            raise ValueError(f"Clean-release sequence SHA256 mismatch: {row.get('id')}")
        row["_graph_id"] = f"clean:{row['id']}"
        row["_kind"] = "clean"
        row["_split"] = None
        row["_components"] = sorted(components_by_sequence.get(row["sequence"], set()))
        row["_pdb_ids"] = sorted({value.split("_")[0] for value in row.get("alternative_structure_ids", [row["id"]])})
        row["_study_ids"] = sorted({study for pdb_id in row["_pdb_ids"] for study in studies_by_pdb.get(pdb_id, [])})
        row["_study_metadata_complete"] = all(studies_by_pdb.get(pdb_id) for pdb_id in row["_pdb_ids"])

    historical_test_paths = {
        "legacy_processed:test": ROOT / "data/processed/test.jsonl",
        "legacy_processed:test_external": ROOT / "data/processed/test_external.jsonl",
        "paired_v0_2:test": ROOT / "data/experimental/paired_v0_2/test.jsonl",
    }
    historical_test_hashes = {}
    historical_test_counts = {}
    for pool, path in historical_test_paths.items():
        hashes = set()
        for row in read_jsonl(path):
            sequence_hash = hashlib.sha256(row["sequence"].encode("utf-8")).hexdigest()
            if row.get("sequence_sha256") != sequence_hash:
                raise ValueError(f"Historical test sequence hash mismatch in {pool}: {row.get('id')}")
            hashes.add(sequence_hash)
        historical_test_hashes[pool] = hashes
        historical_test_counts[pool] = len(hashes)

    all_rows = current_records + clean_records
    by_id = {row["_graph_id"]: index for index, row in enumerate(all_rows)}
    groups = DisjointSet(len(all_rows))
    token_owner = {}
    for index, row in enumerate(all_rows):
        if row["_kind"] == "current":
            components = [value for value in row.get("homology_components", []) if value != "component_0"]
            pdb_ids = row.get("source_pdb_ids", [])
            studies = row.get("study_ids", [])
        else:
            components = row["_components"]
            pdb_ids = row["_pdb_ids"]
            studies = row["_study_ids"]
        tokens = (
            ["sequence:" + row["sequence_sha256"]]
            + ["homology:" + value for value in components]
            + ["study:" + value for value in studies]
            + ["pdb:" + value for value in pdb_ids]
        )
        for token in tokens:
            if token in token_owner:
                groups.union(index, token_owner[token])
            token_owner[token] = index

    similarity_records = [
        {"id": row["_graph_id"], "sequence": row["sequence"]}
        for row in all_rows
    ]
    alignment_edges = list(sequence_edges(similarity_records))
    for edge in alignment_edges:
        groups.union(by_id[edge["a"]], by_id[edge["b"]])

    current_splits_by_group: dict[int, set[str]] = defaultdict(set)
    clean_rows_by_group: dict[int, list[dict]] = defaultdict(list)
    for index, row in enumerate(all_rows):
        root = groups.find(index)
        if row["_kind"] == "current":
            current_splits_by_group[root].add(row["_split"])
        else:
            clean_rows_by_group[root].append(row)

    exact_current_count = sum(row["sequence_sha256"] in current_hashes for row in clean_records)
    exact_history_counts = {
        pool: sum(row["sequence_sha256"] in hashes for row in clean_records)
        for pool, hashes in historical_test_hashes.items()
    }
    row_decisions = Counter()
    connected_record_counts = Counter()
    connected_unique_sequences = defaultdict(set)
    metadata_counts = Counter()
    for row in clean_records:
        root = groups.find(by_id[row["_graph_id"]])
        split_links = current_splits_by_group[root]
        historical_exposed = any(row["sequence_sha256"] in hashes for hashes in historical_test_hashes.values())
        has_component = bool(row["_components"])
        has_construct = bool(row.get("construct_provenance") or row.get("construct_id"))
        status = classify_candidate_group(
            split_links,
            historical_exposed,
            has_component,
            row["_study_metadata_complete"],
            has_construct,
        )
        row_decisions[status] += 1
        metadata_counts["nonzero_rna3db_homology_component" if has_component else "no_nonzero_rna3db_homology_component"] += 1
        metadata_counts["primary_study_complete" if row["_study_metadata_complete"] else "primary_study_incomplete"] += 1
        metadata_counts["construct_provenance_present" if has_construct else "construct_provenance_missing"] += 1
        metadata_counts["single_RNA_context" if row.get("context") == "single_RNA" else "complex_or_other_context"] += 1
        for split in split_links:
            connected_record_counts[split] += 1
            connected_unique_sequences[split].add(row["sequence_sha256"])

    nonexact = [row for row in clean_records if row["sequence_sha256"] not in current_hashes]
    nonexact_group_roots = {groups.find(by_id[row["_graph_id"]]) for row in nonexact}
    group_status_counts = Counter()
    for root in nonexact_group_roots:
        split_links = current_splits_by_group[root]
        rows = clean_rows_by_group[root]
        historical_exposed = any(row["sequence_sha256"] in hashes for row in rows for hashes in historical_test_hashes.values())
        representative = rows[0]
        has_component = all(bool(row["_components"]) for row in rows)
        has_study = all(bool(row["_study_metadata_complete"]) for row in rows)
        has_construct = all(bool(row.get("construct_provenance") or row.get("construct_id")) for row in rows)
        group_status_counts[classify_candidate_group(split_links, historical_exposed, has_component, has_study, has_construct)] += 1

    exact_test_hashes = {
        pool: hashes for pool, hashes in historical_test_hashes.items()
    }
    exact_old_test_exposed = set().union(*exact_test_hashes.values())
    clean_summary_counts = {
        "provenance_verified": Counter(str(row.get("provenance_verified")).lower() for row in clean_records),
        "quality_verified": Counter(str(row.get("quality_verified")).lower() for row in clean_records),
        "contains_model_predictions": Counter(str(row.get("contains_model_predictions")).lower() for row in clean_records),
        "pair_labels_available": Counter(str(row.get("pair_labels_available")).lower() for row in clean_records),
    }
    return {
        "audit": "clean-release-independent-evaluation-candidate-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Read-only exact, RNA3DB component, primary-study, PDB-entry, and project-threshold sequence relation audit. No test set is created or admitted.",
        "clean_release_pool": {
            "paired_file_sha256": sha256(clean_pairs_path),
            "paired_records": len(clean_records),
            "exact_sequence_aliases_to_current_joint": exact_current_count,
            "nonexact_sequence_records_to_current_joint": len(nonexact),
            "evidence_flags": {key: dict(sorted(value.items())) for key, value in clean_summary_counts.items()},
            "rna3db_records_reviewed": rna3db_rows,
            "primary_citation_missing_clean_records": sum(not row["_study_metadata_complete"] for row in clean_records),
            "clean_records_with_nonzero_rna3db_component": sum(bool(row["_components"]) for row in clean_records),
            "clean_records_with_explicit_construct_provenance": sum(bool(row.get("construct_provenance") or row.get("construct_id")) for row in clean_records),
            "clean_records_by_context": dict(sorted(Counter(row.get("context", "unknown") for row in clean_records).items())),
        },
        "candidate_relation_graph_to_frozen_joint": {
            "sequence_edge_rule": "same as build_joint_splits.py: local alignment, identity>=0.8 and shorter-sequence coverage>=0.8",
            "alignment_edge_count": len(alignment_edges),
            "nonexact_records_connected_to_current_split": dict(sorted(connected_record_counts.items())),
            "nonexact_unique_sequences_connected_to_current_split": {
                split: len(values) for split, values in sorted(connected_unique_sequences.items())
            },
            "nonexact_group_count": len(nonexact_group_roots),
            "nonexact_group_relation_counts": dict(sorted(group_status_counts.items())),
            "all_clean_record_relation_counts": dict(sorted(row_decisions.items())),
            "candidate_test_eligible_records": 0,
            "new_independent_test_established": False,
        },
        "historical_test_exact_aliases": {
            pool: {"historical_test_records": historical_test_counts[pool], "clean_release_exact_aliases": exact_history_counts[pool]}
            for pool in historical_test_paths
        },
        "decisions": {
            "exact_current_aliases": "retain their frozen train/validation/test role; do not duplicate into a new test",
            "current_group_connections": "exclude from new independent-test candidates, including nonexact study/PDB/homology/alignment connections",
            "unconnected_groups": "remain quarantine from evaluation until family, primary-study, construct, and assessment-blindness evidence is complete",
            "training": "no clean_release records are made training-ready by this candidate audit",
        },
        "limitations": [
            "RNA3DB components are matched only by exact sequence; absent component metadata is unknown, not evidence of no homology.",
            "PDB primary citations do not establish construct ancestry, assay independence, or that the structure is independent of a prior evaluation.",
            "The sequence alignment rule is a relation screen, not proof of remote-homology independence.",
            "Only exact aliases to historical test/test_external records are counted here; historical similarity relationships require separate evidence.",
            "Complex-context records cannot automatically be interpreted as isolated-RNA folding labels.",
            "No model predictions or historical model scores were opened; no test was selected from performance.",
        ],
        "input_fingerprints": {
            "clean_release_summary_sha256": sha256(clean_summary_path),
            "clean_release_validation_sha256": sha256(clean_validation_path),
            "rna3db_all_chain_metadata_sha256": sha256(rna3db_path),
            "joint_manifest_sha256": sha256(joint_manifest_path),
            "joint_validation_sha256": sha256(joint_validation_path),
            "joint_1d_assignments_sha256": sha256(joint_directory / "joint_1d_assignments.jsonl"),
            "primary_study_sources": {
                "experimental_acquisition_sha256": sha256(acquisition_path),
                "coordinate_source_verification_sha256": sha256(coordinate_receipts_path),
            },
            "historical_test_source_sha256": {pool: sha256(path) for pool, path in historical_test_paths.items()},
        },
        "test_predictions_read": False,
        "training_started": False,
    }


def main() -> None:
    report = audit()
    path = ROOT / "data/manifests/data_first_clean_release_eval_candidate_audit_v1.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite audit report: {path}")
    path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "clean_records": report["clean_release_pool"]["paired_records"],
        "current_exact_aliases": report["clean_release_pool"]["exact_sequence_aliases_to_current_joint"],
        "new_independent_test_established": report["candidate_relation_graph_to_frozen_joint"]["new_independent_test_established"],
        "test_predictions_read": report["test_predictions_read"],
        "training_started": report["training_started"],
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
