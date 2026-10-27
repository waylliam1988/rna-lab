"""Audit local mmCIF citation and entity metadata for clean-release records.

This report is aggregate-only. It does not admit records, alter labels, create a
split, or inspect model outputs. Source values and row-level identifiers remain
local to the process and are never written to the report.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re

import gemmi

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def cif_rows(block: gemmi.cif.Block, tags: list[str]) -> list[list[str]]:
    try:
        table = block.find(tags)
    except (RuntimeError, ValueError):
        return []
    return [[gemmi.cif.as_string(value) for value in row] for row in table]


def is_known(value: str | None) -> bool:
    return value is not None and value.strip() not in {"", ".", "?"}


def normalize_polymer_sequence(value: str) -> str:
    return re.sub(r"\s+", "", value).upper()


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        self.parent[self.find(right)] = self.find(left)


def local_source_index() -> dict[str, list[Path]]:
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in (ROOT / "data").rglob("*"):
        if path.is_file() and path.suffix.lower() in {".cif", ".mmcif"}:
            by_name[path.name.casefold()].append(path)
    return by_name


def verified_study_map(
    records: list[dict], acquisition_path: Path, verification_path: Path
) -> tuple[dict[str, list[str]], dict[str, int]]:
    locations: dict[str, dict] = {}
    for report_path in (acquisition_path, verification_path):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for receipt in report["results"]:
            if receipt.get("kind") == "pdb_cif_gz" and str(receipt.get("status", "")).startswith("verified"):
                locations[receipt["id"].split(":")[-1]] = {
                    **receipt,
                    "source_collection": "coordinate_source_report",
                }
    raw_artifacts_path = ROOT / "data/manifests/raw_artifacts_v1.jsonl"
    raw_artifacts_summary_path = ROOT / "data/manifests/raw_artifacts_v1_summary.json"
    raw_artifacts_summary = json.loads(raw_artifacts_summary_path.read_text(encoding="utf-8"))
    if raw_artifacts_summary.get("manifest_sha256") != sha256(raw_artifacts_path):
        raise ValueError("Raw-artifact manifest differs from its recorded fingerprint")
    raw_cif_receipts = {
        row["id"].split(":")[-1].upper(): row
        for row in read_jsonl(raw_artifacts_path)
        if row.get("kind") == "pdb_cif"
    }
    for path in (ROOT / "data/raw/experimental_cif_v1").glob("*.cif"):
        receipt_path = path.with_suffix(".receipt.json")
        if not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        manifest_receipt = raw_cif_receipts.get(path.stem.upper())
        if manifest_receipt is None or receipt.get("sha256") != manifest_receipt.get("sha256"):
            raise ValueError("A local raw CIF receipt differs from the frozen raw-artifact manifest")
        locations[path.stem.upper()] = {
            **receipt,
            "id": "raw-cif:" + path.stem.upper(),
            "kind": "pdb_cif",
            "path": str(path.relative_to(ROOT)),
            "source_collection": "raw_experimental_cif_receipt",
            "bytes": manifest_receipt.get("bytes"),
        }

    identifiers = {
        value.split("_")[0]
        for record in records
        for value in record.get("alternative_structure_ids", [record["id"]])
    }
    study_keys: dict[str, list[str]] = {}
    local_sources = 0
    verified_hashes = 0
    source_collections: Counter = Counter()
    for identifier in sorted(identifiers):
        receipt = locations.get(identifier)
        if receipt is None:
            study_keys[identifier] = []
            continue
        path = ROOT / receipt["path"]
        if not path.is_file():
            study_keys[identifier] = []
            continue
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Related mmCIF source path escapes the repository root")
        local_sources += 1
        source_collections[receipt.get("source_collection", "unknown")] += 1
        stored, payload = source_bytes(path)
        expected_hash = receipt.get("sha256")
        if not isinstance(expected_hash, str) or hashlib.sha256(stored).hexdigest() != expected_hash.lower():
            raise ValueError("A locally resolved related mmCIF failed its recorded source checksum")
        if receipt.get("bytes") is not None and len(stored) != receipt["bytes"]:
            raise ValueError("A locally resolved related mmCIF differs from its receipt size")
        verified_hashes += 1
        block = gemmi.cif.read_string(payload.decode("utf-8")).sole_block()
        citations = cif_rows(block, [
            "_citation.id", "_citation.pdbx_database_id_PubMed", "_citation.pdbx_database_id_DOI",
        ])
        primary = [row for row in citations if row and row[0].casefold() == "primary"]
        keys = []
        for row in primary:
            if len(row) > 1 and is_known(row[1]):
                keys.append("pmid:" + row[1])
            if len(row) > 2 and is_known(row[2]):
                keys.append("doi:" + row[2].lower())
        study_keys[identifier] = sorted(set(keys))
    return study_keys, {
        "distinct_related_entries": len(identifiers),
        "entries_with_local_sources": local_sources,
        "entries_with_recorded_source_hash_match": verified_hashes,
        "local_source_receipt_collections": dict(sorted(source_collections.items())),
        "entries_missing_local_sources": len(identifiers) - local_sources,
    }


def resolve_source(record: dict, by_name: dict[str, list[Path]]) -> tuple[Path, str]:
    receipt = record.get("source_download_receipt") or {}
    receipt_path = receipt.get("path")
    if isinstance(receipt_path, str) and receipt_path:
        path = Path(receipt_path)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            return path, "receipt_path"

    source_member = record.get("source_member")
    if isinstance(source_member, str) and source_member:
        matches = by_name.get(Path(source_member).name.casefold(), [])
        if len(matches) == 1:
            return matches[0], "unique_source_member_basename"
    raise FileNotFoundError("A clean-release source file could not be resolved locally")


def source_bytes(path: Path) -> tuple[bytes, bytes]:
    stored = path.read_bytes()
    payload = gzip.decompress(stored) if path.suffix.lower() == ".gz" else stored
    return stored, payload


def summarize_record(block: gemmi.cif.Block, record: dict) -> dict:
    entry_ids = block.find_values("_entry.id")
    entry_matches = any(
        gemmi.cif.as_string(value).casefold() == str(record.get("pdb_id", "")).casefold()
        for value in entry_ids
    )

    citations = cif_rows(block, [
        "_citation.id",
        "_citation.pdbx_database_id_PubMed",
        "_citation.pdbx_database_id_DOI",
        "_citation.title",
    ])
    primary = [row for row in citations if row and row[0].casefold() == "primary"]
    primary_dois = {row[2].strip().casefold() for row in primary if len(row) > 2 and is_known(row[2])}
    primary_pmids = {row[1].strip() for row in primary if len(row) > 1 and is_known(row[1])}
    primary_titles = {row[3] for row in primary if len(row) > 3 and is_known(row[3])}

    entity_id = str(record.get("entity_id", ""))
    entities = cif_rows(block, [
        "_entity.id", "_entity.src_method", "_entity.pdbx_description",
        "_entity.pdbx_mutation", "_entity.pdbx_fragment",
    ])
    entity_rows = [row for row in entities if row and row[0] == entity_id]
    entity = entity_rows[0] if entity_rows else []

    asym_rows = cif_rows(block, ["_struct_asym.id", "_struct_asym.entity_id"])
    label_asym = str(record.get("label_asym_id", ""))
    label_entity_match = any(row[0] == label_asym and row[1] == entity_id for row in asym_rows if len(row) > 1)

    auth_asym = str(record.get("auth_asym_id", ""))
    scheme_rows = cif_rows(block, ["_pdbx_poly_seq_scheme.asym_id", "_pdbx_poly_seq_scheme.pdb_strand_id"])
    label_auth_mapping_match = any(
        row[0] == label_asym and row[1] == auth_asym for row in scheme_rows if len(row) > 1
    )

    polymer_rows = cif_rows(block, ["_entity_poly.entity_id", "_entity_poly.pdbx_seq_one_letter_code_can"])
    canonical_sequences = [row[1] for row in polymer_rows if len(row) > 1 and row[0] == entity_id]
    target_sequence = str(record.get("sequence", "")).upper()
    exact_polymer_sequence_match = any(
        normalize_polymer_sequence(sequence) == target_sequence for sequence in canonical_sequences
    )

    source_gen_rows = cif_rows(block, ["_entity_src_gen.entity_id"])
    source_nat_rows = cif_rows(block, ["_entity_src_nat.entity_id"])
    source_syn_rows = cif_rows(block, ["_entity_src_syn.entity_id"])
    ref_rows = cif_rows(block, ["_struct_ref.entity_id", "_struct_ref.id"])
    entity_reference_ids = {
        row[1] for row in ref_rows if len(row) > 1 and row[0] == entity_id and is_known(row[1])
    }
    ref_seq_rows = cif_rows(block, ["_struct_ref_seq.ref_id"])
    has_reference_sequence_mapping = any(row and row[0] in entity_reference_ids for row in ref_seq_rows)

    return {
        "entry_id_match": entry_matches,
        "has_any_citation_row": bool(citations),
        "has_primary_citation": bool(primary),
        "has_primary_title": bool(primary_titles),
        "has_primary_doi": bool(primary_dois),
        "has_primary_pubmed_id": bool(primary_pmids),
        "primary_dois": primary_dois,
        "primary_pmids": primary_pmids,
        "label_asym_entity_match": label_entity_match,
        "label_auth_mapping_match": label_auth_mapping_match,
        "entity_row_match": bool(entity),
        "canonical_entity_sequence_present": bool(canonical_sequences),
        "canonical_entity_sequence_exact_match": exact_polymer_sequence_match,
        "entity_src_method_known": len(entity) > 1 and is_known(entity[1]),
        "entity_description_known": len(entity) > 2 and is_known(entity[2]),
        "entity_mutation_known": len(entity) > 3 and is_known(entity[3]),
        "entity_fragment_known": len(entity) > 4 and is_known(entity[4]),
        "entity_source_gen_present": any(row and row[0] == entity_id for row in source_gen_rows),
        "entity_source_nat_present": any(row and row[0] == entity_id for row in source_nat_rows),
        "entity_source_syn_present": any(row and row[0] == entity_id for row in source_syn_rows),
        "entity_reference_present": bool(entity_reference_ids),
        "entity_reference_sequence_mapping_present": has_reference_sequence_mapping,
    }


def audit() -> dict:
    manifests = ROOT / "data/manifests"
    clean_dir = ROOT / "data/experimental/clean_release_v1"
    pair_path = clean_dir / "paired_deduplicated.jsonl"
    summary_path = clean_dir / "summary.json"
    validation_path = manifests / "clean_release_validation_v1.json"
    prior_audit_path = manifests / "data_first_clean_release_eval_candidate_audit_v1.json"
    receipt_report_path = ROOT / "data/experimental/verified_coordinate_v1/source_verification_report.json"
    acquisition_report_path = ROOT / "data/reports/experimental_acquisition_v1_all.json"
    raw_artifacts_path = manifests / "raw_artifacts_v1.jsonl"
    raw_artifacts_summary_path = manifests / "raw_artifacts_v1_summary.json"
    output_path = manifests / "data_first_clean_release_cif_metadata_audit_v1.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit report: {output_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    prior_audit = json.loads(prior_audit_path.read_text(encoding="utf-8"))
    if validation.get("passed") is not True or validation.get("training_split_ready") is not False:
        raise ValueError("Clean-release validation or admission state changed")
    if summary.get("fingerprints", {}).get("paired_deduplicated.jsonl") != sha256(pair_path):
        raise ValueError("Clean-release paired input differs from its recorded fingerprint")

    records = read_jsonl(pair_path)
    if len(records) != summary.get("unique_2d_sequences"):
        raise ValueError("Clean-release record count differs from its frozen summary")

    by_name = local_source_index()
    source_resolution = Counter()
    source_paths: set[Path] = set()
    stored_bytes = 0
    payload_bytes = 0
    receipt_hash_verified = 0
    cif_hash_verified = 0
    field_counts: Counter = Counter()
    study_tokens_by_record: list[set[str]] = []

    for record in records:
        path, resolution = resolve_source(record, by_name)
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Resolved source path escapes the repository root")
        source_resolution[resolution] += 1
        source_paths.add(path.resolve())
        stored, payload = source_bytes(path)
        stored_bytes += len(stored)
        payload_bytes += len(payload)

        receipt = record.get("source_download_receipt") or {}
        expected_source_hash = receipt.get("sha256")
        if not isinstance(expected_source_hash, str) or hashlib.sha256(stored).hexdigest() != expected_source_hash.lower():
            raise ValueError("Local source bytes do not match the recorded download checksum")
        receipt_hash_verified += 1
        expected_cif_hash = record.get("cif_sha256")
        if not isinstance(expected_cif_hash, str) or hashlib.sha256(payload).hexdigest() != expected_cif_hash.lower():
            raise ValueError("Local CIF payload does not match the record fingerprint")
        cif_hash_verified += 1

        block = gemmi.cif.read_string(payload.decode("utf-8")).sole_block()
        facts = summarize_record(block, record)
        for key, value in facts.items():
            if isinstance(value, bool):
                field_counts[key] += int(value)

        tokens = {"doi:" + value for value in facts["primary_dois"]}
        tokens.update("pmid:" + value for value in facts["primary_pmids"])
        study_tokens_by_record.append(tokens)

    # Match the existing candidate audit's primary-study rule across every
    # alternative entry, checking local source hashes before parsing metadata.
    studies_by_pdb, related_source_verification = verified_study_map(
        records, acquisition_report_path, receipt_report_path
    )
    related_pdbs_by_record = [
        {value.split("_")[0] for value in record.get("alternative_structure_ids", [record["id"]])}
        for record in records
    ]
    related_study_tokens_by_record = [
        {study for pdb_id in pdb_ids for study in studies_by_pdb.get(pdb_id, [])}
        for pdb_ids in related_pdbs_by_record
    ]
    related_entries_with_study = {
        pdb_id for pdb_ids in related_pdbs_by_record for pdb_id in pdb_ids if studies_by_pdb.get(pdb_id)
    }
    related_entries = set().union(*related_pdbs_by_record)
    records_with_all_related_studies = sum(
        all(studies_by_pdb.get(pdb_id) for pdb_id in pdb_ids) for pdb_ids in related_pdbs_by_record
    )
    records_with_any_related_study_gap = len(records) - records_with_all_related_studies
    related_tokens_by_record = [
        {"doi:" + value.split(":", 1)[1] if value.startswith("doi:") else value for value in studies}
        for studies in related_study_tokens_by_record
    ]
    related_token_records: dict[str, set[int]] = defaultdict(set)
    for index, tokens in enumerate(related_tokens_by_record):
        for token in tokens:
            related_token_records[token].add(index)
    related_shared_indices = {
        index for indices in related_token_records.values() if len(indices) > 1 for index in indices
    }

    # Count only aggregate study-link coverage; never serialize DOI/PMID values.
    token_records: dict[str, set[int]] = defaultdict(set)
    for index, tokens in enumerate(study_tokens_by_record):
        for token in tokens:
            token_records[token].add(index)
    study_groups = DisjointSet(len(records))
    for indices in token_records.values():
        indices = sorted(indices)
        for index in indices[1:]:
            study_groups.union(indices[0], index)
    linked_indices = {index for indices in token_records.values() for index in indices}
    shared_indices = {index for indices in token_records.values() if len(indices) > 1 for index in indices}
    linked_group_sizes = Counter(study_groups.find(index) for index in linked_indices)

    if receipt_hash_verified != len(records) or cif_hash_verified != len(records):
        raise AssertionError("Not all local sources passed checksum verification")

    prior_pool = prior_audit["clean_release_pool"]
    report = {
        "audit": "clean-release-cif-metadata-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Read-only local mmCIF source, primary-citation, selected-chain/entity mapping, and construct-related metadata coverage. Aggregate counts only; no row-level source values are emitted.",
        "clean_release_records": len(records),
        "local_source_verification": {
            "records_with_local_source": len(records),
            "source_resolution_methods": dict(sorted(source_resolution.items())),
            "unique_local_source_files": len(source_paths),
            "stored_bytes_read": stored_bytes,
            "decompressed_cif_bytes_read": payload_bytes,
            "download_receipt_sha256_matches": receipt_hash_verified,
            "record_cif_sha256_matches": cif_hash_verified,
        },
        "citation_coverage": {
            "records_with_any_citation_row": field_counts["has_any_citation_row"],
            "records_with_primary_citation_row": field_counts["has_primary_citation"],
            "records_with_primary_title": field_counts["has_primary_title"],
            "records_with_primary_doi": field_counts["has_primary_doi"],
            "records_with_primary_pubmed_id": field_counts["has_primary_pubmed_id"],
            "distinct_primary_doi_count": len({token for tokens in study_tokens_by_record for token in tokens if token.startswith("doi:")}),
            "distinct_primary_pubmed_id_count": len({token for tokens in study_tokens_by_record for token in tokens if token.startswith("pmid:")}),
            "records_with_any_primary_study_key": len(linked_indices),
            "distinct_primary_study_keys": len(token_records),
            "records_linked_to_another_record_by_primary_study": len(shared_indices),
            "linked_study_group_count": sum(size > 1 for size in linked_group_sizes.values()),
        },
        "alternative_structure_entry_study_coverage": {
            "distinct_related_structure_entries": len(related_entries),
            "related_entry_local_source_verification": related_source_verification,
            "related_entries_with_primary_study_key": len(related_entries_with_study),
            "records_with_primary_study_key_for_every_related_entry": records_with_all_related_studies,
            "records_with_at_least_one_related_entry_missing_primary_study_key": records_with_any_related_study_gap,
            "records_linked_to_another_record_by_related_primary_study": len(related_shared_indices),
            "distinct_related_primary_study_keys": len(related_token_records),
        },
        "selected_chain_entity_crosswalk": {
            "entry_id_matches_record": field_counts["entry_id_match"],
            "label_chain_maps_to_record_entity": field_counts["label_asym_entity_match"],
            "label_author_chain_mapping_matches": field_counts["label_auth_mapping_match"],
            "selected_entity_row_present": field_counts["entity_row_match"],
            "canonical_entity_sequence_present": field_counts["canonical_entity_sequence_present"],
            "canonical_entity_sequence_exactly_matches_record": field_counts["canonical_entity_sequence_exact_match"],
        },
        "selected_entity_construct_related_metadata": {
            "entity_source_method_known": field_counts["entity_src_method_known"],
            "entity_description_known": field_counts["entity_description_known"],
            "entity_mutation_known": field_counts["entity_mutation_known"],
            "entity_fragment_known": field_counts["entity_fragment_known"],
            "entity_genetic_source_row_present": field_counts["entity_source_gen_present"],
            "entity_natural_source_row_present": field_counts["entity_source_nat_present"],
            "entity_synthetic_source_row_present": field_counts["entity_source_syn_present"],
            "entity_sequence_reference_present": field_counts["entity_reference_present"],
            "entity_reference_alignment_present": field_counts["entity_reference_sequence_mapping_present"],
            "clean_json_records_with_explicit_construct_provenance": sum(
                bool(record.get("construct_provenance") or record.get("construct_id")) for record in records
            ),
        },
        "relationship_to_existing_candidate_audit": {
            "prior_candidate_audit_sha256": sha256(prior_audit_path),
            "prior_new_independent_test_eligible_records": prior_audit["candidate_relation_graph_to_frozen_joint"]["candidate_test_eligible_records"],
            "new_independent_test_established": False,
            "training_admission_changed": False,
        },
        "decisions": {
            "citation_use": "Primary DOI/PubMed values may support PDB-entry study links; they do not establish construct ancestry, assay independence, or assessment blindness.",
            "construct_use": "Entity source, description, mutation, fragment, polymer sequence, and reference fields are supporting metadata only; this audit does not establish parent-construct lineage or independent constructs.",
            "record_mapping": "Chain/entity/sequence matches confirm which mmCIF entity underlies each clean-release record; they do not prove that different entities are biologically independent.",
            "admission": "No training or evaluation eligibility is added; existing quarantine and frozen split decisions remain unchanged.",
        },
        "limitations": [
            "Citation coverage is metadata presence, not manual verification against the article or a study-level independence review.",
            "Entity source and polymer reference annotations can describe expression source or sequence similarity without documenting the full construct parent, engineered changes, or biological replicate.",
            "Exact entity-sequence and chain crosswalks do not replace family/remote-homology review.",
            "No model prediction, historical test score, or test-set outcome was read; no data, labels, or split assignments were modified.",
            "The report contains aggregate counts and input fingerprints only; row-level identifiers, sequence, citation values, and URLs are not emitted.",
        ],
        "input_fingerprints": {
            "clean_release_pairs_sha256": sha256(pair_path),
            "clean_release_summary_sha256": sha256(summary_path),
            "clean_release_validation_sha256": sha256(validation_path),
            "coordinate_source_verification_sha256": sha256(receipt_report_path),
            "experimental_acquisition_report_sha256": sha256(acquisition_report_path),
            "raw_artifacts_manifest_sha256": sha256(raw_artifacts_path),
            "raw_artifacts_summary_sha256": sha256(raw_artifacts_summary_path),
            "prior_candidate_audit_sha256": sha256(prior_audit_path),
        },
        "test_predictions_read": False,
        "training_started": False,
        "source_data_modified": False,
    }
    return report


def main() -> None:
    report = audit()
    path = ROOT / "data/manifests/data_first_clean_release_cif_metadata_audit_v1.json"
    path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "records": report["clean_release_records"],
        "local_sources": report["local_source_verification"]["records_with_local_source"],
        "receipt_hash_matches": report["local_source_verification"]["download_receipt_sha256_matches"],
        "cif_hash_matches": report["local_source_verification"]["record_cif_sha256_matches"],
        "primary_citation_records": report["citation_coverage"]["records_with_any_primary_study_key"],
        "new_independent_test_established": report["relationship_to_existing_candidate_audit"]["new_independent_test_established"],
        "test_predictions_read": report["test_predictions_read"],
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
