"""Resolve remaining PDB primary-citation metadata from the official RCSB API.

Only entries missing a primary-study key in the local verified mmCIF sources
are queried. Raw responses and the per-entry ledger stay local; the published
report contains aggregate counts and fingerprints only. No data is admitted.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_clean_release_cif_metadata import (
    read_jsonl,
    sha256,
    verified_study_map,
)


def citation_keys_from_api(payload: dict) -> set[str]:
    citation = payload.get("rcsb_primary_citation")
    if not isinstance(citation, dict):
        return set()
    result = set()
    doi = citation.get("pdbx_database_id_DOI")
    pmid = citation.get("pdbx_database_id_PubMed")
    if isinstance(doi, str) and doi.strip() not in {"", ".", "?"}:
        result.add("doi:" + doi.strip().lower())
    if pmid is not None and str(pmid).strip() not in {"", ".", "?"}:
        result.add("pmid:" + str(pmid).strip())
    return result


def is_valid_response_for_entry(payload: dict, entry_id: str) -> bool:
    response_id = payload.get("rcsb_id")
    return isinstance(response_id, str) and response_id.upper() == entry_id.upper()


def fetch_entry(entry_id: str) -> tuple[int | None, bytes | None, str]:
    url = f"https://data.rcsb.org/rest/v1/core/entry/{entry_id}"
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "RNA-folding-provenance-audit/1.0"})
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read(5 * 1024 * 1024 + 1)
            if len(raw) > 5 * 1024 * 1024:
                return response.status, None, "response_exceeds_size_limit"
            return response.status, raw, "received"
    except HTTPError as exc:
        try:
            body = exc.read(5 * 1024 * 1024 + 1)
            if len(body) > 5 * 1024 * 1024:
                body = None
        except Exception:
            body = None
        return exc.code, body, "http_error"
    except (URLError, TimeoutError, OSError):
        return None, None, "network_error"


def audit() -> dict:
    manifests = ROOT / "data/manifests"
    pair_path = ROOT / "data/experimental/clean_release_v1/paired_deduplicated.jsonl"
    prior_path = manifests / "data_first_clean_release_cif_metadata_audit_v1.json"
    acquisition_path = ROOT / "data/reports/experimental_acquisition_v1_all.json"
    verification_path = ROOT / "data/experimental/verified_coordinate_v1/source_verification_report.json"
    response_dir = ROOT / "data/raw/primary_citation_review_v1"
    detail_path = manifests / "data_first_clean_release_citation_review_v1.jsonl"
    output_path = manifests / "data_first_clean_release_remote_citations_v1.json"

    if detail_path.exists() or output_path.exists() or response_dir.exists():
        raise FileExistsError("Refusing to overwrite existing remote-citation review artifacts")
    if not prior_path.is_file():
        raise FileNotFoundError("The local CIF metadata audit is required")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    records = read_jsonl(pair_path)
    studies, _ = verified_study_map(records, acquisition_path, verification_path)
    related_by_record = [
        {value.split("_")[0] for value in record.get("alternative_structure_ids", [record["id"]])}
        for record in records
    ]
    entries = set().union(*related_by_record)
    missing_entries = sorted(entry for entry in entries if not studies.get(entry))
    if len(missing_entries) != prior["alternative_structure_entry_study_coverage"]["distinct_related_structure_entries"] - prior["alternative_structure_entry_study_coverage"]["related_entries_with_primary_study_key"]:
        raise ValueError("Current local citation gaps differ from the frozen aggregate audit")

    response_dir.mkdir(parents=True, exist_ok=False)
    results = []
    fetched_keys: dict[str, set[str]] = {}
    response_bytes = 0
    for index, entry_id in enumerate(missing_entries):
        status, raw, state = fetch_entry(entry_id)
        item = {
            "entry_id": entry_id,
            "source_url": f"https://data.rcsb.org/rest/v1/core/entry/{entry_id}",
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "http_status": status,
            "state": state,
            "response_sha256": None,
            "response_bytes": None,
            "entry_identity_verified": False,
            "primary_citation_object_present": False,
            "primary_doi_present": False,
            "primary_pubmed_id_present": False,
            "usable_primary_study_key_present": False,
        }
        keys: set[str] = set()
        if raw is not None:
            response_bytes += len(raw)
            item["response_sha256"] = hashlib.sha256(raw).hexdigest()
            item["response_bytes"] = len(raw)
            response_path = response_dir / f"{entry_id.lower()}.json"
            receipt_path = response_path.with_suffix(".receipt.json")
            response_path.write_bytes(raw)
            receipt_path.write_text(json.dumps({
                "entry_id": entry_id,
                "source_url": item["source_url"],
                "retrieved_at_utc": item["retrieved_at_utc"],
                "http_status": status,
                "sha256": item["response_sha256"],
                "bytes": len(raw),
            }, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
            if status == 200:
                try:
                    payload = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    item["state"] = "invalid_json"
                else:
                    item["entry_identity_verified"] = is_valid_response_for_entry(payload, entry_id)
                    if item["entry_identity_verified"]:
                        item["primary_citation_object_present"] = isinstance(payload.get("rcsb_primary_citation"), dict)
                        keys = citation_keys_from_api(payload)
                        item["primary_doi_present"] = any(key.startswith("doi:") for key in keys)
                        item["primary_pubmed_id_present"] = any(key.startswith("pmid:") for key in keys)
                        item["usable_primary_study_key_present"] = bool(keys)
                        item["state"] = "entry_identity_verified"
                    else:
                        item["state"] = "entry_identity_mismatch"
        fetched_keys[entry_id] = keys
        results.append(item)
        if index + 1 < len(missing_entries):
            time.sleep(0.25)

    detail_path.write_text(
        "".join(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n" for item in results),
        encoding="utf-8",
        newline="\n",
    )

    supplemented = {entry: set(studies.get(entry, [])) | fetched_keys.get(entry, set()) for entry in entries}
    before_complete = sum(all(studies.get(entry) for entry in related) for related in related_by_record)
    after_complete = sum(all(supplemented.get(entry) for entry in related) for related in related_by_record)
    entry_counts = Counter(item["state"] for item in results)
    detail_hash = sha256(detail_path)
    response_hashes = [item["response_sha256"] for item in results if item["response_sha256"]]
    report = {
        "audit": "clean-release-official-remote-primary-citation-review-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Limited official RCSB Data API lookup for locally unresolved primary-study keys; row-level identifiers and responses remain local.",
        "query_summary": {
            "source": "RCSB Data API core entry endpoint",
            "entries_missing_local_primary_study_key": len(missing_entries),
            "entries_queried": len(missing_entries),
            "response_state_counts": dict(sorted(entry_counts.items())),
            "http_200_responses": sum(item["http_status"] == 200 for item in results),
            "entry_identity_verified": sum(item["entry_identity_verified"] for item in results),
            "primary_citation_object_present": sum(item["primary_citation_object_present"] for item in results),
            "primary_doi_present": sum(item["primary_doi_present"] for item in results),
            "primary_pubmed_id_present": sum(item["primary_pubmed_id_present"] for item in results),
            "usable_primary_study_key_present": sum(item["usable_primary_study_key_present"] for item in results),
            "responses_with_recorded_sha256": len(response_hashes),
            "response_bytes_total": response_bytes,
        },
        "record_coverage": {
            "records_with_all_related_primary_study_keys_before_lookup": before_complete,
            "records_with_all_related_primary_study_keys_after_lookup": after_complete,
            "records_newly_complete_from_api_metadata": after_complete - before_complete,
            "records_still_missing_at_least_one_related_primary_study_key": len(records) - after_complete,
        },
        "decision": {
            "training_candidate_admission": 0,
            "new_independent_test_admission": 0,
            "metadata_lookup_proves_construct_ancestry": False,
            "metadata_lookup_proves_family_independence": False,
            "data_or_label_modified": False,
            "test_predictions_or_scores_read": False,
        },
        "local_only_detail": {
            "ledger_path": str(detail_path.relative_to(ROOT)),
            "ledger_rows": len(results),
            "ledger_sha256": detail_hash,
            "raw_response_directory": str(response_dir.relative_to(ROOT)),
            "raw_response_count": len(response_hashes),
        },
        "input_fingerprints": {
            "clean_release_pairs_sha256": sha256(pair_path),
            "prior_cif_metadata_audit_sha256": sha256(prior_path),
            "coordinate_source_verification_sha256": sha256(verification_path),
            "experimental_acquisition_report_sha256": sha256(acquisition_path),
        },
        "limitations": [
            "Official PDB database citation metadata is a source cross-check, not manual primary-article review.",
            "A citation or DOI does not establish construct-parent lineage, experimental independence, or assessment blindness.",
            "Remote homology and family relationships remain unresolved unless supported by a separate audited relation source.",
            "Only the small locally unresolved set was queried; no bulk download or new dataset split was created.",
        ],
        "test_predictions_read": False,
        "training_started": False,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = audit()
    print(json.dumps({
        "report_path": str((ROOT / "data/manifests/data_first_clean_release_remote_citations_v1.json").relative_to(ROOT)),
        "report_sha256": sha256(ROOT / "data/manifests/data_first_clean_release_remote_citations_v1.json"),
        "entries_queried": report["query_summary"]["entries_queried"],
        "http_200_responses": report["query_summary"]["http_200_responses"],
        "verified_entry_responses": report["query_summary"]["entry_identity_verified"],
        "newly_complete_records": report["record_coverage"]["records_newly_complete_from_api_metadata"],
        "records_still_missing_study_keys": report["record_coverage"]["records_still_missing_at_least_one_related_primary_study_key"],
        "training_admission": report["decision"]["training_candidate_admission"],
        "new_test_admission": report["decision"]["new_independent_test_admission"],
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
