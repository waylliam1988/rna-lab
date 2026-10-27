"""Cross-reference locally recovered PDB article titles against Crossref.

This limited second-pass review does not treat a title search as verified
provenance. Per-entry metadata and responses remain local; only counts and
hashes are publishable, and no sample is admitted by this script.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def primary_article(payload: dict) -> dict | None:
    citations = payload.get("citation")
    if not isinstance(citations, list):
        return None
    primary = [
        item for item in citations
        if isinstance(item, dict)
        and (str(item.get("id", "")).casefold() == "primary" or item.get("rcsb_is_primary") is True)
    ]
    if not primary:
        return None
    return primary[0]


def first_author_family(article: dict | None) -> str | None:
    authors = article.get("rcsb_authors") if isinstance(article, dict) else None
    if not isinstance(authors, list) or not authors or not isinstance(authors[0], str):
        return None
    first = authors[0].strip()
    family = first.split(",", 1)[0] if "," in first else first.split()[-1] if first.split() else ""
    family = normalize_title(family)
    return family or None


def crossref_matches(article: dict | None, response: dict) -> list[dict]:
    if not isinstance(article, dict):
        return []
    target_title = article.get("title")
    if not isinstance(target_title, str) or not target_title.strip():
        return []
    target_norm = normalize_title(target_title)
    source_year = article.get("year")
    try:
        source_year = int(str(source_year)[:4])
    except (TypeError, ValueError):
        source_year = None
    source_author = first_author_family(article)

    message = response.get("message") if isinstance(response, dict) else None
    items = message.get("items") if isinstance(message, dict) else None
    if not isinstance(items, list):
        return []

    matches = []
    for item in items:
        if not isinstance(item, dict):
            continue
        titles = item.get("title")
        candidate_title = titles[0] if isinstance(titles, list) and titles and isinstance(titles[0], str) else None
        if not candidate_title or normalize_title(candidate_title) != target_norm:
            continue

        candidate_year = None
        date = item.get("published-print") or item.get("published-online") or item.get("published")
        date_parts = date.get("date-parts") if isinstance(date, dict) else None
        if isinstance(date_parts, list) and date_parts and isinstance(date_parts[0], list) and date_parts[0]:
            try:
                candidate_year = int(date_parts[0][0])
            except (TypeError, ValueError):
                candidate_year = None

        candidate_author = None
        authors = item.get("author")
        if isinstance(authors, list) and authors and isinstance(authors[0], dict):
            family = authors[0].get("family")
            if isinstance(family, str):
                normalized = normalize_title(family)
                candidate_author = normalized or None

        year_match = source_year is not None and candidate_year == source_year
        author_match = source_author is not None and candidate_author is not None and candidate_author == source_author
        strict_match = (
            source_year is not None
            and candidate_year is not None
            and source_author is not None
            and candidate_author is not None
            and year_match
            and author_match
        )
        doi = item.get("DOI")
        matches.append({
            "doi": doi.strip().lower() if isinstance(doi, str) and doi.strip() else None,
            "exact_title_match": True,
            "source_year_available": source_year is not None,
            "crossref_year_available": candidate_year is not None,
            "year_match": year_match,
            "source_first_author_available": source_author is not None,
            "crossref_first_author_available": candidate_author is not None,
            "first_author_match": author_match,
            "strict_title_year_author_match": strict_match,
        })
    return matches


def query_crossref(title: str) -> tuple[int | None, bytes | None, str]:
    query = urlencode({
        "query.title": title,
        "rows": 5,
        "select": "DOI,title,author,published,published-print,published-online",
    })
    url = "https://api.crossref.org/works?" + query
    request = Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "RNA-folding-provenance-audit/1.0",
    })
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                return response.status, None, "response_exceeds_size_limit"
            return response.status, raw, "received"
    except HTTPError as exc:
        try:
            raw = exc.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raw = None
        except Exception:
            raw = None
        return exc.code, raw, "http_error"
    except (URLError, TimeoutError, OSError):
        return None, None, "network_error"


def audit() -> dict:
    manifests = ROOT / "data/manifests"
    remote_report_path = manifests / "data_first_clean_release_remote_citations_v1.json"
    detail_path = manifests / "data_first_clean_release_crossref_review_v1.jsonl"
    response_dir = ROOT / "data/raw/primary_citation_crossref_review_v1"
    output_path = manifests / "data_first_clean_release_crossref_review_v1.json"
    if detail_path.exists() or response_dir.exists() or output_path.exists():
        raise FileExistsError("Refusing to overwrite existing Crossref title-review artifacts")

    remote_report = json.loads(remote_report_path.read_text(encoding="utf-8"))
    if remote_report["query_summary"]["entry_identity_verified"] != remote_report["query_summary"]["entries_queried"]:
        raise ValueError("Crossref follow-up requires identity-verified RCSB metadata for every queried entry")
    raw_dir = ROOT / "data/raw/primary_citation_review_v1"
    response_dir.mkdir(parents=True, exist_ok=False)

    source_files = sorted(path for path in raw_dir.glob("*.json") if path.is_file() and path.name not in {p.name for p in raw_dir.glob("*.receipt.json")})
    results = []
    total_response_bytes = 0
    for index, source_path in enumerate(source_files):
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        entry_id = str(payload.get("rcsb_id", "")).upper()
        article = primary_article(payload)
        title = article.get("title") if isinstance(article, dict) else None
        if not entry_id or not isinstance(title, str) or not title.strip():
            results.append({
                "entry_id": entry_id or None,
                "state": "no_primary_article_title",
                "crossref_http_status": None,
                "crossref_response_sha256": None,
                "exact_title_match_count": 0,
                "strict_title_year_author_match_count": 0,
                "unique_candidate_doi_count": 0,
            })
            continue

        status, raw, state = query_crossref(title)
        item = {
            "entry_id": entry_id,
            "state": state,
            "crossref_http_status": status,
            "crossref_response_sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
            "crossref_response_bytes": len(raw) if raw is not None else None,
            "exact_title_match_count": 0,
            "strict_title_year_author_match_count": 0,
            "unique_candidate_doi_count": 0,
            "unique_strict_candidate_doi_count": 0,
        }
        matches = []
        if raw is not None:
            total_response_bytes += len(raw)
            response_path = response_dir / f"{entry_id.lower()}.json"
            response_path.write_bytes(raw)
            receipt_path = response_path.with_suffix(".receipt.json")
            receipt_path.write_text(json.dumps({
                "entry_id": entry_id,
                "source_url": "https://api.crossref.org/works",
                "query_title_sha256": hashlib.sha256(title.encode("utf-8")).hexdigest(),
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "http_status": status,
                "sha256": item["crossref_response_sha256"],
                "bytes": len(raw),
            }, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
            if status == 200:
                try:
                    crossref_payload = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    item["state"] = "invalid_json"
                else:
                    matches = crossref_matches(article, crossref_payload)
                    item["state"] = "crossref_response_received"
        unique_dois = {match["doi"] for match in matches if match["doi"]}
        strict_dois = {match["doi"] for match in matches if match["doi"] and match["strict_title_year_author_match"]}
        item["exact_title_match_count"] = len(matches)
        item["strict_title_year_author_match_count"] = sum(match["strict_title_year_author_match"] for match in matches)
        item["unique_candidate_doi_count"] = len(unique_dois)
        item["unique_strict_candidate_doi_count"] = len(strict_dois)
        results.append(item)
        if index + 1 < len(source_files):
            time.sleep(1.0)

    detail_path.write_text(
        "".join(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n" for item in results),
        encoding="utf-8",
        newline="\n",
    )
    state_counts = Counter(item["state"] for item in results)
    report = {
        "audit": "clean-release-crossref-primary-citation-title-review-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Limited title cross-reference for RCSB primary article titles where local PDB records lack DOI/PubMed keys; candidate matches remain unverified pending manual review.",
        "query_summary": {
            "source": "Crossref REST API works search",
            "entry_count": len(source_files),
            "response_state_counts": dict(sorted(state_counts.items())),
            "http_200_responses": sum(item["crossref_http_status"] == 200 for item in results),
            "responses_with_recorded_sha256": sum(bool(item.get("crossref_response_sha256")) for item in results),
            "primary_titles_present": sum(item["state"] != "no_primary_article_title" for item in results),
            "entries_with_exact_title_matches": sum(item["exact_title_match_count"] > 0 for item in results),
            "exact_title_matches_total": sum(item["exact_title_match_count"] for item in results),
            "entries_with_strict_title_year_author_match": sum(item["strict_title_year_author_match_count"] > 0 for item in results),
            "strict_title_year_author_matches_total": sum(item["strict_title_year_author_match_count"] for item in results),
            "entries_with_unique_strict_candidate_doi": sum(item["unique_strict_candidate_doi_count"] == 1 for item in results),
            "response_bytes_total": total_response_bytes,
        },
        "decision": {
            "crossref_matches_are_automatically_verified": False,
            "primary_study_keys_added_to_frozen_relation_graph": 0,
            "training_candidate_admission": 0,
            "new_independent_test_admission": 0,
            "construct_ancestry_or_homology_resolved": False,
        },
        "local_only_detail": {
            "ledger_path": str(detail_path.relative_to(ROOT)),
            "ledger_rows": len(results),
            "ledger_sha256": sha256(detail_path),
            "raw_response_directory": str(response_dir.relative_to(ROOT)),
        },
        "input_fingerprints": {
            "rcsb_remote_citation_review_sha256": sha256(remote_report_path),
            "prior_cif_metadata_audit_sha256": sha256(manifests / "data_first_clean_release_cif_metadata_audit_v1.json"),
        },
        "limitations": [
            "Title similarity is not a primary-source citation verification; exact-title metadata matches must be reviewed against the article record and project source mapping.",
            "No DOI from a title search is added to the frozen relation graph without that review.",
            "This audit does not establish construct ancestry, assay independence, evaluation blindness, or remote-homology independence.",
            "Only the existing unresolved citation set was queried; no bulk download, training, test evaluation, label change, or split change occurred.",
        ],
        "test_predictions_read": False,
        "training_started": False,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = audit()
    print(json.dumps({
        "report_path": "data/manifests/data_first_clean_release_crossref_review_v1.json",
        "report_sha256": sha256(ROOT / "data/manifests/data_first_clean_release_crossref_review_v1.json"),
        "entries_checked": report["query_summary"]["entry_count"],
        "exact_title_match_entries": report["query_summary"]["entries_with_exact_title_matches"],
        "strict_match_entries": report["query_summary"]["entries_with_strict_title_year_author_match"],
        "doi_added_to_relation_graph": report["decision"]["primary_study_keys_added_to_frozen_relation_graph"],
        "new_test_admission": report["decision"]["new_independent_test_admission"],
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
