"""Audit quarantined direct-experiment records without changing admission."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    path = ROOT / "data/experimental/joint_split_v2/quarantine.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    homology_missing = [r for r in rows if not r.get("homology_components")]
    study_incomplete = [r for r in rows if not r.get("study_metadata_complete")]
    study_missing = [r for r in rows if not r.get("study_ids")]
    report = {
        "audit": "joint-quarantine-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(path.relative_to(ROOT)),
        "source_sha256": sha256(path),
        "records": len(rows),
        "unique_sequence_sha256": len({r["sequence_sha256"] for r in rows}),
        "unique_pdb_ids": len({r["pdb_id"] for r in rows}),
        "quarantine_reasons": dict(Counter(r.get("quarantine_reason", "missing") for r in rows)),
        "missing_or_unresolved": {
            "homology_components": len(homology_missing),
            "study_metadata_complete_false": len(study_incomplete),
            "study_ids": len(study_missing),
            "both_homology_and_study_unresolved": sum(not r.get("homology_components") and not r.get("study_ids") for r in rows),
        },
        "context": dict(Counter(r.get("context", "unknown") for r in rows)),
        "evidence_checks": {
            "all_experimental_coordinate_annotation": all(r.get("evidence_class") == "experimental_coordinate_annotation" for r in rows),
            "all_provenance_verified": all(r.get("provenance_verified") is True for r in rows),
            "all_quality_verified": all(r.get("quality_verified") is True for r in rows),
            "model_predictions_present": any(r.get("contains_model_predictions") for r in rows),
        },
        "decision": "retain_quarantine",
        "admission_state": "not_admitted",
        "reason": "The local evidence resolves experimental coordinate provenance and labels but does not resolve homology, primary-study, construct, or family independence for all records.",
        "next_action": "Obtain and review traceable family/construct/primary-study evidence in a new version; do not alter this frozen split or promote records by metadata completeness alone.",
        "limitations": [
            "This is a missing-evidence audit, not proof that the records are biologically invalid.",
            "No sequence or label values were modified.",
            "No record was moved into train, validation, or test.",
        ],
    }
    out = ROOT / "data/manifests/joint_quarantine_audit_v1.json"
    out.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
