"""Manifest checks shared by approved experimental training/evaluation entrypoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_split_manifest(
    data_dir: Path,
    manifest_path: Path,
    splits: Iterable[str],
) -> str:
    """Require a passing split manifest whose recorded hashes match these files."""
    data_dir = Path(data_dir)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("passed") is not True:
        raise ValueError(f"Split manifest did not pass validation: {manifest_path}")
    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise ValueError(f"Split manifest has no file fingerprints: {manifest_path}")
    for split in splits:
        path = data_dir / f"{split}.jsonl"
        expected_sha256 = expected.get(split)
        if not path.is_file() or not isinstance(expected_sha256, str):
            raise ValueError(f"Split manifest does not cover {split}: {path}")
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"{split} split fingerprint does not match manifest: {path}")
    return file_sha256(manifest_path)
