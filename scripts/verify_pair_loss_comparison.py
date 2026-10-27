"""Independently reload all frozen loss-comparison checkpoints on validation."""
import hashlib
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from rna2d.sequence_encoders import SequencePairModel
from rna2d.training import Records, collate, evaluate


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def group_macro_f1(details, rows):
    group_by_id = {row["id"]: row["split_group_id"] for row in rows}
    values = {}
    for detail in details:
        values.setdefault(group_by_id[detail["id"]], []).append(detail["f1"])
    per_group = {group: statistics.mean(scores) for group, scores in values.items()}
    return statistics.mean(per_group.values()), per_group


def group_macro_conflict_rate(details, rows):
    group_by_id = {row["id"]: row["split_group_id"] for row in rows}
    values = {}
    for detail in details:
        values.setdefault(group_by_id[detail["id"]], []).append(
            detail["multi_partner_conflict_endpoint_rate"]
        )
    per_group = {group: statistics.mean(rates) for group, rates in values.items()}
    return statistics.mean(per_group.values()), per_group


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="data/manifests/data_first_loss_protocol_v2.json")
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol
    version = protocol_path.stem.removeprefix("data_first_loss_protocol_")
    if not version.startswith("v"):
        raise ValueError("Protocol filename must include an explicit version")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = sha256(protocol_path)
    for relative_path, expected_hash in protocol["source_sha256"].items():
        if sha256(ROOT / relative_path) != expected_hash:
            raise ValueError(f"Frozen source fingerprint changed: {relative_path}")
    validation_path = ROOT / protocol["dataset"] / "validation.jsonl"
    if sha256(validation_path) != protocol["data_sha256"]["validation"]:
        raise ValueError("Validation data fingerprint changed")
    dataset = Records(validation_path)
    loader = DataLoader(dataset, batch_size=protocol["evaluation_batch_size"], shuffle=False,
                        collate_fn=collate, num_workers=0)
    device = torch.device("cuda:0")
    torch.set_num_threads(protocol["torch_num_threads"])
    checks = []
    for condition in protocol["conditions"]:
        for seed in protocol["seeds"]:
            directory = ROOT / "runs" / f"data_first_loss_{version}" / condition / f"seed{seed}"
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            checkpoint_path = directory / "checkpoint.pt"
            if sha256(checkpoint_path) != result["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {condition}/{seed}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            if checkpoint["protocol_sha256"] != protocol_hash:
                raise ValueError(f"Protocol mismatch: {condition}/{seed}")
            model = SequencePairModel(
                protocol["model"]["kind"], protocol["model"]["width"],
                protocol["model"]["hidden"], protocol["model"]["depth"],
            ).to(device)
            model.load_state_dict(checkpoint["model"], strict=True)
            metrics, details = evaluate(model, loader, device, protocol["threshold"])
            group_macro, per_group = group_macro_f1(details, dataset.rows)
            conflict_group_macro, conflict_per_group = group_macro_conflict_rate(details, dataset.rows)
            if abs(group_macro - result["validation_group_macro_f1"]) > protocol["reload_tolerance"]:
                raise ValueError(f"Group macro changed after reload: {condition}/{seed}")
            if abs(metrics["macro_multi_partner_conflict_endpoint_rate"] -
                   result["validation"]["macro_multi_partner_conflict_endpoint_rate"]) > protocol["reload_tolerance"]:
                raise ValueError(f"Multi-partner conflict rate changed after reload: {condition}/{seed}")
            if any(abs(a["multi_partner_conflict_endpoint_rate"] - b["multi_partner_conflict_endpoint_rate"])
                   > protocol["reload_tolerance"] for a, b in zip(details, result["details"])):
                raise ValueError(f"Per-RNA conflict rates changed after reload: {condition}/{seed}")
            if abs(conflict_group_macro -
                   result["validation_group_macro_multi_partner_conflict_endpoint_rate"]) > protocol["reload_tolerance"]:
                raise ValueError(f"Group macro conflict rate changed after reload: {condition}/{seed}")
            expected_per_group = result["validation_per_group_multi_partner_conflict_endpoint_rate"]
            if conflict_per_group.keys() != expected_per_group.keys() or any(
                abs(conflict_per_group[group] - expected_per_group[group]) > protocol["reload_tolerance"]
                for group in conflict_per_group
            ):
                raise ValueError(f"Per-group conflict rates changed after reload: {condition}/{seed}")
            if any(a["predicted_pairs"] != b["predicted_pairs"] for a, b in zip(details, result["details"])):
                raise ValueError(f"Decoded predictions changed after reload: {condition}/{seed}")
            checks.append({
                "condition": condition, "seed": seed,
                "checkpoint_sha256": result["checkpoint_sha256"],
                "validation_group_macro_f1": group_macro,
                "validation_macro_multi_partner_conflict_endpoint_rate":
                    metrics["macro_multi_partner_conflict_endpoint_rate"],
                "validation_group_macro_multi_partner_conflict_endpoint_rate": conflict_group_macro,
                "all_predictions_reproduced": True,
                "test_accessed": False,
            })
            del model
            torch.cuda.empty_cache()
    output = {
        "verification": f"pair-loss-comparison-{version}",
        "protocol_sha256": protocol_hash,
        "all_checks_passed": True,
        "reloaded_checkpoints": len(checks),
        "checks": checks,
        "test_accessed": False,
    }
    path = ROOT / f"data/manifests/data_first_loss_verification_{version}.json"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(output, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
