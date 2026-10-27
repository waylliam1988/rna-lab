"""Run one frozen BCE or BCE+partner-competition seed; never loads test data."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
import time
import atexit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader
from rna2d.evidence import pairing_evidence_issues
from rna2d.losses import pair_prediction_loss
from rna2d.sampling import group_balanced_sample_indices, sample_schedule_sha256
from rna2d.sequence_encoders import SequencePairModel
from rna2d.training import Records, collate, evaluate, seed_all

ACTIVE_RUN = None
RUN_COMPLETED = False


def append_progress(row):
    progress = ROOT / "docs/DATA_FIRST_PROGRESS_V1.md"
    content = progress.read_text(encoding="utf-8")
    marker = "\n每次追加："
    if marker not in content:
        raise RuntimeError("Progress ledger append marker is missing")
    content = content.replace(marker, "\n" + row + marker, 1)
    progress.write_text(content, encoding="utf-8", newline="\n")


def log_unexpected_failure():
    if ACTIVE_RUN is not None and not RUN_COMPLETED:
        condition, seed, protocol_hash = ACTIVE_RUN
        exc_type, exc, _ = sys.exc_info()
        if exc_type is not None:
            append_progress(
                f"| {datetime.now().astimezone().date().isoformat()} | S6-L运行失败 | `{condition}` seed{seed}；protocol `{protocol_hash}`；异常 `{exc_type.__name__}: {exc}` | 失败运行目录保留，未触碰test；按冻结规则不自动重试 | 保留失败原因；该条件/种子不重跑 |")
            globals()["ACTIVE_RUN"] = None


atexit.register(log_unexpected_failure)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_sha256(state) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, ensure_ascii=True, indent=2) + "\n")


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
    parser.add_argument("--condition", choices=["bce", "bce_competition"], required=True)
    parser.add_argument("--seed", choices=[42, 43, 44], type=int, required=True)
    parser.add_argument("--device", choices=["cuda:0"], default="cuda:0")
    args = parser.parse_args()

    protocol_path = ROOT / "data/manifests/data_first_loss_protocol_v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = sha256(protocol_path)
    for relative_path, expected_hash in protocol["source_sha256"].items():
        if sha256(ROOT / relative_path) != expected_hash:
            raise ValueError(f"Frozen source fingerprint changed: {relative_path}")
    if protocol["status"] != "frozen_before_training":
        raise ValueError("Loss comparison protocol is not frozen")
    if args.seed not in protocol["seeds"]:
        raise ValueError("Seed is not in the frozen protocol")
    if args.condition not in protocol["conditions"]:
        raise ValueError("Condition is not in the frozen protocol")
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen protocol requires cuda:0")
    global ACTIVE_RUN

    train_path = ROOT / protocol["dataset"] / "train.jsonl"
    validation_path = ROOT / protocol["dataset"] / "validation.jsonl"
    if sha256(train_path) != protocol["data_sha256"]["train"]:
        raise ValueError("Training data fingerprint changed")
    if sha256(validation_path) != protocol["data_sha256"]["validation"]:
        raise ValueError("Validation data fingerprint changed")
    train_data = Records(train_path)
    validation_data = Records(validation_path)
    for split, dataset in (("train", train_data), ("validation", validation_data)):
        for row in dataset.rows:
            issues = pairing_evidence_issues(row)
            if issues:
                raise ValueError(f"Evidence gate rejected {split}:{row['id']}: {issues}")
    train_groups = {row["split_group_id"] for row in train_data.rows}
    validation_groups = {row["split_group_id"] for row in validation_data.rows}
    if train_groups & validation_groups:
        raise ValueError("Train and validation groups overlap")
    if len(train_data) != protocol["split_counts"]["train"] or len(train_groups) != protocol["split_groups"]["train"]:
        raise ValueError("Training records or groups differ from the frozen protocol")
    if len(validation_data) != protocol["split_counts"]["validation"] or len(validation_groups) != protocol["split_groups"]["validation"]:
        raise ValueError("Validation records or groups differ from the frozen protocol")

    output = ROOT / "runs/data_first_loss_v1" / args.condition / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    ACTIVE_RUN = (args.condition, args.seed, protocol_hash)
    append_progress(
        f"| {datetime.now().astimezone().date().isoformat()} | S6-L运行启动 | `{args.condition}` seed{args.seed}；protocol `{protocol_hash}`；train `{protocol['data_sha256']['train']}`；validation `{protocol['data_sha256']['validation']}`；550更新 | 仅加载train/validation；检查点本地保存，test未打开 | 等待运行完成或保留异常 |")
    device = torch.device(args.device)
    torch.set_num_threads(protocol["torch_num_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    seed_all(args.seed)
    model = SequencePairModel(
        protocol["model"]["kind"], protocol["model"]["width"],
        protocol["model"]["hidden"], protocol["model"]["depth"],
    ).to(device)
    initial_hash = state_sha256(model.state_dict())
    expected_initial_hash = protocol["initial_state_sha256"][str(args.seed)]
    if initial_hash != expected_initial_hash:
        raise ValueError("Initialized model state does not match frozen seed fingerprint")

    schedule = group_balanced_sample_indices(
        [row["split_group_id"] for row in train_data.rows],
        updates=protocol["optimizer_updates"], batch_size=protocol["batch_size"], seed=args.seed,
    )
    schedule_hash = sample_schedule_sha256(schedule, [row["id"] for row in train_data.rows])
    if schedule_hash != protocol["sample_order_sha256"][str(args.seed)]:
        raise ValueError("Sampling schedule hash does not match frozen protocol")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=protocol["optimizer"]["lr"],
        weight_decay=protocol["optimizer"]["weight_decay"],
    )
    weight = protocol["lambda"] if args.condition == "bce_competition" else 0.0
    config = {
        "condition": args.condition,
        "seed": args.seed,
        "protocol_sha256": protocol_hash,
        "train_sha256": protocol["data_sha256"]["train"],
        "validation_sha256": protocol["data_sha256"]["validation"],
        "model": protocol["model"],
        "initial_state_sha256": initial_hash,
        "loss": "balanced_pair_bce_plus_partner_competition" if weight else "balanced_pair_bce",
        "lambda": weight,
        "device": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_accessed": False,
        "selection": "fixed final update; no validation-based checkpoint selection",
    }
    write_json(output / "config.json", config)
    epoch_losses = []
    total_updates = 0
    start = time.perf_counter()
    step_log = (output / "steps.jsonl").open("w", encoding="utf-8", newline="\n")
    try:
        for epoch in range(1, protocol["epochs"] + 1):
            model.train()
            losses = []
            first_update = (epoch - 1) * protocol["updates_per_epoch"]
            last_update = epoch * protocol["updates_per_epoch"]
            for step, indices in enumerate(schedule[first_update:last_update], start=1):
                rows = [train_data.rows[index] for index in indices]
                tokens, labels, mask, rows = collate(rows)
                tokens, labels, mask = tokens.to(device), labels.to(device), mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(tokens)
                total_loss, details = pair_prediction_loss(
                    logits, labels, mask, competition_weight=weight, return_details=True,
                )
                if not torch.isfinite(total_loss):
                    raise FloatingPointError("Nonfinite pair loss")
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["optimizer"]["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                total_updates += 1
                losses.append(float(total_loss.detach()))
                step_log.write(json.dumps({
                    "epoch": epoch, "step": step, "optimizer_update": total_updates,
                    "loss_total": float(total_loss.detach()),
                    "loss_bce": float(details["loss_bce"].detach()),
                    "loss_competition": float(details["loss_competition"].detach()),
                    "competition_weight": weight,
                    "applicable_rnas": details["applicable_rnas"],
                    "applicable_endpoints": details["applicable_endpoints"],
                    "candidate_negatives": details["candidate_negatives"],
                    "positive_edges": details["positive_edges"],
                    "skipped_endpoints_without_negative": details["skipped_endpoints_without_negative"],
                }, ensure_ascii=True) + "\n")
            epoch_losses.append(statistics.mean(losses))
            print(json.dumps({"condition": args.condition, "seed": args.seed, "epoch": epoch,
                              "updates": total_updates, "mean_train_loss": epoch_losses[-1]}), flush=True)
    finally:
        step_log.close()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if total_updates != protocol["optimizer_updates"]:
        raise RuntimeError(f"Expected {protocol['optimizer_updates']} updates, got {total_updates}")

    checkpoint_path = output / "checkpoint.pt"
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "protocol_sha256": protocol_hash,
        "final_update": total_updates,
    }, checkpoint_path)
    validation_loader = DataLoader(validation_data, batch_size=protocol["evaluation_batch_size"],
                                   shuffle=False, collate_fn=collate, num_workers=0)
    metrics, details = evaluate(model, validation_loader, device, protocol["threshold"])
    group_macro, per_group = group_macro_f1(details, validation_data.rows)
    conflict_group_macro, conflict_per_group = group_macro_conflict_rate(details, validation_data.rows)
    result = {
        "condition": args.condition,
        "seed": args.seed,
        "updates": total_updates,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_state_sha256": initial_hash,
        "sample_order_sha256": schedule_hash,
        "epoch_train_loss_mean": epoch_losses,
        "training_seconds": elapsed,
        "peak_gpu_allocated_MiB": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "checkpoint_sha256": sha256(checkpoint_path),
        "validation": metrics,
        "validation_group_macro_f1": group_macro,
        "validation_per_group_f1": per_group,
        "validation_group_macro_multi_partner_conflict_endpoint_rate": conflict_group_macro,
        "validation_per_group_multi_partner_conflict_endpoint_rate": conflict_per_group,
        "details": details,
        "reload_verified_in_runner": False,
        "test_accessed": False,
    }
    write_json(output / "result.json", result)
    append_progress(
        f"| {datetime.now().astimezone().date().isoformat()} | S6-L运行完成 | `{args.condition}` seed{args.seed}；550更新；schedule `{schedule_hash}`；checkpoint `{result['checkpoint_sha256']}` | validation group-macro F1={group_macro:.6f}；test未访问；独立重载待统一执行 | 继续冻结名单中的其他条件/种子，不改协议 |")
    global RUN_COMPLETED
    RUN_COMPLETED = True
    print(json.dumps({"condition": args.condition, "seed": args.seed,
                      "validation_group_macro_f1": group_macro,
                      "checkpoint_sha256": result["checkpoint_sha256"], "test_accessed": False}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_unexpected_failure()
        raise
