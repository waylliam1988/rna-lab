"""Deterministic group-balanced sampling helpers for paired experiments."""
from __future__ import annotations

import hashlib
import random


def group_balanced_sample_indices(group_ids, *, updates: int, batch_size: int, seed: int):
    """Sample groups uniformly, then records uniformly within each group.

    Sampling is with replacement. Returned batches are fixed independently of
    model initialization, so paired loss conditions consume identical samples.
    """
    if updates < 1 or batch_size < 1:
        raise ValueError("updates and batch_size must be positive")
    groups = {}
    for index, group_id in enumerate(group_ids):
        if group_id is None:
            raise ValueError("Every sampled record must have a split group")
        groups.setdefault(group_id, []).append(index)
    if not groups:
        raise ValueError("Cannot sample an empty grouped dataset")
    group_keys = sorted(groups)
    rng = random.Random(seed)
    schedule = []
    for _ in range(updates):
        batch = []
        for _ in range(batch_size):
            group = rng.choice(group_keys)
            batch.append(rng.choice(groups[group]))
        schedule.append(batch)
    return schedule


def sample_schedule_sha256(schedule, record_ids) -> str:
    h = hashlib.sha256()
    for batch in schedule:
        h.update("\n".join(record_ids[index] for index in batch).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()
