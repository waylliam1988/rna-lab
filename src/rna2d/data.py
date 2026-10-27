import hashlib

CANONICAL = {'AU', 'UA', 'GC', 'CG', 'GU', 'UG'}


def parse_bpseq(text):
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        if len(parts) != 3:
            raise ValueError('Malformed numeric BPSEQ row')
        rows.append((int(parts[0]), parts[1].upper(), int(parts[2])))
    if not rows:
        raise ValueError('Empty BPSEQ')
    n = len(rows)
    for i, (index, base, partner) in enumerate(rows, 1):
        if index != i or len(base) != 1 or not 0 <= partner <= n or partner == i:
            raise ValueError('Invalid index, base or partner')
        if partner and rows[partner - 1][2] != i:
            raise ValueError('Asymmetric pair')
    original = ''.join(row[1] for row in rows)
    seq = original.replace('T', 'U')
    return seq, [[i - 1, p - 1] for i, _, p in rows if p > i], original.count('T')


def sequence_hash(seq):
    return hashlib.sha256(seq.encode()).hexdigest()


def crossing_pairs(pairs):
    edges = {tuple(sorted(p)) for p in pairs}
    return {e for e in edges if any(e[0] < f[0] < e[1] < f[1] or
                                   f[0] < e[0] < f[1] < e[1] for f in edges)}


def pair_metrics(predicted, truth):
    p = {tuple(sorted(x)) for x in predicted}
    t = {tuple(sorted(x)) for x in truth}
    tp = len(p & t)
    precision = tp / len(p) if p else float(not t)
    recall = tp / len(t) if t else float(not p)
    return {'tp': tp, 'fp': len(p - t), 'fn': len(t - p),
            'precision': precision, 'recall': recall,
            'f1': 2 * tp / (len(p) + len(t)) if p or t else 1.0}


def observable_pairs(pairs, mask):
    """Ignore unknown endpoints after prediction; never label them negative."""
    return [list(pair) for pair in pairs if mask[pair[0]] and mask[pair[1]]]


def greedy_decode(scores, sequence, threshold=0.5, min_separation=4):
    candidates = [(float(scores[i][j]), i, j) for i in range(len(sequence))
                  for j in range(i + min_separation, len(sequence))
                  if sequence[i] + sequence[j] in CANONICAL and float(scores[i][j]) >= threshold]
    used, result = set(), []
    for score, i, j in sorted(candidates, key=lambda x: (-x[0], x[1], x[2])):
        if i not in used and j not in used:
            result.append([i, j])
            used.update((i, j))
    return sorted(result)
