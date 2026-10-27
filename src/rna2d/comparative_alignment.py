"""Strict Stockholm ingestion; parsing never grants biological label approval."""
from collections import defaultdict
from pathlib import Path
import re


def read_stockholm(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    lines = text.splitlines()
    if not lines or lines[0].strip() != '# STOCKHOLM 1.0':
        raise ValueError('Not Stockholm 1.0')
    gf, sequences, gc = defaultdict(list), defaultdict(list), defaultdict(list)
    ended = False
    for line in lines[1:]:
        if not line.strip():
            continue
        if ended:
            raise ValueError('Unexpected content after Stockholm terminator')
        if line.strip() == '//':
            ended = True
        elif line.startswith('#=GF '):
            _, key, value = line.split(maxsplit=2)
            gf[key].append(value)
        elif line.startswith('#=GC '):
            _, key, value = line.split(maxsplit=2)
            gc[key].append(value.strip())
        elif line.startswith('#'):
            continue
        else:
            parts = line.split()
            if len(parts) != 2:
                raise ValueError('Malformed aligned sequence line')
            sequences[parts[0]].append(parts[1])
    seqs = {key: ''.join(parts) for key, parts in sequences.items()}
    annotations = {key: ''.join(parts) for key, parts in gc.items()}
    if not ended or not seqs or len(gf.get('AC', [])) != 1:
        raise ValueError('Missing terminator, sequences or unique family accession')
    lengths = {len(s) for s in seqs.values()}
    if len(lengths) != 1:
        raise ValueError('Unequal alignment lengths')
    width = next(iter(lengths))
    if 'SS_cons' not in annotations or any(len(a) != width for a in annotations.values()):
        raise ValueError('Missing/misaligned consensus annotation')
    if 'SQ' in gf and int(gf['SQ'][0]) != len(seqs):
        raise ValueError('Declared sequence count differs')
    if any(re.fullmatch(r'[A-Za-z0-9_.-]+/\d+-\d+', key) is None for key in seqs):
        raise ValueError('Sequence identity lacks accession/range')
    if any(set(s.upper()) - set('ACGUTNRYKMSWBDHV.-~') for s in seqs.values()):
        raise ValueError('Unsupported nucleotide symbols')
    return dict(metadata=dict(gf), sequences=seqs, annotations=annotations, width=width)


def ungap(sequence):
    return ''.join(c for c in sequence.upper().replace('T', 'U') if c not in '.-~')
