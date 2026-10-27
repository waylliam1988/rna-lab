"""Extract only independently useful primitives from the preserved snapshot.

No import of the old Config, BioSystem, or 3D pipeline is performed.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract(filename, class_name):
    source = (ROOT / 'legacy_reference' / filename).read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return '\n'.join(source.splitlines()[node.lineno - 1:node.end_lineno]) + '\n'


def main():
    dest = ROOT / 'src/rna2d/core'
    dest.mkdir(parents=True, exist_ok=True)
    header = '# Extracted verbatim by scripts/migrate_core.py; see docs/MIGRATION.md.\n'
    (dest / 'memory.py').write_text(header + 'import math\nfrom typing import Optional, Tuple\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\n\n' + extract('kernels.py', 'RWKVCellLite'), encoding='utf-8')
    (dest / 'control.py').write_text(header + 'import math\nimport torch\nfrom torch import nn\n\n' + extract('ghost.py', 'ContinuousControlLayer'), encoding='utf-8')
    print('Extracted RWKVCellLite and ContinuousControlLayer without old Config dependency')


if __name__ == '__main__':
    main()
