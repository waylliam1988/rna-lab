"""Download author-distributed benchmarks; no third-party dependencies."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://github.com/mxfold/mxfold2/releases/download/v0.1.0/'
SOURCES = ['archiveII.tar.gz', 'bpRNA.tar.gz', 'bpRNAnew.tar.gz']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=SOURCES, nargs='+', default=SOURCES)
    args = parser.parse_args()
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in args.only:
        dest = raw / name
        if not dest.exists():
            tmp = dest.with_suffix('.partial')
            digest = hashlib.sha256()
            size = 0
            with urlopen(BASE + name, timeout=120) as response, tmp.open('wb') as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
                    digest.update(block)
                    size += len(block)
            os.replace(tmp, dest)
            downloaded_at = datetime.now(timezone.utc).isoformat()
        else:
            digest = hashlib.sha256(dest.read_bytes())
            size = dest.stat().st_size
            old = manifest.get(name, {})
            if old.get('sha256') and old['sha256'] != digest.hexdigest():
                raise ValueError(f'Existing archive checksum differs from manifest: {name}')
            downloaded_at = old.get('downloaded_at_utc')
        manifest[name] = {'url': BASE + name, 'bytes': size,
                          'sha256': digest.hexdigest(), 'downloaded_at_utc': downloaded_at,
                          'verified_at_utc': datetime.now(timezone.utc).isoformat(),
                          'note': 'Local checksum; not an upstream signed checksum. Existing untracked file download time may be unknown.'}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(name, size, digest.hexdigest())


if __name__ == '__main__':
    main()
