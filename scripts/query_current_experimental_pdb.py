"""Freeze the current RNA-containing *experimental* PDB entry universe."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    query = {'query': {'type': 'terminal', 'service': 'text', 'parameters':
                      {'attribute': 'rcsb_entry_info.polymer_entity_count_RNA', 'operator': 'greater', 'value': 0}},
             'return_type': 'entry', 'request_options': {'return_all_hits': True, 'results_content_type': ['experimental']}}
    response = requests.get('https://search.rcsb.org/rcsbsearch/v2/query', params={'json': json.dumps(query)}, timeout=(15, 120))
    response.raise_for_status()
    result = response.json()
    raw = ROOT / 'data/raw/source_catalog/current_experimental_rna_pdb.json'
    raw.write_bytes(response.content)
    ids = sorted(r['identifier'] for r in result['result_set'])
    report = {'queried_at_utc': datetime.now(timezone.utc).isoformat(),
              'endpoint': 'https://search.rcsb.org/rcsbsearch/v2/query', 'query': query,
              'entry_count': len(ids), 'server_total_count': result['total_count'],
              'response_sha256': hashlib.sha256(response.content).hexdigest(), 'entry_ids': ids,
              'note': 'PDB entries, not chains or unique sequences; RNA-protein complexes included. No quality filter or pairing extraction yet.'}
    (ROOT / 'data/reports/current_experimental_rna_pdb.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({k: v for k, v in report.items() if k != 'entry_ids'})


if __name__ == '__main__':
    main()
