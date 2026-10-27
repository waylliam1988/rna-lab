"""Offline, bounded source review. No representative selection or training."""
import json
import sys
from pathlib import Path
from acquire_experimental_corpus import ROOT, digest_file, write_json
sys.path.insert(0, str(ROOT / 'src'))
from rna2d.comparative_alignment import read_stockholm


def main():
    base = ROOT / 'data/raw/rfam_family_pilot_v1'
    output = ROOT / 'data/manifests/rfam_family_pilot_sources_v1.json'
    if output.exists():
        raise FileExistsError('Frozen source review exists')
    protocol = ROOT / 'data/manifests/rfam_family_pilot_selection_v1.json'
    p = json.loads(protocol.read_text())
    receipts = json.loads((base / 'receipts.json').read_text())
    for receipt in receipts:
        if receipt['status'] == 'downloaded':
            assert digest_file(ROOT / receipt['local_path']) == receipt['sha256']
    reviews = {
        'RF00001': dict(provenance_supported=True, scope='Published structure plus database alignment source; individual current consensus edges not independently certified.', evidence_urls=['https://pubmed.ncbi.nlm.nih.gov/11283358/', 'https://pubmed.ncbi.nlm.nih.gov/11752286/', 'https://academic.oup.com/nar/article/30/1/176/1332647'], evidence_location='Rfam SE/SS/RM, database paper Figure 1 aligned secondary structure and Figure 2 crystallographic interactions; cited structure paper abstract.', remaining='Current curator changes and edge-level scope still require mapping review.'),
        'RF00005': dict(provenance_supported=False, scope='Rfam SS cites a 1993 tRNA review; SE is curator attribution, not an independently reproduced current alignment lineage.', evidence_urls=['https://pubmed.ncbi.nlm.nih.gov/8256282/'], evidence_location='Rfam SE/SS; cited PubMed publication type Review.', remaining='Review is real literature but does not by itself audit the current SEED consensus; retain source uncertainty.'),
        'RF00010': dict(provenance_supported=True, scope='Published comparative database source supports alignment/secondary-structure provenance, not per-molecule experimental truth.', evidence_urls=['https://academic.oup.com/nar/article/27/1/314/1239359'], evidence_location='Database paper Introduction (Book of P comparative information), Description (sequence alignments and secondary structures).', remaining='Current sequence-to-consensus mapping and reliable negative-label scope remain unverified.')
    }
    rows = []
    for acc in p['families']:
        d = read_stockholm(base / (acc + '.sto'))
        metadata = json.loads((base / (acc + '.html')).read_text())['rfam']
        assert d['metadata']['AC'] == [acc] and metadata['acc'] == acc
        assert len(d['sequences']) == metadata['curation']['num_seed']
        assert d['metadata']['SS'][0] == metadata['curation']['structure_source']
        rows.append(dict(family=acc, name=metadata['id'], clan=metadata.get('clan'), release=metadata['release'], seed_count=len(d['sequences']), alignment_columns=d['width'], seed_source=d['metadata']['SE'], structure_source=d['metadata']['SS'], seed_sha256=digest_file(base / (acc + '.sto')), metadata_sha256=digest_file(base / (acc + '.html')), review=reviews[acc], model_prediction_status='not_marked_predicted_at_source; current_edge_level_exclusion_not_certified', auxiliary_training_approved=False))
    report = dict(stage='1_source_review', selection_sha256=digest_file(protocol), receipts_sha256=digest_file(base / 'receipts.json'), families=rows, provenance_supported_families=sum(r['review']['provenance_supported'] for r in rows), admitted_families=0, failed_fulltext_downloads=[dict(url=r['url'], error=r['error']) for r in receipts if r['status']=='failed'], literature_review_method='Official publisher indexed body / PubMed metadata and figures, plus cached official Rfam metadata; failed Europe PMC fullTextXML attempts retained, not claimed as read.', limits='Published is not automatic label approval. No representative sequences chosen, no labels generated, no model predictions or training.', source_script_sha256=digest_file(Path(__file__)))
    write_json(output, report)
    print(json.dumps(dict(families=len(rows), provenance_supported=report['provenance_supported_families'], admitted=0)))


if __name__ == '__main__':
    main()
