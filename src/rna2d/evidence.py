"""Evidence gates and conservative screening of raw probing measurements."""
import math
import re


def pairing_evidence_issues(record):
    issues = []
    if record.get('evidence_class') not in {'experimental_coordinate_annotation', 'direct_experimental_pair_measurement'}:
        issues.append('not_direct_experimental_pair_labels')
    if record.get('provenance_verified') is not True:
        issues.append('unverified_provenance')
    if record.get('quality_verified') is not True:
        issues.append('unverified_quality')
    if record.get('contains_model_predictions') is not False:
        issues.append('prediction_status_not_explicitly_false')
    return issues


def screen_rdat(text, min_coverage=0.7):
    """Screen RDAT rows, retaining masks and rejecting NaNs/sentinels.

    This is a preliminary screening, not a full RDAT normalizer. Does not
    manufacture pairs, infer missing values, or use STRUCTURE as truth.
    Profiles with per-construct sequence overrides require separate mapping.
    """
    fields, annotations, profiles = {}, [], {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        key, values = parts[0], parts[1:]
        if key in {'SEQUENCE', 'SEQPOS', 'OFFSET', 'STRUCTURE', 'NAME'}:
            fields[key] = values
        elif key == 'ANNOTATION':
            annotations.extend(values)
        elif key.startswith('ANNOTATION_DATA:'):
            profiles.setdefault(key.split(':', 1)[1], {})['annotations'] = values
        elif key.startswith('REACTIVITY:'):
            profiles.setdefault(key.split(':', 1)[1], {})['raw'] = values
        elif key.startswith('REACTIVITY_ERROR:'):
            profiles.setdefault(key.split(':', 1)[1], {})['errors'] = values
    sequence = ''.join(fields.get('SEQUENCE', [])).upper().replace('T', 'U')
    seqpos = fields.get('SEQPOS', [])
    result = []
    for ident, profile in profiles.items():
        if 'raw' not in profile:
            continue
        reasons, values, measured = [], [], []
        for token in profile['raw']:
            try:
                x = float(token)
                valid = math.isfinite(x) and x > -100  # RDAT -999-like missing sentinels.
            except ValueError:
                x, valid = 0.0, False
            values.append(x if valid else None)
            measured.append(valid)
        coverage = sum(measured) / len(measured) if measured else 0.0
        if not sequence or set(sequence) - set('ACGU'):
            reasons.append('sequence_unresolved')
        if len(seqpos) != len(values):
            reasons.append('seqpos_reactivity_length_mismatch')
        if coverage < min_coverage:
            reasons.append('insufficient_measured_coverage')
        if 'errors' not in profile:
            reasons.append('uncertainty_missing_requires_review')
        elif len(profile['errors']) != len(values):
            reasons.append('error_length_mismatch')
        elif any(not re.fullmatch(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?', x) for x in profile['errors']):
            reasons.append('nonfinite_or_invalid_uncertainty_requires_review')
        elif any(float(x) < 0 for x in profile['errors']):
            reasons.append('negative_uncertainty')
        profile_annotations = profile.get('annotations', [])
        if any(a.lower().startswith(('sequence:', 'mutation:')) for a in profile_annotations):
            reasons.append('construct_sequence_or_mutation_mapping_required')
        result.append({'profile_id': ident, 'sequence': sequence, 'seqpos': seqpos,
                       'values': values, 'measured_mask': measured, 'coverage': coverage,
                       'annotations': annotations, 'profile_annotations': profile_annotations,
                       'reasons': reasons, 'preliminary_screen_pass': not reasons,
                       'evidence_class': 'experimental_probing_measurements',
                       'pair_labels_available': False, 'structure_field_used_as_truth': False,
                       'quality_verified': False,
                       'note': 'Requires assay-specific QC, mapping, normalization and replicate audit before training.'})
    return result
