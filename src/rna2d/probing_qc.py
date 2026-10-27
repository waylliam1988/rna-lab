"""Strict, assay-aware screening of measured RDAT profiles (never pair truth)."""
import re
import numpy as np


def sequence_mapping(header_sequence, annotations, offset, seqpos):
    overrides=[a.split(':',1)[1] for a in annotations if a.lower().startswith('sequence:')]
    mutations=[a.split(':',1)[1] for a in annotations if a.lower().startswith('mutation:')]
    if len(set(overrides))>1: raise ValueError('conflicting_sequence_overrides')
    sequence=(overrides[0] if overrides else header_sequence).upper().replace('T','U')
    if not sequence or set(sequence)-set('ACGU'): raise ValueError('sequence_ambiguous')
    if mutations and not overrides:
        chars=list(sequence)
        for text in mutations:
            if text.lower() in ('wt','wildtype','wild-type'): continue
            # Only unambiguous single-substitution syntax is resolved here.
            match=re.fullmatch(r'([ACGUT])(-?\d+)([ACGUT])',text.upper())
            if not match: raise ValueError('mutation_mapping_unresolved')
            old,number,new=match.groups()
            i=int(number)-offset-1
            if not 0<=i<len(chars) or chars[i]!=old.replace('T','U'):
                raise ValueError('mutation_reference_mismatch')
            chars[i]=new.replace('T','U')
        sequence=''.join(chars)
    indices=[]
    for token in seqpos:
        match=re.fullmatch(r'([ACGUTNXacgutnx]?)(-?\d+)',token)
        if not match: raise ValueError('seqpos_syntax')
        base,number=match.groups()
        i=int(number)-offset-1
        if not 0<=i<len(sequence): raise ValueError('seqpos_out_of_bounds')
        # X/N is a source placeholder for per-construct variants, not a base.
        if base.upper() in 'ACGUT' and base and not overrides and not mutations:
            if sequence[i]!=base.upper().replace('T','U'): raise ValueError('seqpos_base_mismatch')
        indices.append(i)
    if len(set(indices))!=len(indices): raise ValueError('duplicate_seqpos')
    return sequence,indices


def quality_metadata(annotations, construct_annotations):
    all_annotations=construct_annotations+annotations
    reasons=[]
    if any(a.lower().startswith('warning:') for a in all_annotations): reasons.append('source_warning')
    snr=[]
    for annotation in all_annotations:
        if annotation.lower().startswith('signal_to_noise:'):
            try: snr.append(float(annotation.rsplit(':',1)[1]))
            except ValueError: pass
    if not snr or not all(np.isfinite(s) and s>=3 for s in snr): reasons.append('snr_missing_or_below3')
    reads=[]
    for annotation in annotations:
        if annotation.lower().startswith('reads:'):
            try: reads.append(float(annotation.split(':',1)[1]))
            except ValueError: reasons.append('invalid_read_count')
    if reads and any(not np.isfinite(x) or x<1000 for x in reads): reasons.append('reads_below1000')
    modifiers=[a.split(':',1)[1] for a in all_annotations if a.lower().startswith('modifier:')]
    # These assays have explicit measured signal, not inferred base-pair labels.
    supported={'dms','1m7','2a3','shape','nmi','nmia','nai','nai-n3','cmct','kethoxal','2ai','bncn'}
    if not modifiers or any(m.lower() not in supported for m in modifiers): reasons.append('assay_requires_review')
    return sorted(set(reasons)),modifiers


def measured_profile(raw, errors, npos, modifiers):
    if len(raw.split())!=npos or len(errors.split())!=npos:
        raise ValueError('values_errors_seqpos_length_mismatch')
    values=np.fromstring(raw,sep=' ',dtype=np.float32)
    uncertainty=np.fromstring(errors,sep=' ',dtype=np.float32) if errors else np.array([],dtype=np.float32)
    if len(values)!=npos or len(uncertainty)!=npos: raise ValueError('values_errors_seqpos_length_mismatch')
    mask=np.isfinite(values)&(values>-100)&np.isfinite(uncertainty)&(uncertainty>=0)
    if mask.sum()<20: raise ValueError('fewer_than20_measured_positions')
    # Denominator is source-reported SEQPOS, NOT every base in the construct.
    # DMS and CMCT are base-selective: no all-base coverage threshold is imposed.
    if float(mask.mean())<0.8: raise ValueError('reported_position_coverage_below0_8')
    return values,uncertainty,mask
