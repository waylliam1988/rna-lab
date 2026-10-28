"""Next-study candidate review; not wired into legacy training admission."""


def candidate_review(*, provenance_verified, quality_verified,
                     mapping_verified, contains_model_predictions,
                     supervision_scope, active_holdout_relation,
                     retired_evaluation_relation):
    """Retired evaluation history is informative, never a quality waiver.

    active_holdout_relation: True=related, False=reviewed clear, None=unknown.
    False must result from a documented relation review, not a missing match.
    A candidate still needs a versioned dataset-level admission decision.
    """
    if active_holdout_relation not in (True, False, None):
        raise ValueError('Unknown active relation state')
    reasons = []
    if active_holdout_relation is True:
        reasons.append('related_to_active_development_or_test')
    elif active_holdout_relation is not False:
        reasons.append('active_holdout_isolation_unresolved')
    if contains_model_predictions is True:
        reasons.append('model_predicted_or_imputed_labels_excluded')
    elif contains_model_predictions is not False:
        reasons.append('label_origin_unresolved')
    if provenance_verified is not True:
        reasons.append('provenance_unresolved')
    if quality_verified is not True:
        reasons.append('quality_unresolved')
    if mapping_verified is not True:
        reasons.append('sequence_label_mapping_unresolved')
    if supervision_scope != 'reviewed_binary_pair_scope':
        reasons.append('binary_pair_supervision_scope_not_ready')
    return dict(
        status='candidate_for_versioned_dataset_review' if not reasons else 'retain_quarantine',
        can_enter_candidate_manifest=not reasons,
        training_allowed=False,
        blocking_reasons=reasons,
        retired_evaluation_relation=retired_evaluation_relation,
        historical_overlap_is_sole_blocker=False,
    )


def effective_reference_role(pool_role, exact_active_development_alias):
    """A retired pool cannot deprotect the same sequence in active development."""
    return 'protected_development_alias' if exact_active_development_alias else pool_role
