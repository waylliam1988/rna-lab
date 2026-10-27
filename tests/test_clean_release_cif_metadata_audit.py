import unittest

import gemmi

from scripts.audit_clean_release_cif_metadata import summarize_record


class CleanReleaseCifMetadataAuditTests(unittest.TestCase):
    def test_extracts_primary_study_and_selected_entity_crosswalk(self):
        block = gemmi.cif.read_string("""data_fixture
_entry.id TEST
loop_
_citation.id
_citation.pdbx_database_id_PubMed
_citation.pdbx_database_id_DOI
_citation.title
primary 12345 10.1000/example 'Primary paper'
loop_
_entity.id
_entity.src_method
_entity.pdbx_description
_entity.pdbx_mutation
_entity.pdbx_fragment
1 man 'RNA construct' . .
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.pdb_strand_id
A X
loop_
_entity_poly.entity_id
_entity_poly.pdbx_seq_one_letter_code_can
1 ACGU
loop_
_entity_src_gen.entity_id
1
loop_
_struct_ref.entity_id
_struct_ref.id
1 ref1
loop_
_struct_ref_seq.ref_id
ref1
""").sole_block()

        facts = summarize_record(block, {
            "pdb_id": "TEST",
            "entity_id": "1",
            "label_asym_id": "A",
            "auth_asym_id": "X",
            "sequence": "ACGU",
        })

        self.assertTrue(facts["entry_id_match"])
        self.assertTrue(facts["has_primary_citation"])
        self.assertTrue(facts["has_primary_doi"])
        self.assertTrue(facts["has_primary_pubmed_id"])
        self.assertTrue(facts["label_asym_entity_match"])
        self.assertTrue(facts["label_auth_mapping_match"])
        self.assertTrue(facts["canonical_entity_sequence_exact_match"])
        self.assertTrue(facts["entity_source_gen_present"])
        self.assertTrue(facts["entity_reference_sequence_mapping_present"])

    def test_secondary_citation_and_unknown_entity_values_do_not_claim_primary_provenance(self):
        block = gemmi.cif.read_string("""data_fixture
_entry.id TEST
loop_
_citation.id
_citation.pdbx_database_id_PubMed
_citation.pdbx_database_id_DOI
_citation.title
related . ? 'Related paper'
loop_
_entity.id
_entity.src_method
_entity.pdbx_description
_entity.pdbx_mutation
_entity.pdbx_fragment
1 ? . ? .
""").sole_block()

        facts = summarize_record(block, {
            "pdb_id": "TEST",
            "entity_id": "1",
            "label_asym_id": "A",
            "auth_asym_id": "X",
            "sequence": "ACGU",
        })

        self.assertTrue(facts["has_any_citation_row"])
        self.assertFalse(facts["has_primary_citation"])
        self.assertFalse(facts["has_primary_doi"])
        self.assertFalse(facts["has_primary_pubmed_id"])
        self.assertFalse(facts["entity_src_method_known"])
        self.assertFalse(facts["entity_description_known"])


if __name__ == "__main__":
    unittest.main()
