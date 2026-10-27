import unittest

from scripts.audit_clean_release_remote_citations import citation_keys_from_api, is_valid_response_for_entry


class CleanReleaseRemoteCitationTests(unittest.TestCase):
    def test_primary_keys_are_normalized_and_unknown_values_are_ignored(self):
        payload = {
            "rcsb_id": "1ABC",
            "rcsb_primary_citation": {
                "pdbx_database_id_DOI": " 10.1000/Example ",
                "pdbx_database_id_PubMed": 12345,
            },
        }
        self.assertEqual(citation_keys_from_api(payload), {"doi:10.1000/example", "pmid:12345"})

        self.assertTrue(is_valid_response_for_entry(payload, "1abc"))
        self.assertFalse(is_valid_response_for_entry(payload, "2xyz"))

    def test_missing_primary_citation_does_not_become_a_study_key(self):
        payload = {"rcsb_id": "1ABC", "rcsb_primary_citation": None}
        self.assertEqual(citation_keys_from_api(payload), set())
        self.assertTrue(is_valid_response_for_entry(payload, "1ABC"))

    def test_raw_primary_citation_fields_are_used_when_present(self):
        payload = {
            "rcsb_id": "1ABC",
            "citation": [{
                "id": "primary",
                "pdbx_database_id_DOI": "10.1000/raw",
                "pdbx_database_id_PubMed": "67890",
            }],
        }
        self.assertEqual(citation_keys_from_api(payload), {"doi:10.1000/raw", "pmid:67890"})

    def test_entry_database_doi_is_not_misclassified_as_article_citation(self):
        from scripts.audit_clean_release_remote_citations import primary_citation_fields

        payload = {
            "rcsb_id": "1ABC",
            "citation": [{"id": "primary", "title": "A paper", "pdbx_database_id_DOI": "."}],
            "database_2": [{"pdbx_DOI": "10.2210/entry/example"}],
        }
        self.assertEqual(citation_keys_from_api(payload), set())
        fields = primary_citation_fields(payload)
        self.assertFalse(fields["primary_doi_present"])
        self.assertTrue(fields["database_2_doi_present_not_treated_as_primary"])


if __name__ == "__main__":
    unittest.main()
