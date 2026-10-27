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


if __name__ == "__main__":
    unittest.main()
