import unittest

from scripts.review_clean_release_citation_titles import crossref_matches, normalize_title, primary_article


class CleanReleaseCitationTitleReviewTests(unittest.TestCase):
    def test_primary_article_prefers_designated_primary_record(self):
        payload = {"citation": [
            {"id": "related", "title": "Secondary"},
            {"id": "primary", "title": "Primary", "year": "2020", "rcsb_authors": ["Smith, J.A."]},
        ]}
        self.assertEqual(primary_article(payload)["title"], "Primary")

    def test_crossref_title_year_and_author_checks_are_explicit(self):
        article = {"title": "RNA: A Study!", "year": "2020", "rcsb_authors": ["Smith, J.A."]}
        response = {"message": {"items": [
            {
                "DOI": "10.1000/example",
                "title": ["RNA — A Study"],
                "author": [{"family": "Smith"}],
                "published-print": {"date-parts": [[2020]]},
            },
            {
                "DOI": "10.1000/wrong-year",
                "title": ["RNA: A Study!"],
                "author": [{"family": "Smith"}],
                "published-print": {"date-parts": [[2019]]},
            },
        ]}}
        matches = crossref_matches(article, response)
        self.assertEqual(len(matches), 2)
        self.assertTrue(matches[0]["strict_title_year_author_match"])
        self.assertFalse(matches[1]["strict_title_year_author_match"])
        self.assertEqual(normalize_title("RNA — A Study"), "rna a study")

    def test_missing_source_year_or_author_cannot_be_called_a_strict_match(self):
        article = {"title": "RNA: A Study!", "rcsb_authors": ["Smith, J.A."]}
        response = {"message": {"items": [{
            "DOI": "10.1000/example",
            "title": ["RNA: A Study!"],
            "author": [{"family": "Smith"}],
            "published-print": {"date-parts": [[2020]]},
        }]}}
        matches = crossref_matches(article, response)
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0]["strict_title_year_author_match"])


if __name__ == "__main__":
    unittest.main()
