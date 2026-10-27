import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DatasetGroupingTests(unittest.TestCase):
    def test_large_group_is_not_accidentally_most_of_validation(self):
        spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_experimental_pairs.py')
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        groups = [[{'id': f'g{i}_r{j}'} for j in range(size)] for i, size in enumerate([42, 4, 2] + [1] * 13)]
        allocated = builder.assign_groups(groups)
        sizes = {s: sum(map(len, gs)) for s, gs in allocated.items()}
        self.assertEqual(sum(sizes.values()), 61)
        self.assertGreater(sizes['train'], 42)
        self.assertGreater(sizes['validation'], 0)
        self.assertGreater(sizes['test'], 0)
        self.assertEqual(sum(len(gs) for gs in allocated.values()), len(groups))

    def test_similar_sequences_merge_different_metadata_groups(self):
        spec = importlib.util.spec_from_file_location('builder', ROOT / 'scripts/build_experimental_pairs.py')
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        records = [{'id': 'a', 'sequence': 'A' * 40, 'homology_components': ['component_1']},
                   {'id': 'b', 'sequence': 'A' * 39 + 'G', 'homology_components': ['component_2']},
                   {'id': 'c', 'sequence': 'C' * 40, 'homology_components': ['component_3']}]
        groups, edges = builder.grouping(records)
        self.assertEqual(sorted(len(g) for g in groups), [1, 2])
        self.assertEqual(len(edges), 1)
        self.assertGreaterEqual(edges[0]['global_identity'], 0.8)


if __name__ == '__main__':
    unittest.main()
