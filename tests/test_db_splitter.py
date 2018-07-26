import argparse
import io
import unittest
import physionet_tools.db_splitter as db_splitter

import tests

TEST_RESOURCES_PATH = f'{tests.TEST_RESOURCES_PATH}/db_splitter'


class SplitByAgeTest(unittest.TestCase):
    def setUp(self):
        self.rec_to_age = {
            'chf10': 22,
            '16265': 32,
            '111': 47,
            '234': 56,
            'nsr001': 64,
            'chf203': 68,
            '100': 69,
            '103': None,
            'chf06': None
        }
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def _split_and_test_groups(self, age_groups):
        splitter = db_splitter.SplitByAge(age_groups,
                                          in_dir=TEST_RESOURCES_PATH,
                                          stdout=self.stdout,
                                          stderr=self.stderr)
        splitter.split()

        for group, members in splitter.group_affiliation.items():
            for rec in members:
                expected_age = self.rec_to_age[rec.stem]

                if not expected_age:
                    # Age is not available for record, group should be None
                    self.assertIsNone(group,
                                      msg=f'group should be None for {rec}')
                elif not group:
                    # Group is None but age is available, so it must not
                    # fall in any age group
                    for age_group in age_groups:
                        self.assertNotIn(
                            expected_age, range(*age_group),
                            msg=f"{rec} shouldn't be in any group")
                else:
                    actual_range = range(*group)
                    self.assertIn(expected_age, actual_range, msg=str(rec))

    def test_splitting_non_contiguous(self):
        age_groups = [(10, 46), (60, 99)]
        self._split_and_test_groups(age_groups)

    def test_splitting_contiguous_two_groups(self):
        age_groups = [(10, 46), (46, 99)]
        self._split_and_test_groups(age_groups)

    def test_splitting_contiguous_three_groups(self):
        age_groups = [(10, 46), (46, 60), (60, 99)]
        self._split_and_test_groups(age_groups)

    def test_single_group_all(self):
        age_groups = [(0, 100)]
        self._split_and_test_groups(age_groups)

    def test_single_group_none(self):
        age_groups = [(0, 10)]
        self._split_and_test_groups(age_groups)

    def test_single_group_partial(self):
        age_groups = [(40, 60)]
        self._split_and_test_groups(age_groups)

    def test_invalid_age_groups(self):
        self.assertRaises(AssertionError,
                          self._split_and_test_groups, [(-40, -2)])

        self.assertRaises(AssertionError,
                          self._split_and_test_groups, [('foo', 'bar')])

        self.assertRaises(AssertionError,
                          self._split_and_test_groups, [(1, 2, 3)])

        self.assertRaises(AssertionError,
                          self._split_and_test_groups, [(2,)])
