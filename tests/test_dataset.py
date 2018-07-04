import unittest
import numpy as np
import re

from tests import TEST_RESOURCES_PATH

from ecgbc.dataset import ECG_CHANNEL_PATTERN
from ecgbc.dataset.wfdb_dataset import WFDBDataset


class WFDBDatasetTest(unittest.TestCase):

    def setUp(self):
        self.wfdb_dataset = WFDBDataset(f'{TEST_RESOURCES_PATH}/wfdb')

    def test_len(self):
        self.assertEqual(len(self.wfdb_dataset), 2)

    def test_iterable(self):
        dataset_as_list = list(self.wfdb_dataset)
        self.assertEqual(len(dataset_as_list), len(self.wfdb_dataset))

    def test_getitem(self):
        for i in range(0, len(self.wfdb_dataset)):
            sample = self.wfdb_dataset[i]
            self.assertIsInstance(sample, dict)
            self.assertIsInstance(sample['signals'], np.ndarray)
            self.assertIsInstance(sample['fields'], dict)

    def test_data_shape(self):
        for sample in self.wfdb_dataset:
            expected_shape = (
                sample['fields']['sig_len'],
                sample['fields']['n_sig'],
            )
            actual_shape = sample['signals'].shape

            self.assertTupleEqual(expected_shape, actual_shape)

    def test_metadata_has_record_name(self):
        for sample in self.wfdb_dataset:
            self.assertTrue('rec_name' in sample['fields'])


class ECGChannelRegexTest(unittest.TestCase):
    def setUp(self):
        self.pattern = re.compile(ECG_CHANNEL_PATTERN, re.IGNORECASE)

        self.positive_test_cases = [
            'ecg', 'foo ECG', 'ECG_bar', 'ECG1', 'ecg2', 'foo ecg3', 'foo_ecg',
            'MLI', 'MLII', 'MLIII', 'foo MLI', 'MLIIII bar',
            'V5', 'foo v1', 'v4 bar',
            'lead ii', 'foo lead iii', 'lead I bar', 'ECG Lead II',
            'iiii', 'foo II', 'II bar',
        ]

        self.negative_test_cases = [
            'foobar', 'foo bar baz', 'foo_bar', 'foo1', 'foo 1', '2 bar',
            '2_bar', 'foo foo bar', 'bar_foo_baz', 'IIfoo', 'fooIII',
            'wrist_ppg', 'PPG', 'resp', 'ART', 'PAP', 'CVP', 'resp imp.',
            'CO2',
        ]

    def test_positive_cases(self):
        for teststr in self.positive_test_cases:
            for teststr_case in [teststr, teststr.upper(), teststr.lower()]:
                match = self.pattern.search(teststr_case)
                self.assertIsNotNone(match, msg=teststr_case)

    def test_negative_cases(self):
        for teststr in self.negative_test_cases:
            for teststr_case in [teststr, teststr.upper(), teststr.lower()]:
                match = self.pattern.search(teststr_case)
                self.assertIsNone(match, msg=teststr_case)


if __name__ == '__main__':
    unittest.main()
