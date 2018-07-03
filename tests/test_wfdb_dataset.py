import unittest
import numpy as np

from tests import TEST_RESOURCES_PATH
from ecgbc.dataset.wfdb_dataset import WFDBDataset


class WFDBLoaderTest(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
