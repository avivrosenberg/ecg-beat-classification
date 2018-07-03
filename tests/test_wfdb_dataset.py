import unittest
import numpy as np

from tests import TEST_RESOURCES_PATH
from ecgbc.dataset.wfdb_dataset import WFDBDataset


class WFDBLoaderTest(unittest.TestCase):

    def setUp(self):
        self.wfdb_loader = WFDBDataset(f'{TEST_RESOURCES_PATH}/wfdb')

    def test_len(self):
        self.assertEqual(len(self.wfdb_loader), 2)

    def test_getitem(self):
        sample0 = self.wfdb_loader[0]
        sample1 = self.wfdb_loader[1]
        samples = [sample0, sample1]

        for sample in samples:
            self.assertIsInstance(sample, dict)
            self.assertIsInstance(sample['signals'], np.ndarray)
            self.assertIsInstance(sample['fields'], dict)

            expected_shape = (
                sample['fields']['sig_len'],
                sample['fields']['n_sig'],
            )
            actual_shape = sample['signals'].shape

            self.assertTupleEqual(expected_shape, actual_shape)


if __name__ == '__main__':
    unittest.main()
