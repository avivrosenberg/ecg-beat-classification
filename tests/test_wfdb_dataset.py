import os
import unittest

import wfdb

from ecgbc.dataset.wfdb_dataset import WFDBDataset
from tests import TEST_RESOURCES_PATH

WFDB_TEST_RESOURCES_PATH = f'{TEST_RESOURCES_PATH}/wfdb'
WFDB_NUMBER_OF_TEST_RESOURCES = 2


class WFDBDatasetTestBase(object):
    """
    Abstract Base class for testing the WFDBDataset. The test classes should
    inherit this.
    """
    def test_len(self):
        self.assertEqual(len(self.wfdb_dataset), WFDB_NUMBER_OF_TEST_RESOURCES)

    def test_iterable(self):
        dataset_as_list = list(self.wfdb_dataset)
        self.assertEqual(len(dataset_as_list), len(self.wfdb_dataset))

    def test_getitem(self):
        for i in range(0, len(self.wfdb_dataset)):
            sample = self.wfdb_dataset[i]
            self.assertIsInstance(sample, wfdb.Record)

    def test_data_shape(self):
        for sample in self.wfdb_dataset:
            expected_shape = (sample.sig_len, sample.n_sig)
            actual_shape = sample.p_signal.shape

            self.assertTupleEqual(expected_shape, actual_shape)

    def test_metadata_has_record_path(self):
        for sample in self.wfdb_dataset:
            self.assertTrue(hasattr(sample, 'record_path'))
            self.assertEqual(os.path.dirname(sample.record_path),
                             WFDB_TEST_RESOURCES_PATH)


class WFDBDatasetDefaultChannelsTest(WFDBDatasetTestBase, unittest.TestCase):
    def setUp(self):
        self.wfdb_dataset = WFDBDataset(WFDB_TEST_RESOURCES_PATH)

    def test_only_first_channel_taken_by_default(self):
        for sample in self.wfdb_dataset:
            self.assertEqual(sample.n_sig, 1)


class WFDBDatasetMultipleChannelsTest(WFDBDatasetTestBase, unittest.TestCase):
    def setUp(self):
        self.wfdb_dataset = WFDBDataset(WFDB_TEST_RESOURCES_PATH,
                                        first_channel_only=False)

    def test_all_channels_taken(self):
        for sample in self.wfdb_dataset:
            self.assertEqual(sample.n_sig, 2)


class WFDBDatasetCustomChannel1Test(WFDBDatasetTestBase, unittest.TestCase):
    def setUp(self):
        self.wfdb_dataset = WFDBDataset(WFDB_TEST_RESOURCES_PATH,
                                        channel_pattern='MLI+',
                                        first_channel_only=False)

    def test_only_matching_channels_taken(self):
        for sample in self.wfdb_dataset:
            self.assertEqual(sample.n_sig, 1)
            self.assertEqual(sample.sig_name[0], 'MLII')


class WFDBDatasetCustomChannel2Test(WFDBDatasetTestBase, unittest.TestCase):
    def setUp(self):
        self.wfdb_dataset = WFDBDataset(WFDB_TEST_RESOURCES_PATH,
                                        channel_pattern='MLI+|V\d',
                                        first_channel_only=False)

    def test_only_matching_channels_taken(self):
        for sample in self.wfdb_dataset:
            self.assertEqual(sample.n_sig, 2)
            self.assertEqual(sample.sig_name[0], 'MLII')
            self.assertTrue(sample.sig_name[1] == 'V1'or
                            sample.sig_name[1] == 'V5')


class WFDBDatasetCustomRecordName1Test(WFDBDatasetTestBase, unittest.TestCase):
    def setUp(self):
        self.wfdb_dataset = WFDBDataset(WFDB_TEST_RESOURCES_PATH,
                                        recname_pattern=r'\d+1',
                                        first_channel_only=False)

    def test_len(self):
        self.assertEqual(len(self.wfdb_dataset), 1)

    def test_correct_record_was_taken(self):
        self.assertEqual(self.wfdb_dataset[0].record_name, '101')


class WFDBDatasetCustomRecordName2Test(WFDBDatasetTestBase, unittest.TestCase):
    def setUp(self):
        self.wfdb_dataset = WFDBDataset(WFDB_TEST_RESOURCES_PATH,
                                        recname_pattern=r'foo\d+',
                                        first_channel_only=False)

    def test_len(self):
        self.assertEqual(len(self.wfdb_dataset), 0)


if __name__ == '__main__':
    unittest.main()
