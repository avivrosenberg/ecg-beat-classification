import os
import unittest
from pathlib import Path

from ecgbc.dataset.wfdb_dataset import WFDBDataset
from ecgbc.dataset.wfdb_single_beat_dataset import WFDBSingleBeatDataset
from tests import TEST_RESOURCES_PATH

WFDB_TEST_RESOURCES_PATH = f'{TEST_RESOURCES_PATH}/wfdb'
WFDB_NUMBER_OF_TEST_RESOURCES = 2


class WFDBSingleBeatDatasetTest(unittest.TestCase):

    def setUp(self):
        self.wfdb_dataset = WFDBDataset(
            WFDB_TEST_RESOURCES_PATH, first_channel_only=True
        )
        assert len(self.wfdb_dataset) > 0

    @classmethod
    def tearDownClass(cls):
        # Delete previously-created annotations
        for ann_file_path in Path(WFDB_TEST_RESOURCES_PATH).glob("*.ecgatr"):
            os.remove(ann_file_path)

    def test_feature_vector_length_with_rr_features(self):
        # Arrange
        resample_num_samples = 42
        wfdb_single_beat_dataset = WFDBSingleBeatDataset(
            self.wfdb_dataset,
            resample_num_samples=resample_num_samples,
            calculate_rr_features=True)

        expected_feature_vector_length = resample_num_samples + 4
        self._test_feature_vector_length_helper(wfdb_single_beat_dataset,
                                                expected_feature_vector_length)

    def test_feature_vector_length_without_rr_features(self):
        # Arrange
        resample_num_samples = 97
        wfdb_single_beat_dataset = WFDBSingleBeatDataset(
            self.wfdb_dataset,
            resample_num_samples=resample_num_samples,
            calculate_rr_features=False)

        expected_feature_vector_length = resample_num_samples + 0
        self._test_feature_vector_length_helper(wfdb_single_beat_dataset,
                                                expected_feature_vector_length)

    def _test_feature_vector_length_helper(self,
                                           wfdb_single_beat_dataset,
                                           expected_feature_vector_length):

        self.assertGreater(len(wfdb_single_beat_dataset), 0)

        # Act
        for segments, labels in wfdb_single_beat_dataset:
            # Assert
            actual_feature_vector_length = segments.shape[1]
            self.assertEqual(expected_feature_vector_length,
                             actual_feature_vector_length)

    def test_labels_match_segments(self):
        # Arrange
        wfdb_single_beat_dataset = WFDBSingleBeatDataset(self.wfdb_dataset)

        # Act
        for segments, labels in wfdb_single_beat_dataset:
            num_labels = len(labels)
            num_segments = segments.shape[0]

            # Assert
            self.assertEqual(num_labels, num_segments)
            self.assertEqual((num_labels,), labels.shape)


if __name__ == '__main__':
    unittest.main()
