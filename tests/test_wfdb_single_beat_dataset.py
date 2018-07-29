import glob
import os
import shutil
import unittest
from pathlib import Path

from ecgbc.dataset.wfdb_dataset import WFDBDataset
from ecgbc.dataset.wfdb_single_beat import Generator, SingleBeatDataset
from tests import TEST_RESOURCES_PATH

WFDB_TEST_RESOURCES_PATH = f'{TEST_RESOURCES_PATH}/wfdb'
GENERATOR_OUTPUT_FOLDER = f'{WFDB_TEST_RESOURCES_PATH}/sb'
WFDB_NUMBER_OF_TEST_RESOURCES = 2


class WFDBSingleBeatDatasetTest(unittest.TestCase):

    def setUp(self):
        self.wfdb_dataset = WFDBDataset(
            WFDB_TEST_RESOURCES_PATH, first_channel_only=True
        )
        assert len(self.wfdb_dataset) > 0

    @classmethod
    def setUpClass(cls):
        # Delete previously-created annotations if they exist
        for ann_file_path in Path(WFDB_TEST_RESOURCES_PATH).glob("*.ecgatr"):
            os.remove(ann_file_path)

    @classmethod
    def tearDownClass(cls):
        # Clear output folder
        if os.path.isdir(GENERATOR_OUTPUT_FOLDER):
            shutil.rmtree(GENERATOR_OUTPUT_FOLDER)

    def test_feature_vector_length_with_rr_features(self):
        # Arrange
        resample_num_samples = 42
        generator = Generator(
            self.wfdb_dataset,
            resample_num_samples=resample_num_samples,
            calculate_rr_features=True)

        expected_feature_vector_length = resample_num_samples + 4
        self._test_feature_vector_length_helper(generator,
                                                expected_feature_vector_length)

    def test_feature_vector_length_without_rr_features(self):
        # Arrange
        resample_num_samples = 97
        generator = Generator(
            self.wfdb_dataset,
            resample_num_samples=resample_num_samples,
            calculate_rr_features=False)

        expected_feature_vector_length = resample_num_samples + 0
        self._test_feature_vector_length_helper(generator,
                                                expected_feature_vector_length)

    def _test_feature_vector_length_helper(self, generator,
                                           expected_feature_vector_length):

        self.assertGreater(len(generator), 0)

        # Act
        for segments, labels, rec_name in generator:
            # Assert
            actual_feature_vector_length = segments.shape[1]
            self.assertEqual(expected_feature_vector_length,
                             actual_feature_vector_length)

    def test_labels_match_segments(self):
        # Arrange
        generator = Generator(self.wfdb_dataset)

        # Act
        for segments, labels, rec_name in generator:
            num_labels = len(labels)
            num_segments = segments.shape[0]

            # Assert
            self.assertEqual(num_labels, num_segments)
            self.assertEqual((num_labels,), labels.shape)

    def test_write_and_load_dataset(self):
        # Arrange
        generator = Generator(self.wfdb_dataset,
                              resample_num_samples=50,
                              calculate_rr_features=False)

        # Act
        generator.write(GENERATOR_OUTPUT_FOLDER)
        dataset = SingleBeatDataset(GENERATOR_OUTPUT_FOLDER)
        all_seg_paths = glob.glob(f'{GENERATOR_OUTPUT_FOLDER}/**/*.npy')
        all_seg_labels = [
            os.path.basename(p) for p in
            glob.glob(f'{GENERATOR_OUTPUT_FOLDER}/*')
        ]

        # Assert
        self.assertEqual(len(dataset), len(all_seg_paths))
        for seg, label_num in dataset:
            label_str = dataset.classes[label_num]

            self.assertIn(label_str, all_seg_labels)
            self.assertEqual(seg.shape, (50,))

