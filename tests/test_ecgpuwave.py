import glob
import os
import unittest
import warnings
from pathlib import Path
import wfdb

from physionet_tools.ecgpuwave import ECGPuWave
import tests

RESOURCES_PATH = f'{tests.TEST_RESOURCES_PATH}/ecgpuwave'
TEST_ANN_EXT = 'test'


class ECGPuWaveTest(unittest.TestCase):
    def setUp(self):
        self.ecgpuwave = ECGPuWave()
        self.test_rec = f'{RESOURCES_PATH}/100s'
        self.test_rec_ann_ext = 'atr'

    def tearDown(self):
        if glob.glob(f'{RESOURCES_PATH}/fort.*'):
            self.fail("Found un-deleted temp files")

    @classmethod
    def tearDownClass(cls):
        # Delete previously-created annotations
        for ann_file_path in Path(RESOURCES_PATH).glob(f"*.{TEST_ANN_EXT}"):
            os.remove(ann_file_path)

    def test_sig0_full_noatr_num_annotations(self):
        for signal_idx in [0, None]:
            res = self.ecgpuwave(self.test_rec, TEST_ANN_EXT, signal=signal_idx)
            self.assertTrue(res)
            self._helper_check_num_annotations(684)

    def test_sig1_full_noatr_num_annotations(self):
        res = self.ecgpuwave(self.test_rec, TEST_ANN_EXT, signal=1)
        self.assertTrue(res)
        self._helper_check_num_annotations(655)

    def test_sig0_first30s_noatr_num_annotations(self):
        for to_time in ['0:30', '0:0:30', '00:00:30', 's10800']:
            res = self.ecgpuwave(self.test_rec, TEST_ANN_EXT,
                                 to_time=to_time)
            self.assertTrue(res)
            self._helper_check_num_annotations(349)

    def test_sig1_last20s_noatr_num_annotations(self):
        for from_time in ['0:40', '0:0:40', '00:00:40', 's14400']:
            res = self.ecgpuwave(self.test_rec, TEST_ANN_EXT, signal=1,
                                 from_time=from_time)
            self.assertTrue(res)
            self._helper_check_num_annotations(209)

    def test_sig0_full_withatr_num_annotations(self):
        res = self.ecgpuwave(self.test_rec, TEST_ANN_EXT,
                             in_ann_ext=self.test_rec_ann_ext)
        self.assertTrue(res)
        self._helper_check_num_annotations(670)

    def test_invalid_record_should_fail(self):
        try:
            with warnings.catch_warnings(record=True) as w:
                res = self.ecgpuwave(f'{RESOURCES_PATH}/foo', 'bar')
                self.assertFalse(res)
                self.assertEqual(1, len(w))
        finally:
            os.remove(f'{RESOURCES_PATH}/foo.bar')

    def _helper_check_num_annotations(self, expected):
        ann = wfdb.rdann(self.test_rec, TEST_ANN_EXT)
        actual = len(ann.sample)
        self.assertEqual(expected, actual, "Incorrect number of annotations")
