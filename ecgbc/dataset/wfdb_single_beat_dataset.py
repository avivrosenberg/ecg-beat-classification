import math

import numpy as np
import scipy.interpolate as interp
import scipy.signal as signal
import os.path
import re
import subprocess
import warnings

import wfdb
from torch.utils.data import Dataset

from ecgbc.dataset import BEAT_ANNOTATIONS_PATTERN
from ecgbc.dataset.wfdb_dataset import WFDBDataset


class WFDBSingleBeatDataset(Dataset):
    ECGPUWAVE_BIN = 'bin/ecgpuwave-1.3.3/ecgpuwave'
    DEFAULT_RESAMPLE_DURATION_S = 0.8
    DEFAULT_RESAMPLE_NUM_SAMPLES = 50
    RR_MAX = 1.5
    RR_MIN = 0.4

    def __init__(self, wfdb_dataset, in_ann_ext='atr', out_ann_ext='ecgatr',
                 resample_duration_s=DEFAULT_RESAMPLE_DURATION_S,
                 resample_num_samples=DEFAULT_RESAMPLE_NUM_SAMPLES,
                 calculate_rr_features=True, filter_rri=False,
                 ecgpuwave_bin=ECGPUWAVE_BIN, transform=None):
        """
        A dataset of single ECG beats extracted from WFDB records.
        :type wfdb_dataset: WFDBDataset
        :param wfdb_dataset: A dataset of WFDB records.
        :param in_ann_ext: File extension of annotator containing beat types.
        :param out_ann_ext: File extension of annotator containing morphology.
        """
        self.wfdb_dataset = wfdb_dataset
        self.in_ann_ext = in_ann_ext
        self.out_ann_ext = out_ann_ext
        self.resample_duration_s = resample_duration_s
        self.resample_num_samples = resample_num_samples
        self.calculate_rr_features = calculate_rr_features
        self.filter_rri = filter_rri
        self.ecgpuwave_bin = ecgpuwave_bin
        self.transform = transform

        # Compile once to reduce per-record overhead
        self.beat_annotations_pattern = re.compile(
            BEAT_ANNOTATIONS_PATTERN, re.VERBOSE)

    def __getitem__(self, index):
        record = self.wfdb_dataset[index]

        morph_ann = self.load_morphology_annotation(record)
        if not morph_ann:
            return None, None

        segments, labels = self.generate_beat_segments(record, morph_ann)

        if self.transform is not None:
            segments, labels = self.transform((segments, labels))

        return segments, labels

    def __len__(self):
        return len(self.wfdb_dataset)

    def generate_beat_segments(self, record, ann):
        # Construct a long string of annotation symbols, since we are
        # looking for a specific pattern of symbols.
        joined_ann = str.join('', ann.symbol)

        # Find patterns of p-wave then a beat annotation (N-normal,
        # V-premature ventricular, S-premature supraventricular, F-fusion
        # of ventricular and normal, Q-Unclassifiable) then a t-wave. The
        # parenthesis '(' and ')' annotations mark the start an end of a
        # waveform. We want to extract the signal from the start of p-wave
        # to the end of the t-wave so we're looking for annotations patterns
        # like this: (p)(N)(t).
        matches = list(self.beat_annotations_pattern.finditer(joined_ann))

        num_segments = len(matches)
        seg_length = self.resample_num_samples

        r_peaks = []
        beat_segments = np.empty((seg_length, num_segments))
        beat_labels = np.empty((num_segments,), dtype=str)

        for i, m in enumerate(matches):
            ecg_beat_feature_samples = [
                ann.sample[m.start('p_start')],
                ann.sample[m.start('t_end')],
            ]

            seg = self.resample_segment(record, ecg_beat_feature_samples)
            r_peaks.append(ann.sample[m.start('r')])

            beat_segments[:, i] = seg
            beat_labels[i] = m.group('r')

        if self.calculate_rr_features:
            r_peak_times = np.array(r_peaks) / record.fs

            rri, rrt, filter_idx = self.rr_intervals(r_peak_times)
            beat_segments = beat_segments[:, filter_idx]
            beat_labels = beat_labels[filter_idx]

            rr_features, filter_idx = self.rri_features(rri, rrt)
            beat_segments = beat_segments[:, filter_idx]
            beat_labels = beat_labels[filter_idx]

            beat_segments = np.vstack((
                beat_segments,
                rr_features
            ))

        return beat_segments, beat_labels

    def rr_intervals(self, r_peak_times):
        rri = np.diff(r_peak_times)  # Prepend zero interval
        rrt = np.r_[0, np.cumsum(rri[:-1])]

        # Filter out non-physiological intervals
        if self.filter_rri:
            filter_idx = (rri < self.RR_MAX) & (rri > self.RR_MIN)
            rri = rri[filter_idx]
            rrt = rrt[filter_idx]
        else:
            filter_idx = np.full_like(rri, True, dtype=bool)

        # Prepend false because the first peak in r_peak_times has no interval
        filter_idx = np.r_[False, filter_idx]

        return rri, rrt, filter_idx

    def rri_features(self, rri, rrt):
        rr_features_next = rri
        rr_features_prev = np.r_[0, rri[:-1]]

        rri_features = np.vstack((
            rr_features_next,
            rr_features_prev,
        ))

        # Resample the RR intervals with uniform rate so we can apply filters
        fs_resampled = 4  # Hz
        rrt_rs = np.r_[0:rrt[-1]:1/fs_resampled]
        rri_rs = interp.interp1d(rrt, rri)(rrt_rs)

        def sliding_window_mean(sig, m):
            filter_kernel = np.ones((m,))
            filter_kernel /= m
            result = signal.convolve(sig, filter_kernel, mode='full')
            # Define N as the signal length and M as the filter length.
            # Taking the first N values from the 'full' convolution means were
            # doing a moving average that looks at the past. Taking the last N
            # would be like looking at the future. Taking the 'middle' would be
            # like a centered moving average.
            # result = result[m-1:]  # looking forward
            result = result[0:len(rri_rs)]  # looking back
            return result

        sliding_win_durations_s = [10, 5 * 60]
        for sliding_window_s in sliding_win_durations_s:
            filter_len_samples = sliding_window_s * fs_resampled
            rr_intervals_mean = sliding_window_mean(rri_rs, filter_len_samples)
            rr_intervals_mean_at = interp.interp1d(
                rrt_rs, rr_intervals_mean, fill_value='extrapolate'
            )

            # Stack the filter result in the features matrix
            rri_features = np.vstack((
                rri_features,
                # interpolate filter results at rrt
                rr_intervals_mean_at(rrt)
            ))

        # Remove features up to time of the longest filter, because only then
        # is the filter performing a valid calculation
        filter_idx = rrt > np.max(sliding_win_durations_s)
        rrt = rrt[filter_idx]
        rri = rri[filter_idx]
        rri_features = rri_features[:, filter_idx]

        return rri_features, filter_idx

    def resample_segment(self, record, segment_idx):
        """
        Resamples a given segment from the record.
        :param record: wfdb record.
        :param segment_idx: An array of at least two sample indices. The first
        is the first sample of the segment and the last is the last sample
        of the segment.
        :return: The values of the resampled segment
        """
        seg_start = segment_idx[0]
        seg_end = segment_idx[-1]

        duration_s = (seg_end - seg_start) / record.fs

        delta_samples = math.floor(
            math.fabs(self.resample_duration_s - duration_s) * record.fs / 2
        )

        # Calculate start and end indices of the resampled segment
        if duration_s < self.resample_duration_s:
            seg_start -= delta_samples
            seg_end += delta_samples
        else:
            seg_start += delta_samples
            seg_end -= delta_samples

        # Find resampled-segment indices
        segment_idx = np.r_[seg_start:seg_end+1]
        segment_sig = np.reshape(record.p_signal[segment_idx],
                                 segment_idx.shape)
        resampled_idx = np.linspace(
            seg_start, seg_end, num=self.resample_num_samples, endpoint=True
        )

        # Interpolation
        interpolator = interp.interp1d(segment_idx, segment_sig)
        segment_sig_resampled = interpolator(resampled_idx)

        return segment_sig_resampled

    def load_morphology_annotation(self, record):
        record_path = record.record_path

        if not os.path.isfile(f'{record_path}.{self.out_ann_ext}'):
            if not self.generate_morphology_annotation(record):
                return None

        return wfdb.rdann(record_path, self.out_ann_ext)

    def generate_morphology_annotation(self, record):
        """
        Runs an annotator (ecgpuwave) to create morphology annotations.
        :param record: The record to create annotations for.
        """
        ecgpuwave_args = [
            self.ecgpuwave_bin,
            '-r', record.record_path,
            '-a', self.out_ann_ext,
        ]

        if self.in_ann_ext:
            ecgpuwave_args += ['-i', self.in_ann_ext]

        try:
            ecgpuwave_result = subprocess.run(
                ecgpuwave_args,
                check=True, shell=False, universal_newlines=True, timeout=10,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # ecgpuwave can sometimes fail but still return 0, so need to
            # also check the stderr output.
            if ecgpuwave_result.stderr:
                # Annoying case: sometimes ecgpuwave writes to stderr but it's
                # not an error...
                if not re.match(r'Rearranging annotations[\w\s.]+done!',
                                ecgpuwave_result.stderr):
                    raise subprocess.CalledProcessError(0, ecgpuwave_args)

        except subprocess.CalledProcessError as process_err:
            warnings.warn(f'Failed to run ecgpuwave on record '
                          f'{record.record_path}:\n'
                          f'stderr: {ecgpuwave_result.stderr}\n'
                          f'stdout: {ecgpuwave_result.stdout}\n')
            return False

        except subprocess.TimeoutExpired as timeout_err:
            warnings.warn(f'Timed-out runnning ecgpuwave on record '
                          f'{record.record_path}: '
                          f'{ecgpuwave_result.stdout}')
            return False

        return True
