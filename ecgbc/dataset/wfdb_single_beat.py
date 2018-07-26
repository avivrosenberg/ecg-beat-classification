import collections
import math
import os.path
import re

import numpy as np
import scipy.interpolate as interp
import scipy.signal as signal
import torchvision
import tqdm
import wfdb

import ecgbc.dataset
import physionet_tools.ecgpuwave

from ecgbc.dataset.wfdb_dataset import WFDBDataset


class Dataset(torchvision.datasets.DatasetFolder):
    """
    A dataset of WFDB single beats.
    Use the Generator class in this module to write a dataset based on WFDB
    records to some folder. Then, this class cal be used to load the samples
    from that folder.
    """
    def __init__(self, root_folder):
        super().__init__(root_folder, self.load_segment, ['.npy'])

    def load_segment(self, path):
        return np.load(path)


class Generator(object):
    """
    Generates a WFDB single-beat database and writes it to a folder where
    each file is a labeled sample.
    """
    DEFAULT_RESAMPLE_DURATION_S = 0.8
    DEFAULT_RESAMPLE_NUM_SAMPLES = 50
    RR_MAX = 1.5
    RR_MIN = 0.4

    def __init__(self, wfdb_dataset, in_ann_ext='atr',
                 out_ann_ext='ecgatr',
                 resample_duration_s=DEFAULT_RESAMPLE_DURATION_S,
                 resample_num_samples=DEFAULT_RESAMPLE_NUM_SAMPLES,
                 calculate_rr_features=True, filter_rri=False):
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

        # Compile once to reduce per-record overhead
        self.beat_annotations_pattern = re.compile(
            ecgbc.dataset.BEAT_ANNOTATIONS_PATTERN, re.VERBOSE)

    def write(self, output_folder):
        """
        Writes the segments generated from the underlying wfdb_dataset to a
        given output folder.
        :param output_folder: Where to write to.
        """
        if not len(self):
            return

        paths = collections.deque(self.wfdb_dataset.rec_paths)
        self_iter_with_progress_bar = tqdm.tqdm(self, desc=paths.popleft())

        for segments, labels, rec_name in self_iter_with_progress_bar:
            for seg_idx, segment in enumerate(segments):
                seg_label = labels[seg_idx]
                seg_dir = f'{output_folder}/{seg_label}'
                seg_path = f'{seg_dir}/{rec_name}_{seg_idx}'
                os.makedirs(seg_dir, exist_ok=True)

                np.save(seg_path, segment, allow_pickle=False)

            if len(paths):
                self_iter_with_progress_bar.set_description(paths.popleft())

    def __getitem__(self, index):
        """
        Returns a record from the wfdb_dataset after segmenting it to single
        beats, with a label for each beat.
        :param index: Index of file in the wfdb_dataset
        :return: Tuple of:
            - segments: NxM matrix (N segments, M features per segments).
            - labels: vector of length N, containing the beat type per segment.
            - rec_name: The name of the record the segments came from.
        """
        record = self.wfdb_dataset[index]

        morph_ann = self.load_morphology_annotation(record)
        if not morph_ann:
            raise RuntimeError(f"Can't load or generate morphology "
                               f"annotations for record {record.record_name}")

        segments, labels = self.generate_beat_segments(record, morph_ann)

        # Transpose because externally (specifically in pytorch) the convention
        # is that the first dimension is the batch dimension.
        segments = segments.transpose()

        return segments, labels, record.record_name

    def __len__(self):
        return len(self.wfdb_dataset)

    def __iter__(self):
        class GeneratorIter:
            def __init__(self, generator):
                self.generator = generator
                self.i = 0

            def __next__(self):
                if self.i == len(self.generator):
                    raise StopIteration
                next_res = self.generator[self.i]
                self.i += 1
                return next_res

        return GeneratorIter(self)

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

        if num_segments < 2:
            return beat_segments, beat_labels

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

            if len(rri) > 2:
                rr_features = self.rri_features(rri, rrt)

                beat_segments = np.vstack((
                    beat_segments,
                    rr_features
                ))

        ### DEBUG
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(nrows=8, ncols=3, sharex='col')
        # axes = np.reshape(axes, (-1,))
        # beat_idx = np.random.permutation(beat_segments.shape[1])[0:len(axes)]
        # for i, ax in enumerate(axes):
        #     ax.plot(beat_segments[:-4, beat_idx[i]])
        # fig.show()
        #

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
        if len(rri) < 2:
            return np.empty((0,))

        rr_features_next = rri
        rr_features_prev = np.r_[0, rri[:-1]]

        rri_features = np.vstack((
            rr_features_next,
            rr_features_prev,
        ))

        # Resample the RR intervals with uniform rate so we can apply filters
        fs_resampled = 4  # Hz
        rrt_rs = np.r_[rrt[0]:rrt[-1]:1/fs_resampled]
        rri_rs = interp.interp1d(rrt, rri)(rrt_rs)

        def sliding_window_mean(sig, m):
            # Take half the filter length since we're using filtfilt
            m = np.min((m, len(sig))) // 2
            filter_kernel = np.ones((m,))
            filter_kernel /= m
            result = signal.filtfilt(filter_kernel, 1, sig, method='gust')
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

        ### DEBUG
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(nrows=1, ncols=1)
        # ax.plot(rrt, rri)
        # ax.plot(rrt, rri_features[2, :], 'o')
        # ax.plot(rrt, rri_features[3, :], 'x')
        # ax.plot(rrt, np.ones_like(rrt)*np.mean(rri), '+')
        # ax.legend(['rri_rs', '10sec', '5min', 'peak_times'])
        # fig.show()
        #

        return rri_features

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
            # Use an external tool, ecgpuwave, to generate the annotations
            ecgpuwave = physionet_tools.ecgpuwave.ECGPuWave()
            if not ecgpuwave(record_path, self.out_ann_ext, self.in_ann_ext):
                return None

        return wfdb.rdann(record_path, self.out_ann_ext)

