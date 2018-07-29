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


class DatasetFolder(torchvision.datasets.DatasetFolder):
    def __init__(self, root, loader, extensions, transform=None,
                 target_transform=None):
        super(DatasetFolder, self).__init__(
            root, loader, extensions, transform=transform,
            target_transform=target_transform)

        self.idx_to_class = {idx: cls
                             for cls, idx in self.class_to_idx.items()}

    def samples_per_class(self):
        samples_per_class = {cls: 0 for cls in self.classes}
        for _, class_idx in self.samples:
            samples_per_class[self.idx_to_class[class_idx]] += 1
        return samples_per_class

    def __repr__(self):
        fmt_str = super(DatasetFolder, self).__repr__()
        fmt_str += '\n'
        fmt_str +=\
            '    Samples per class: {}\n'.format(self.samples_per_class())
        return fmt_str


class SingleBeatDataset(DatasetFolder):
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
    DEFAULT_RESAMPLE_DURATION_S = 0.65  # For humans should be 0.55 ~ 0.75
    DEFAULT_RESAMPLE_NUM_SAMPLES = 50
    RR_MAX = 1.5
    RR_MIN = 0.4

    AAMI_TO_WFDB_CLASSES = {
        'N': {'N', 'L', 'R', 'B'},
        'S': {'a', 'J', 'A', 'S', 'j', 'e', 'n'},
        'V': {'V', 'E'},
        'F': {'F'},
        'Q': {'/', 'f', 'Q'},
    }

    WFDB_TO_AAMI_CLASSES = {
        wfdb: aami
        for aami, wfdb_set in AAMI_TO_WFDB_CLASSES.items()
        for wfdb in wfdb_set
    }

    def __init__(self, wfdb_dataset, in_ann_ext='atr',
                 out_ann_ext='ecgatr',
                 resample_duration_s=DEFAULT_RESAMPLE_DURATION_S,
                 resample_num_samples=DEFAULT_RESAMPLE_NUM_SAMPLES,
                 calculate_rr_features=True, filter_rri=False,
                 aami_compatible=False):
        """
        A dataset of single ECG beats extracted from WFDB records.
        :type wfdb_dataset: WFDBDataset
        :param wfdb_dataset: A dataset of WFDB records.
        :param in_ann_ext: File extension of annotator containing beat types.
        :param out_ann_ext: File extension of annotator containing morphology.
        :param aami_compatible: Whether labels should be AAMI classes or
            PhysioNet classes.
        """
        self.wfdb_dataset = wfdb_dataset
        self.in_ann_ext = in_ann_ext
        self.out_ann_ext = out_ann_ext
        self.resample_duration_s = resample_duration_s
        self.resample_num_samples = resample_num_samples
        self.calculate_rr_features = calculate_rr_features
        self.filter_rri = filter_rri
        self.aami_compatible = aami_compatible

        # Compile once to reduce per-record overhead
        self.beat_annotations_pattern = re.compile(
            ecgbc.dataset.BEAT_ANNOTATIONS_PATTERN_PEAKS_ONLY, re.VERBOSE)

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
            # Extract ECG beat segment
            seg = self.resample_segment(record,
                                        r_peak_idx=ann.sample[m.start('r')])
            beat_segments[:, i] = seg

            # Save R-peak location
            r_peaks.append(ann.sample[m.start('r')])

            # Save beat label
            beat_labels[i] = m.group('r')
            if self.aami_compatible:
                beat_labels[i] = self.WFDB_TO_AAMI_CLASSES[beat_labels[i]]

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
            result = signal.filtfilt(filter_kernel, 1, sig, method='pad')
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

    def resample_segment(self, record, p_start_idx=None, r_peak_idx=None):
        """
        Resamples a given segment from the record.
        :param p_start_idx: Sample index of p-wave onset.
        :param r_peak_idx: Sample index of r-peak. One of the two indices
            must be provided.
        :param record: wfdb record.
        :return: The values of the resampled segment
        """

        # Calculate length of segment in samples
        delta_samples = math.floor(self.resample_duration_s * record.fs)

        if p_start_idx:
            seg_start_idx = p_start_idx
        elif r_peak_idx:
            seg_start_idx = r_peak_idx - delta_samples // 2
        else:
            raise ValueError('Either p_start or r_peak index must be provided')

        # Calculate end of segment index
        seg_end_idx = seg_start_idx + delta_samples

        # Get signal data within the segment
        seg_idx = np.r_[seg_start_idx:seg_end_idx+1]
        segment_sig = np.reshape(record.p_signal[seg_idx],
                                 seg_idx.shape)

        resampled_idx = np.linspace(
            seg_start_idx, seg_end_idx,
            num=self.resample_num_samples, endpoint=True
        )

        # Interpolation
        interpolator = interp.interp1d(seg_idx, segment_sig)
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

