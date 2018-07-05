import numpy as np
import scipy.signal as scipysignal


class SubtractMedianFilterWFDB(object):
    """
    Transform that removes the baseline of a signal by subtracting the
    result of one or more median filters from the signal. Works with WFDB
    records.
    """

    FILTER1_MS = 200
    FILTER2_MS = 600

    def __init__(self, filters=(FILTER1_MS, FILTER2_MS)):
        assert len(filters) > 0
        self.filter_durations = filters

    def __call__(self, record):
        """
        Applies the transformation to the given record (each channel
        individually).

        :param record: A wfdb record to transform.
        :type record: wfdb.Record
        :return: The same WFDB record, with filtered data.
        """
        filtered_signal = self.filter_signal(record.p_signal, record.fs)
        record.p_signal = filtered_signal
        return record

    def filter_signal(self, signal_data, fs):
        result = np.zeros_like(signal_data)
        n_channels = signal_data.shape[1]

        for chan_idx in range(n_channels):
            chan_data = signal_data[:, chan_idx]
            chan_data_filtered = self.filter_channel(chan_data, fs)

            # subtract the filter result to remove the baseline
            result[:, chan_idx] = chan_data - chan_data_filtered

        return result

    def filter_channel(self, channel_data, fs):
        result = channel_data

        for filter_duration_ms in self.filter_durations:
            # Calculate the length of the filter in samples, and make sure
            # it's odd
            filter_in_samples = int(fs * (filter_duration_ms / 1000))
            if filter_in_samples % 2 == 0:
                filter_in_samples += 1

            # Apply the current filter on the result of the previous filter
            result = scipysignal.medfilt(result, filter_in_samples)

        return result
