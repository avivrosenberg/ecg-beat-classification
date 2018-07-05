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


class LowPassFilterWFDB(object):
    """
    Transform that applies a lowpass filter to a signal.
    Works with WFDB records.
    """

    DEFAULT_ORDER = 12
    DEFAULT_CUTOFF_HZ = 35

    def __init__(self,
                 filter_order=DEFAULT_ORDER, cutoff_freq_hz=DEFAULT_CUTOFF_HZ,
                 zero_phase=False):
        """
        Initializes the lowpass filter transform.
        :param filter_order: Order of the FIR filter that will be created.
        :param cutoff_freq_hz: Cutoff frequency in Hz.
        :param zero_phase: Whether to use zero-phase filtering (i.e.
        filtfilt(), which means the filter will be applied twice so slower).
        Even if set to False, the filter delay will be removed and the end
        of the signal will be padded with zeros.
        """
        self.filter_order = filter_order
        self.cutoff_freq_hz = cutoff_freq_hz
        self.zero_phase = zero_phase

    def __call__(self, record):
        """
        Applies the transformation to the given record (each channel
        individually).

        :param record: A wfdb record to transform.
        :type record: wfdb.Record
        :return: The same WFDB record, with filtered data.
        """
        fs = record.fs
        signal = record.p_signal
        filtered_signal = self.filter_signal(signal, fs)
        record.p_signal = filtered_signal

        return record

    def filter_signal(self, signal, fs):
        # Calculate Nyquist frequency and cutoff frequency in normalized units
        f_nyq = fs / 2
        f_cutoff = self.cutoff_freq_hz / f_nyq

        # Create the filter coefficients
        fir_order = self.filter_order + 1
        fir_b = scipysignal.firwin(fir_order, f_cutoff)
        fir_a = [1]
        axis = 0  # filter should operate along columns

        # Apply filter
        if not self.zero_phase:
            filtered_signal = scipysignal.lfilter(fir_b, fir_a, signal,
                                                  axis=axis)
            # Remove the delay
            fir_delay = int(fir_order / 2)
            filtered_signal = np.roll(filtered_signal, -fir_delay)
            filtered_signal[-fir_delay:0, :] = 0
        else:
            filtered_signal = scipysignal.filtfilt(fir_b, fir_a, signal,
                                                   axis=axis)

        return filtered_signal
