import re
import warnings

from ecgbc.dataset import WFDB_HEADER_EXT, ECG_CHANNEL_PATTERN

import torch.utils.data as data
import wfdb

from pathlib import Path


class WFDBDataset(data.Dataset):
    def __init__(self, root_path,
                 transform=None,
                 channel_pattern=ECG_CHANNEL_PATTERN,
                 first_channel_only=True):
        """
        Defines a dataset of WFDB record. The dataset is defined given a
        root directory, and will contain all WFDB record found recursively
        in that directory and it's subdirectories.

        By default only the first found ECG channel will be loaded from each
        record.

        The dataset returns objects of type :class:`wfdb.io.record.Record`.

        :param root_path: The path of the directory to search for records in.
        :param transform: A transformation to apply.
        :param channel_pattern: The pattern to identify channels to read.
        :param first_channel_only: Whether to read only the first or all
        channels that match the pattern.
        """

        self.root_path = root_path
        self.transform = transform
        self.channel_pattern = re.compile(channel_pattern, re.IGNORECASE)
        self.first_channel_only = first_channel_only

        # Generate record paths. A PhysioNet record has two or more files:
        # one header (.hea) file and one or more data (.dat) or annotation
        # files (.atr, .qrs, .ecg, ...)
        self.rec_paths = Path(root_path).glob(f'**/*{WFDB_HEADER_EXT}')
        self.rec_paths = list(str(rec).split(WFDB_HEADER_EXT)[0]
                              for rec in self.rec_paths)

    def __getitem__(self, item):
        rec_path = self.rec_paths[item]

        # Get indices of channels to read from the record based on the pattern
        channels = self._get_channels_to_read(rec_path)

        if not channels:
            warnings.warn(f'No channels in the record {rec_path} have '
                          f'channels which match the given pattern.')
            return None

        # Read raw data
        record = wfdb.rdrecord(rec_path, channels=channels, physical=True)

        # Add record's path to it's metadata
        record.record_path = rec_path

        if self.transform is not None:
            record = self.transform(record)

        return record

    def __len__(self):
        return len(self.rec_paths)

    def _get_channels_to_read(self, rec_name):
        header = wfdb.rdheader(rec_name)

        matching_channels = [
            chan_idx
            for chan_idx, chan_name in enumerate(header.sig_name)
            if self.channel_pattern.search(chan_name) is not None
        ]

        if self.first_channel_only and len(matching_channels) > 0:
            return matching_channels[0:1]
        else:
            return matching_channels


