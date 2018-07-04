from ecgbc.dataset import WFDB_HEADER_EXT

import torch.utils.data as data
import wfdb

from pathlib import Path


class WFDBDataset(data.Dataset):
    def __init__(self, root_path, transform=None):
        self.transform = transform

        # Generate record paths. A PhysioNet record has two or more files:
        # one header (.hea) file and one or more data (.dat) or annotation
        # files (.atr, .qrs, .ecg, ...)
        self.rec_paths = Path(root_path).glob(f'**/*{WFDB_HEADER_EXT}')
        self.rec_paths = list(str(rec).split(WFDB_HEADER_EXT)[0]
                              for rec in self.rec_paths)

    def __getitem__(self, item):
        rec_name = self.rec_paths[item]

        # Read raw data (signals) as ndarray and metadata (fields) as a dict
        signals, fields = wfdb.rdsamp(rec_name)

        # Add record name to metadata
        fields['rec_name'] = rec_name

        sample = {'signals': signals, 'fields': fields}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.rec_paths)

