import argparse
import os

import ecgbc.dataset.transforms
import ecgbc.dataset.wfdb_dataset
import torchvision.transforms

import ecgbc.dataset.wfdb_single_beat


def parse_cli():
    def is_dir(dirname):
        if not os.path.isdir(dirname):
            raise argparse.ArgumentTypeError(f'{dirname} is not a directory')
        else:
            return dirname

    parser = argparse.ArgumentParser(
        description='ECG single beat classification'
    )
    subparsers = parser.add_subparsers(help='Sub-command help')

    # Generate dataset
    subparser_generate = subparsers.add_parser(
        'generate-dataset', help='Generate a single-beat dataset')
    subparser_generate.set_defaults(subcmd_fn=generate_dataset)
    subparser_generate.add_argument('--in', '-i', type=is_dir, dest='in_dir',
                                    help='Input dir', required=True)
    subparser_generate.add_argument('--out', '-o', type=str, dest='out_dir',
                                    help='Output dir', required=True)
    subparser_generate.add_argument('--transform-lpf', action='store_true',
                                    help='Apply lowpass-filter')
    subparser_generate.add_argument('--transform-sma', action='store_true',
                                    help='Subtract moving average')
    subparser_generate.add_argument('--filter-rri', action='store_true',
                                    help='Filter out non physiological RR'
                                         'intervals')
    subparser_generate.add_argument('--ann-ext', '-a', type=str, default='atr',
                                    help='Input annotation extension')
    subparser_generate.add_argument('--rec-pattern', type=str, default=None,
                                    help='Pattern for matching record names')
    subparser_generate.add_argument('--aami', '-A', action='store_true',
                                    default=None,
                                    help='Create AAMI compatible class labels')

    return parser.parse_args()


def generate_dataset(in_dir, out_dir, ann_ext, **kwargs):
    """
    Generates a single-beat dataset based on a folder containing WFDB records.
    """
    transforms = []
    if kwargs['transform_lpf']:
        transforms.append(ecgbc.dataset.transforms.LowPassFilterWFDB())
    if kwargs['transform_sma']:
        transforms.append(ecgbc.dataset.transforms.SubtractMovingAverageWFDB())

    wfdb_dataset = ecgbc.dataset.wfdb_dataset.WFDBDataset(
        root_path=in_dir,
        recname_pattern=kwargs['rec_pattern'],
        transform=torchvision.transforms.Compose(transforms))

    generator = ecgbc.dataset.wfdb_single_beat.Generator(
        wfdb_dataset, in_ann_ext=ann_ext, out_ann_ext=f'ecg{ann_ext}',
        calculate_rr_features=True, filter_rri=kwargs['filter_rri'],
        aami_compatible=kwargs['aami']
    )

    generator.write(out_dir)

    dataset = ecgbc.dataset.wfdb_single_beat.SingleBeatDataset(out_dir)
    print(dataset)


if __name__ == '__main__':
    parsed_args = parse_cli()
    parsed_args.subcmd_fn(**vars(parsed_args))
