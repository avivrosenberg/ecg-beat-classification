import argparse
import json
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data.dataloader
import torchvision.transforms

import ecgbc.dataset.transforms
import ecgbc.dataset.wfdb_dataset
import ecgbc.dataset.wfdb_single_beat
import ecgbc.models
import ecgbc.autoencoder
import ecgbc.classifier

DEFAULT_NUM_EPOCHS = 10
DEFAULT_BATCH_SIZE = 128


def parse_cli():
    def is_dir(dirname):
        if not os.path.isdir(dirname):
            raise argparse.ArgumentTypeError(f'{dirname} is not a directory')
        else:
            return dirname

    def is_file(filename):
        if not os.path.isfile(filename):
            raise argparse.ArgumentTypeError(f'{filename} is not a file')
        else:
            return filename

    p = argparse.ArgumentParser(description='ECG single beat classification')
    sp = p.add_subparsers(help='Sub-command help')

    # Generate dataset
    sp_generate = sp.add_parser('generate-dataset',
                                help='Generate a single-beat dataset')
    sp_generate.set_defaults(subcmd_fn=generate_dataset)
    sp_generate.add_argument('--in', '-i', type=is_dir, dest='in_dir',
                             help='Input dir', required=True)
    sp_generate.add_argument('--out', '-o', type=str, dest='out_dir',
                             help='Output dir', required=True)
    sp_generate.add_argument('--transform-lpf', action='store_true',
                             help='Apply lowpass-filter')
    sp_generate.add_argument('--transform-sma', action='store_true',
                             help='Subtract moving average')
    sp_generate.add_argument('--filter-rri', action='store_true',
                             help='Filter out non physiological RR'
                                  'intervals')
    sp_generate.add_argument('--ann-ext', '-a', type=str, default='atr',
                             help='Input annotation extension')
    sp_generate.add_argument('--rec-pattern', type=str, default=None,
                             help='Pattern for matching record names')
    sp_generate.add_argument('--aami', '-A', action='store_true',
                             default=None,
                             help='Create AAMI compatible class labels')

    # Debug
    sp_train = sp.add_parser('debug', help='Debugging functions')
    sp_train.set_defaults(subcmd_fn=debug)
    sp_train.add_argument('--ds', '-d', type=is_dir, dest='dataset_dir',
                          help='Show dataset', required=True)

    # Training
    sp_train_dae = sp.add_parser('train-dae', help='Train DAE on ECG beats')
    sp_train_dae.set_defaults(subcmd_fn=train_dae)

    sp_train_cls = sp.add_parser('train-cls', help='Train ECG beat classifier')
    sp_train_cls.set_defaults(subcmd_fn=train_cls)

    for sp_train in (sp_train_dae, sp_train_cls):
        sp_train.add_argument('--ds-train', '-d', type=is_dir,
                                 help='Training dataset dir', required=True)
        sp_train.add_argument('--ds-test', '-D', type=is_dir,
                              help='Test dataset dir', required=True)
        sp_train.add_argument('--num-epochs', '-e', type=int,
                              default=DEFAULT_NUM_EPOCHS,
                              help='Number of epochs', required=False)
        sp_train.add_argument('--batch-size', '-b', type=int,
                              default=DEFAULT_BATCH_SIZE,
                              help='Training batch size', required=False)
        sp_train.add_argument('--load-model', '-L', type=is_file, default=None,
                              help='Load model state file', required=False)
        sp_train.add_argument('--save-model', '-S', type=str, default=None,
                              help='Save model state file', required=False)
        sp_train.add_argument('--save-losses', '-o', type=str, default=None,
                              help='Save training losses', required=False)

    return p.parse_args()


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


def debug(dataset_dir, **kwargs):
    dataset = ecgbc.dataset.wfdb_single_beat.SingleBeatDataset(
        dataset_dir, transform=ecgbc.dataset.transforms.Normalize1D()
    )
    print(dataset)

    batch_size = 32
    loader = torch.utils.data.dataloader.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    plt.ion()
    cmap = plt.get_cmap('Set1')
    fig, axes = plt.subplots(nrows=8, ncols=batch_size//8,
                             sharex='col', sharey='row')
    axes = np.reshape(axes, (-1,))
    fig.tight_layout()

    for batch_idx, (samples, labels) in enumerate(loader):
        fig.canvas.set_window_title(f'Batch {batch_idx}')
        for i, ax in enumerate(axes):
            y = samples[i, :-4].numpy()
            x = np.r_[1:len(y)+1]
            ax.clear()
            ax.plot(x, y)

            label_idx = labels[i].item()
            label_text = dataset.idx_to_class[label_idx]
            label_color = cmap.colors[label_idx]
            ax.text(0, 0, label_text, color=label_color, weight='bold')
        pdb.set_trace()


def train(trainer_class, ds_train, ds_test, num_epochs, batch_size,
          load_model=None, save_model=None, save_losses=None, **kwargs):

    # Create data sets and loaders
    data_tf = ecgbc.dataset.transforms.Normalize1D()
    subset = dict(Q=0,)

    ds_train = ecgbc.dataset.wfdb_single_beat.SingleBeatDataset(
        ds_train, transform=data_tf, subset=subset)
    ds_test = ecgbc.dataset.wfdb_single_beat.SingleBeatDataset(
        ds_test, transform=data_tf, subset=subset)
    dl_args = dict(batch_size=batch_size, shuffle=True, num_workers=2,
                   drop_last=False)
    dl_train = torch.utils.data.DataLoader(ds_train, **dl_args)
    dl_test = torch.utils.data.DataLoader(ds_test, **dl_args)

    feature_size = ds_train[0][0].shape[0]
    num_classes = len(ds_train.classes)
    print("Train", ds_train, end='\n\n')
    print("Test", ds_test)

    # Create trainer & fit model
    trainer = trainer_class(load_params_file=load_model,
                            feature_size=feature_size,
                            num_classes=num_classes,
                            hidden_layer_sizes=(100,),
                            weight_decay=0, learn_rate=0.5)

    result = trainer.fit(dl_train, dl_test, num_epochs, verbose=True)

    # Write outputs
    if save_model is not None:
        torch.save(trainer.model.state_dict(), f'{save_model}.pt')

    if save_losses is not None:
        with open(f'{save_losses}.json', mode='w', encoding='utf-8') as f:
            json.dump(result._asdict(), f, indent=2, sort_keys=True)


    # Plot
    epochs = np.arange(num_epochs)
    fig, ax = plt.subplots(2, 1, sharex='all')
    ax[0].plot(epochs, result.train_loss, epochs, result.test_loss)
    ax[0].legend(['train', 'test'])
    ax[0].set_title('Loss')
    ax[1].plot(epochs, result.train_acc, epochs, result.test_acc)
    ax[1].legend(['train', 'test'])
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epoch #')
    plt.show()


def train_dae(**kwargs):
    train(ecgbc.autoencoder.Trainer, **kwargs)


def train_cls(**kwargs):
    train(ecgbc.classifier.Trainer, **kwargs)


if __name__ == '__main__':
    parsed_args = parse_cli()
    parsed_args.subcmd_fn(**vars(parsed_args))

