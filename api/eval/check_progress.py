"""Simple script to monitor the progress of models in a target directory.
Can also be used to check epoch of the `best` model.
"""
import torch

from src import MODEL_ROOTS
from src.tools import ops


def main():
    device = ops.get_device('cpu')
    root = MODEL_ROOTS[-1]
    exp, ckpt_name = 'benchmark', 'last'
    files = sorted((root/exp).glob(f'**/{ckpt_name}.ckpt'))

    for f in files:
        n = str(f).replace(str(root), '')
        is_training = (f.parent/'training').is_file()
        is_finished = (f.parent/'finished').is_file()
        try:
            ckpt = torch.load(f, map_location=device)
            print(f'{n} - Epoch: {ckpt["epoch"]} - Training: {is_training} - Finished: {is_finished}')
        except EOFError:
            print(f'CORRUPTED! {f} - Training: {is_training} - Finished: {is_finished}')


if __name__ == '__main__':
    main()
