"""Script to precompute the depth hints used for proxy supervision.
Based on https://github.com/nianticlabs/depth-hints/blob/master/precompute_depth_hints.py

Hints are computed by using multiple SGBM estimates using different hyperparameters.
The final estimate is obtained by fusing these estimates based on the minimum reconstruction loss (from Monodepth2).

NOTE: This process takes quite a while! It's generally faster to use multiprocessing without the GPU.
Results are cached (i.e. hints aren't recomputed unless forced) so the script can be interrupted and restarted.
"""
from argparse import ArgumentParser
from multiprocessing import Pool
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from src.datasets import KittiRawDataset
from src.devkits import PATHS
from src.losses import PhotoError
from src.tools import ViewSynth, ops, to_inv
from src.typing import BatchData


def generate_matchers(block_sizes: Sequence[int] = (1, 2, 3),
                      disps: Sequence[int] = (64, 96, 128, 160)) -> Sequence[cv2.StereoSGBM]:
    """Instantiate stereo matchers with different hyperparameters to build fused depth hints."""
    matchers = []
    w = 3  # SAD window size
    for b in block_sizes:
        for d in disps:
            matchers.append(cv2.StereoSGBM_create(
                preFilterCap=63, P1=w*w*4, P2=w*w*32, minDisparity=0, numDisparities=d,
                uniquenessRatio=10, speckleWindowSize=100, speckleRange=16, blockSize=b,
            ))
    return matchers


def compute_depth(img: NDArray, img_st: NDArray, K: NDArray, T: NDArray, matchers: Sequence[cv2.StereoSGBM]) -> NDArray:
    """Compute the predicted depth for each of the given stereo matchers.

    NOTE: Since SGBM always computes the left-right disparity, we horizontally flip the images to perform right-left.

    :param img: (ndarray) (h, w, 3) Image we want to compute depth for.
    :param img_st: (ndarray) (h, w, 3) Stereo pair to compute disparity.
    :param K: (ndarray) (4, 4) Camera intrinsic parameters.
    :param T: (ndarray) (4, 4) Transform to stereo pair.
    :param matchers: (list) (n,) SGBM matchers to use.
    :return: (ndarray, ndarray) (n, h, w, 1)
    """
    img = (255*img).astype(np.uint8)
    img_st = (255*img_st).astype(np.uint8)

    is_invert = T[0, 3] > 0  # Matching is done from left to right images.
    if is_invert: img, img_st = img[:, ::-1], img_st[:, ::-1]

    disps = []
    for m in matchers:
        disp = m.compute(img, img_st) / 16  # Convert to pixel disparity.
        if is_invert: disp = disp[:, ::-1]
        disps.append(disp[..., None])

    disps = np.stack(disps).astype(np.float32)
    disps *= (disps > 0).astype(float)

    # NOTE: These depths still need to be scaled by a factor of 5.4 to align them to the ground-truth depths.
    # This is the same procedure used when evaluating models trained with the stereo pair.
    depths = K[0, 0] * abs(T[0, 3]) * to_inv(disps)
    return depths


def export_hint(item: BatchData, verbose: bool = False) -> None:
    """Compute a fused depth hint from a dataset item."""
    x, y, m = item

    save_file = save_dir / f'{m["stem"]}.npy'
    save_file.parent.mkdir(exist_ok=True, parents=True)
    if save_file.is_file() and not args.overwrite:
        if verbose: print(f'-> Skipping file {save_file}... Set `--overwrite 1 to overwrite.')
        return

    if verbose: print(f'-> Starting file {save_file}...')
    depths = compute_depth(y['imgs'], y['supp_imgs'][0], K=y['K'], T=y['T_stereo'], matchers=matchers)

    y, depths = ops.to_torch((y, depths), device=device)
    imgs = ops.expand_dim(y['imgs'], num=b, insert=True)

    supp_imgs_warp, _, _ = synth.forward(
        input=y['supp_imgs'].expand(b, -1, -1, -1), depth=depths,
        T=ops.expand_dim(y['T_stereo'], num=b, insert=True), K=ops.expand_dim(y['K'], num=b, insert=True),
    )

    # Min reconstruction error to determine "best" depth hint for each pixel.
    err = crit(supp_imgs_warp, imgs)
    idxs = err.argmin(dim=0, keepdim=True)
    depth = depths.gather(index=idxs, dim=0)
    depth = ops.to_numpy(depth).squeeze()

    np.save(save_file, depth)


if __name__ == '__main__':
    # NOTE: This script doesn't run in a `main` function to keep variables global (`matchers`, `criterion`, `synth`...)
    # for ease of use with multiprocessing!
    SPLITS = ('eigen', 'eigen_zhou', 'eigen_benchmark', 'benchmark')
    parser = ArgumentParser(description='Script to precompute the depth hints used for proxy supervision.')
    parser.add_argument('--split', required=True, choices=SPLITS, help='Kitti depth split to compute hints for.')
    parser.add_argument('--mode', default='train', choices={'train', 'val', 'test'}, help='Training mode to load.')
    parser.add_argument('--item', default=None, type=int, help='Run only a specific item in the dataset.')
    parser.add_argument('--n-proc', default=20, type=int, help='Number of parallel processes to run. 0 to disable.')
    parser.add_argument('--overwrite', default=0, type=int, help='If 1, overwrite existing depth hints.')
    args = parser.parse_args()

    save_dir = PATHS['kitti_raw']/'depth_hints'
    save_dir.mkdir(exist_ok=True)

    matchers = generate_matchers()

    b = len(matchers)
    w, h = size = (1024, 320)

    device = ops.get_device('cpu' if args.n_proc > 1 else 'cuda')
    crit = PhotoError(weight_ssim=0.85).to(device)
    synth = ViewSynth(shape=(h, w)).to(device)

    ds = KittiRawDataset(split=args.split, mode=args.mode, size=size, supp_idxs=0, use_depth=True, as_torch=False)

    if args.item is not None:
        export_hint(ds[args.item], verbose=True)
        exit()

    if args.n_proc:
        with Pool(args.n_proc) as p:
            for _ in tqdm(p.imap_unordered(export_hint, ds), total=len(ds)): pass
    else:
        [export_hint(item) for item in tqdm(ds)]
