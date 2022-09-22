"""Script to convert the Kitti Raw Sync dataset (& additions) to LMDB.

LMDBs (http://www.lmdb.tech/doc/) should provide faster loading & less load on the filesytem.

NOTE: This process takes quite a while!
Results are cached (i.e. LMDBs aren't recomputed unless forced) so the script can be interrupted and restarted.
"""
import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from src.devkits import PATHS, kitti_raw as kr
from src.external_libs import write_array_database, write_image_database, write_label_database


# DIRECTORY PARSING
# -----------------------------------------------------------------------------
def process_dataset(src_dir: Path,
                    dst_dir: Path,
                    use_hints: bool = True,
                    use_benchmark: bool = True,
                    overwrite: bool = False) -> None:
    """Process the entire Kitti Raw Sync datsets."""
    HINTS_DIR, BENCHMARK_DIR = 'depth_hints', 'depth_benchmark'

    # DEPTH SPLITS
    if not (path := dst_dir/'splits').is_dir():
        shutil.copytree(src_dir/'splits', path)

    # MAIN DATASET
    for seq in kr.SEQS:
        src_path = src_dir/seq
        dst_path = dst_dir/seq

        export_calibration(src_path, dst_path, overwrite)
        process_sequence(src_path, dst_path, overwrite)

    # DEPTH HINTS
    if use_hints:
        src_hints, dst_hints = src_dir/HINTS_DIR, dst_dir/HINTS_DIR
        for src_scene in sorted(src_hints.iterdir()):
            dst_scene = dst_hints/src_scene.name

            process_sequence(src_scene, dst_scene, overwrite)

    # DEPTH BENCHMARK
    if use_benchmark:
        src_benchmark, dst_benchmark = src_dir/BENCHMARK_DIR, dst_dir/BENCHMARK_DIR
        for src_scene in sorted(src_benchmark.iterdir()):
            dst_scene = dst_benchmark/src_scene.name

            process_sequence(src_scene, dst_scene, overwrite)


def process_sequence(src_dir: Path, dst_dir: Path, overwrite: bool = False) -> None:
    """Process a full Kitti Raw sequence: e.g. kitti_raw_sync/2011_09_26."""
    print(f'-> Processing sequence "{src_dir}"')
    for src_path in sorted(src_dir.iterdir()):
        if src_path.is_file(): continue
        dst_path = dst_dir/src_path.name

        process_drive(src_path, dst_path, overwrite)


def process_drive(src_dir: Path, dst_dir: Path, overwrite: bool = False) -> None:
    """Process a full Kitti Raw sequence: e.g. kitti_raw_sync/2011_09_26/2011_09_26_drive_0005."""
    print(f'\t-> Processing drive "{src_dir}"')
    for src_path in sorted(src_dir.iterdir()):
        dst_path = dst_dir/src_path.name

        process_dir(src_path, dst_path, overwrite)


def process_dir(src_dir: Path, dst_dir: Path, overwrite: bool = False) -> None:
    """Processes a data directory within a given drive.

    Cases:
        - Base dataset: images_00, images_01, velodyne_points, oxts (/data & /timestamps for each)
        - Depth hints: images_02, images_03
        - Depth benchmark: groundtruth/image_02, groundtruth/image_03
    """
    print(f'\t\t-> Processing dir "{src_dir}"')

    if 'depth_hints' in str(src_dir):
        if not overwrite and dst_dir.is_dir():
            print(f'\t\t-> Skipping dir "{dst_dir}"')
            return

        export_hints(src_dir, dst_dir)

    elif 'depth_benchmark' in str(src_dir):
        for src_path in sorted((src_dir/'groundtruth').iterdir()):
            dst_path = dst_dir/'groundtruth'/src_path.name

            if not overwrite and dst_path.is_dir():
                print(f'\t\t-> Skipping dir "{dst_path}"')
                continue

            export_images(src_path, dst_path)

    else:
        for src_path in sorted(src_dir.iterdir()):
            dst_path = dst_dir/src_path.name

            if src_path.is_file():
                if not dst_path.is_file():
                    shutil.copy(src_path, dst_path)
            else:
                assert src_path.stem == 'data'
                file = next(src_path.iterdir(), None)
                if file is None:
                    dst_path.mkdir(exist_ok=True, parents=True)
                    print(f'\t\t-> Skipping empty dir "{dst_path}"')
                    continue

                ext = file.suffix
                if not overwrite and dst_path.is_dir():
                    print(f'\t\t-> Skipping dir "{dst_path}"')
                    continue

                if ext == '.png':
                    export_images(src_path, dst_path)
                elif ext == '.bin':
                    export_velodyne(src_path, dst_path)
                elif ext == '.txt':
                    export_oxts(src_path, dst_path)
# -----------------------------------------------------------------------------


# LMDB CREATION
# -----------------------------------------------------------------------------
def export_calibration(src_seq: Path, dst_seq: Path, overwrite: bool = False) -> None:
    """Exports sequence calibration information as a LabelDatabase of arrays."""
    dst_dir = dst_seq/'calibration'
    if not overwrite and dst_dir.is_dir():
        print(f'\t-> Skipping calib "{dst_dir}"')
        return
    else:
        print(f'\t-> Processing calib "{dst_dir}"')

    cam2cam, imu2velo, velo2cam = kr.load_calib(src_seq.stem)
    data = {
        'cam2cam': cam2cam,
        'imu2velo': imu2velo,
        'velo2cam': velo2cam,
    }
    data = {f'{k1}/{k2}': v2 for k1, v1 in data.items() for k2, v2 in v1.items()}
    write_label_database(data, dst_dir)


def export_images(src_dir: Path, dst_dir: Path) -> None:
    """Export images as an ImageDatabase."""
    image_paths = {file.stem: file for file in sorted(src_dir.iterdir())}
    write_image_database(image_paths, dst_dir)


def export_oxts(src_dir: Path, dst_dir: Path) -> None:
    """Export OXTS dicts as a LabelDatabase."""
    data = {file.stem: kr.load_oxts(file) for file in sorted(src_dir.iterdir())}
    write_label_database(data, dst_dir)


def export_velodyne(src_dir: Path, dst_dir: Path) -> None:
    """Export Velodyne points as a LabelDatabase of arrays."""
    data = {file.stem: kr.load_velo(file) for file in sorted(src_dir.iterdir())}
    write_label_database(data, dst_dir)


def export_hints(src_dir: Path, dst_dir: Path) -> None:
    """Export depth hints as a LabelDatabase of arrays."""
    data = {file.stem: np.load(file) for file in sorted(src_dir.iterdir())}
    write_array_database(data, dst_dir)
# -----------------------------------------------------------------------------


if __name__ == '__main__':
    parser = ArgumentParser(description='Script to convert the Kitti Raw Sync dataset (& additions) to LMDB.')
    parser.add_argument('--use-hints', default=1, type=int, help='If 1, export precomputed depth hints.')
    parser.add_argument('--use-benchmark', default=1, type=int, help='If 1, export ground-truth depths from Kitti Benchmark.')
    parser.add_argument('--overwrite', default=0, type=int, help='If 1, overwrite existing LMDBs.')
    args = parser.parse_args()

    process_dataset(
        PATHS['kitti_raw'], PATHS['kitti_raw_lmdb'],
        use_hints=args.use_hints, use_benchmark=args.use_benchmark,
        overwrite=args.overwrite
    )
