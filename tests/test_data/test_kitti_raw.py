
import src.devkits.kitti_raw as kr
from src.utils import io

SPLITS = ['eigen_zhou', 'benchmark', 'eigen', 'eigen_benchmark']
MODES = ['train', 'val', 'test']


def _get_items(split, mode):
    file = kr.get_split_file(split, mode)
    if split == 'benchmark' and mode == 'test': return []

    side2cam = {'l': 'image_02', 'r': 'image_03'}
    lines = [line.split() for line in io.readlines(file)]
    items = [{
        'seq': line[0],
        'cam': side2cam[line[2]],
        'stem': int(line[1]),
    } for line in lines]

    return items


class TestKitti:
    def test_image_files(self):
        splits, seqs = [], []
        for s in SPLITS:
            for m in MODES:
                for i in _get_items(s, m):
                    f = kr.get_image_file(i['seq'], i['cam'], i['stem'])
                    if not f.is_file():
                        seqs.append(i['seq'])
                        splits.append(f'{s} {m}')

        assert not seqs, f"Missing image files in sequences. {set(splits)} {set(seqs)}"

    def test_velo_files(self):
        splits, seqs = [], []
        for s in SPLITS:
            for m in MODES:
                for i in _get_items(s, m):
                    f = kr.get_velodyne_file(i['seq'], i['stem'])
                    if not f.is_file():
                        seqs.append(i['seq'])
                        splits.append(f'{s} {m}')

        assert not seqs, f"Missing velodyne files in sequences. {set(splits)} {set(seqs)}"

    def test_hints_files(self):
        splits, seqs = [], []
        for s in SPLITS:
            for m in ['train', 'val']:
                for i in _get_items(s, m):
                    f = kr.get_hint_file(i['seq'], i['cam'], i['stem'])
                    if not f.is_file():
                        seqs.append(i['seq'])
                        splits.append(f'{s} {m}')

        assert not seqs, f"Missing depth hints files in sequences. {set(splits)} {set(seqs)}"

    def test_benchmark_files(self):
        splits, seqs = [], []
        for s in ['benchmark', 'eigen_benchmark']:
            for m in MODES:
                for i in _get_items(s, m):
                    f = kr.get_depth_file(i['seq'], i['cam'], i['stem'])
                    if not f.is_file():
                        seqs.append(i['seq'])
                        splits.append(f'{s} {m}')

        assert not seqs, f"Missing benchmark files in sequences. {set(splits)} {set(seqs)}"
