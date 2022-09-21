
import src.devkits.syns_patches as syp

MODES = ['all', 'val', 'test']


def _get_items(mode):
    return syp.load_split(mode)[1]


class TestKitti:
    def test_image_files(self):
        modes, scenes = [], []
        for m in MODES:
            for i in _get_items(m):
                f = syp.get_image_file(*i)
                if not f.is_file():
                    scenes.append(i[0])
                    modes.append(f'{m}')

        assert not scenes, f"Missing image files in sequences. {set(modes)} {set(scenes)}"

    def test_depth_files(self):
        modes, scenes = [], []
        for m in {'val'}:
            for i in _get_items(m):
                f = syp.get_depth_file(*i)
                if not f.is_file():
                    scenes.append(i[0])
                    modes.append(f'{m}')

        assert not scenes, f"Missing depth files in sequences. {set(modes)} {set(scenes)}"

    def test_edges_files(self):
        modes, scenes = [], []
        for m in {'val'}:
            for i in _get_items(m):
                f = syp.get_edges_file(i[0], 'edges', i[1])
                if not f.is_file():
                    scenes.append(i[0])
                    modes.append(f'{m}')

        assert not scenes, f"Missing edges files in sequences. {set(modes)} {set(scenes)}"
