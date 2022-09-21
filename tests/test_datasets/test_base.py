from unittest import mock

import pytest
import torch
import matplotlib.pyplot as plt

from src.datasets import BaseDataset
from src.utils import MultiLevelTimer


class TmpData(BaseDataset):
    def __init__(self, n, **kwargs):
        self.n = range(n)
        super().__init__(**kwargs)

    def __len__(self):
        return len(self.n)

    def load(self, item, x, y, meta):
        x['item'] = self.n[item]
        return x, y, meta

    def augment(self, x, y, meta):
        x['item'] *= 100
        meta['augs'] = 'helloworld'
        return x, y, meta

    def show(self, x, y, meta, axs=None): ...


class TestBaseDataset:
    def test_base(self):
        """Test that we have the expected functions."""
        with pytest.raises(TypeError):
            _ = BaseDataset()

        assert hasattr(BaseDataset, '__repr__'), "Missing attribute from base dataset."
        assert hasattr(BaseDataset, '__len__'), "Missing attribute from base dataset."
        assert hasattr(BaseDataset, '__getitem__'), "Missing attribute from base dataset."
        assert hasattr(BaseDataset, 'load'), "Missing attribute from base dataset."
        assert hasattr(BaseDataset, 'collate_fn'), "Missing attribute from base dataset."
        assert hasattr(BaseDataset, 'augment'), "Missing attribute from base dataset."
        assert hasattr(BaseDataset, 'to_torch'), "Missing attribute from base dataset."
        assert hasattr(BaseDataset, 'create_axs'), "Missing attribute from base dataset."
        assert hasattr(BaseDataset, 'show'), "Missing attribute from base dataset."
        assert hasattr(BaseDataset, 'play'), "Missing attribute from base dataset."

        dataset = TmpData(10)
        assert hasattr(dataset, 'logger'), "Missing logger in dataset."
        assert dataset.logger.name == 'BaseDataset.TmpData', "Incorrect logger name."

    def test_timer(self):
        """Test that timing can be enabled/disabled."""
        dataset = TmpData(10, log_time=True)
        x, y, meta = dataset[0]
        assert isinstance(dataset.timer, MultiLevelTimer), "Incorrect timer class."
        assert 'data_timer' in meta, "Missing timing information in meta."

        dataset = TmpData(10, log_time=False)
        x, y, meta = dataset[0]
        assert not isinstance(dataset.timer, MultiLevelTimer), "Incorrect timer class."
        assert 'data_timer' not in meta, "Unexpected timing information in meta."

    def test_augment(self):
        """Test that augmentations can be enabled/disabled."""
        dataset = TmpData(10, use_aug=True)
        x, y, meta = dataset[5]
        assert x['item'] == 500, "Augmentation not correctly applied."
        assert 'augs' in meta, "Missing augmentations information in meta."
        assert meta['augs'] == 'helloworld', "Augmentation not correctly applied."

        dataset = TmpData(10, use_aug=False)
        x, y, meta = dataset[5]
        assert x['item'] == 5, "Unexpected augmentation applied."
        assert 'augs' not in meta, "Unexpected augmentation information in meta."

        x2, y2, meta2 = BaseDataset.augment(dataset, x, y, meta)
        assert x2 == x, "Incorrect default augmentation"
        assert y2 == y, "Incorrect default augmentation"
        assert meta2 == meta, "Incorrect default augmentation"

    def test_as_torch(self):
        """Test that conversion to torch can be enabled/disabled."""
        dataset = TmpData(10, as_torch=True)
        x, y, meta = dataset[0]
        assert isinstance(x['item'], torch.Tensor), "Incorrect conversion to torch."
        assert isinstance(meta['items'], str), "Unexpected meta conversion to torch."

        dataset = TmpData(10, as_torch=False)
        x, y, meta = dataset[0]
        assert isinstance(x['item'], int), "Unexpected conversion to torch."
        assert isinstance(meta['items'], str), "Unexpected meta conversion to torch."

    def test_retry(self):
        """Test that exception retrying can be enabled/disabled."""

        # Dummy dataset
        class TmpData2(TmpData):
            def load(self, item, x, y, meta):
                if item % 2 == 0: raise ValueError  # Fail on even items.
                return super().load(item, x, y, meta)

        with pytest.raises(ValueError):  # By default shouldn't retry
            _ = TmpData2(10)[2]

        class TmpData3(TmpData2, retry_exc=Exception): pass
        x, y, meta = TmpData3(10)[2]
        assert x['item'] != 2, "Error retrying all exceptions."

        class TmpData3(TmpData2, retry_exc=ValueError): pass
        x, y, meta = TmpData3(10)[2]
        assert x['item'] != 2, "Error retrying on a specific exception."

    @pytest.mark.skip(reason="Creates multiple windows on PyCharm")
    def test_play(self):
        """Test dataset playing iterates correctly and sets window sizes."""
        dataset = TmpData(5, as_torch=True)
        with pytest.raises(ValueError):  # Should work only with non-torch data.
            dataset.play()

        dataset = TmpData(5, as_torch=False)
        dataset.show = mock.Mock()  # Allow for call count
        dataset.play()
        assert dataset.show.call_count == 5, "Incorrect number of calls to show."
        plt.close()

    @pytest.mark.skip(reason="Images are same size on PyCharm")
    def test_fullscreen(self):
        TmpData(5, as_torch=False).play()
        size1 = plt.get_current_fig_manager().canvas.get_width_height()
        plt.close()

        TmpData(5, as_torch=False).play(fullscreen=True)
        size2 = plt.get_current_fig_manager().canvas.get_width_height()
        plt.close()
        assert size1 != size2, "Error setting figure to fullscreen."

    def test_dataset_collate(self):
        """Test that we can collate data by default correctly."""
        class TmpData2(TmpData, retry_exc=ValueError):
            def load(self, item, x, y, meta):
                if item % 2 == 0: raise ValueError  # Fail on even items.
                return super().load(item, x, y, meta)

            def augment(self, x, y, meta):
                return super().augment(x, y, meta) if torch.rand(1) < 0.5 else (x, y, meta)

        dataset = TmpData2(10, as_torch=True, use_aug=True)
        batch_size = 4
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, collate_fn=dataset.collate_fn)
        x, y, meta = next(iter(loader))

        assert isinstance(x['item'], torch.Tensor),  "Incorrect conversion to torch."
        assert x['item'].shape == (batch_size,),  "Incorrect item batch size."
        assert x['item'][0] != 0,  "Incorrect retry on error."

        assert 'items' in meta, "Missing items in meta."
        assert len(meta['items']) == batch_size, "Incorrect items batch size."
        assert isinstance(meta['items'][0], str), "Incorrect items type."

        assert 'data_timer' in meta, "Missing data_timer in meta."
        assert len(meta['data_timer']) == batch_size, "Incorrect data_timer batch size."
        assert isinstance(meta['data_timer'][0], MultiLevelTimer), "Incorrect items type."

        assert 'errors' in meta, "Missing errors in meta."
        assert len(meta['errors']) == batch_size, "Incorrect errors batch size."
        assert isinstance(meta['errors'][0], str), "Incorrect errors type."

        assert 'augs' in meta, "Missing augmentations in meta."
        assert len(meta['augs']) == batch_size, "Incorrect augmentations batch size."
        assert isinstance(meta['augs'][0], str), "Incorrect augmentations type."
