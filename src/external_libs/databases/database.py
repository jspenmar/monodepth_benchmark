"""Modified by @jspenmar.
Add __all__ & based on earlier code release.
"""
import io
import pickle
from os import PathLike

import lmdb
import numpy as np
import torch
from PIL import Image, ImageFile

__all__ = ['Database', 'ImageDatabase', 'LabelDatabase', 'MaskDatabase', 'ArrayDatabase', 'TensorDatabase']


ImageFile.LOAD_TRUNCATED_IMAGES = True


class Database:
    _database = None
    _protocol = None
    _length = None

    def __init__(self, path: PathLike, readahead: bool = True, pre_open: bool = False):
        """Base class for LMDB-backed _databases.

        :param path: (PathLike) Path to the database.
        :param readahead: (bool) If `True`, enables the filesystem readahead mechanism.
        :param pre_open: (bool) If `True`, the first iterations will be faster, but it will raise error when doing multi-gpu training.
            If `False`, the database will open when you will retrieve the first item.
        """
        self.path = str(path)
        self.readahead = readahead
        self.pre_open = pre_open

        self._has_fetched_an_item = False

    @property
    def database(self):
        if self._database is None:
            self._database = lmdb.open(
                path=self.path,
                readonly=True,
                readahead=self.readahead,
                max_spare_txns=256,
                lock=False,
            )
        return self._database

    @database.deleter
    def database(self):
        if self._database is not None:
            self._database.close()
            self._database = None

    @property
    def protocol(self):
        """Read the pickle protocol contained in the database.

        :return: The set of available keys.
        """
        if self._protocol is None:
            self._protocol = self._get(
                item='protocol',
                convert_key=lambda key: key.encode('ascii'),
                convert_value=lambda value: pickle.loads(value),
            )
        return self._protocol

    @property
    def keys(self):
        """Read the keys contained in the database.

        :return: The set of available keys.
        """
        protocol = self.protocol
        keys = self._get(
            item='keys',
            convert_key=lambda key: pickle.dumps(key, protocol=protocol),
            convert_value=lambda value: pickle.loads(value),
        )
        return keys

    def __len__(self):
        """Returns the number of keys available in the database.

        :return: The number of keys.
        """
        if self._length is None:
            self._length = len(self.keys)
        return self._length

    def __getitem__(self, item):
        """Retrieves an item or a list of items from the database.

        :param item: A key or a list of keys.
        :return: A value or a list of values.
        """
        self._has_fetched_an_item = True
        if not isinstance(item, list):
            item = self._get(item, self._convert_key, self._convert_value)
        else:
            item = self._gets(item, self._convert_keys, self._convert_values)
        return item

    def __contains__(self, item):
        """Check if a given key is in the database."""
        return item in self.keys

    def index(self, index):
        """Retrieves an item or a list of items from the database from an integer index.

        :param index: An index or a list of indexes.
        :return: A value or a list of values.
        """
        key = self.keys[index]
        return key, self[key]

    def _get(self, item, convert_key, convert_value):
        """Instantiates a transaction and its associated cursor to fetch an item.

        :param item: A key.
        :param convert_key:
        :param convert_value:
        :return:
        """
        with self.database.begin() as txn:
            with txn.cursor() as cursor:
                item = self._fetch(cursor, item, convert_key, convert_value)
        self._keep_database()
        return item

    def _gets(self, items, convert_keys, convert_values):
        """Instantiates a transaction and its associated cursor to fetch a list of items.

        :param items: A list of keys.
        :param convert_keys:
        :param convert_values:
        :return:
        """
        with self.database.begin() as txn:
            with txn.cursor() as cursor:
                items = self._fetchs(cursor, items, convert_keys, convert_values)
        self._keep_database()
        return items

    def _fetch(self, cursor, key, convert_key, convert_value):
        """Retrieve a value given a key.

        :param cursor:
        :param key: A key.
        :param convert_key:
        :param convert_value:
        :return: A value.
        """
        key = convert_key(key=key)
        value = cursor.get(key=key)
        value = convert_value(value=value)
        return value

    def _fetchs(self, cursor, keys, convert_keys, convert_values):
        """Retrieve a list of values given a list of keys.

        :param cursor:
        :param keys: A list of keys.
        :param convert_keys:
        :param convert_values:
        :return: A list of values.
        """
        keys = convert_keys(keys=keys)
        _, values = list(zip(*cursor.getmulti(keys)))
        values = convert_values(values=values)
        return values

    def _convert_key(self, key):
        """Converts a key into a byte key.

        :param key: A key.
        :return: A byte key.
        """
        return pickle.dumps(key, protocol=self.protocol)

    def _convert_keys(self, keys):
        """Converts keys into byte keys.

        :param keys: A list of keys.
        :return: A list of byte keys.
        """
        return [self._convert_key(key=key) for key in keys]

    def _convert_value(self, value):
        """Converts a byte value back into a value.

        :param value: A byte value.
        :return: A value
        """
        return pickle.loads(value)

    def _convert_values(self, values):
        """Converts bytes values back into values.

        :param values: A list of byte values.
        :return: A list of values.
        """
        return [self._convert_value(value=value) for value in values]

    def _keep_database(self):
        """Checks if the database must be deleted."""
        if not self.pre_open and not self._has_fetched_an_item:
            del self.database

    def __iter__(self):
        """Provides an iterator over the keys when iterating over the database."""
        return iter(self.keys)

    def __del__(self):
        """Closes the database properly."""
        del self.database


class ImageDatabase(Database):
    def _convert_value(self, value):
        """Converts a byte image back into a PIL Image.

        :param value: A byte image.
        :return: A PIL Image image.
        """
        return Image.open(io.BytesIO(value))


class MaskDatabase(ImageDatabase):
    def _convert_value(self, value):
        """Converts a byte image back into a PIL Image.

        :param value: A byte image.
        :return: A PIL image.
        """
        return Image.open(io.BytesIO(value)).convert('1')


class LabelDatabase(Database):
    pass


class ArrayDatabase(Database):
    _dtype = None
    _shape = None

    @property
    def dtype(self):
        if self._dtype is None:
            protocol = self.protocol
            self._dtype = self._get(
                item='dtype',
                convert_key=lambda key: pickle.dumps(key, protocol=protocol),
                convert_value=lambda value: pickle.loads(value),
            )
        return self._dtype

    @property
    def shape(self):
        if self._shape is None:
            protocol = self.protocol
            self._shape = self._get(
                item='shape',
                convert_key=lambda key: pickle.dumps(key, protocol=protocol),
                convert_value=lambda value: pickle.loads(value),
            )
        return self._shape

    def _convert_value(self, value):
        return np.frombuffer(value, dtype=self.dtype).reshape(self.shape)

    def _convert_values(self, values):
        return np.frombuffer(b''.join(values), dtype=self.dtype).reshape((len(values),) + self.shape)


class TensorDatabase(ArrayDatabase):
    def _convert_value(self, value):
        return torch.from_numpy(super(TensorDatabase, self)._convert_value(value))

    def _convert_values(self, values):
        return torch.from_numpy(super(TensorDatabase, self)._convert_values(values))
