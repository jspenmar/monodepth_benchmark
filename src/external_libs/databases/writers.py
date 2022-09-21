"""Modified by @jspenmar.
Merges all `write_*` scripts from the original repo.
"""
import pickle
import shutil
from pathlib import Path
from time import time

import lmdb
from tqdm import tqdm

__all__ = ['write_image_database', 'write_label_database', 'write_array_database']


def write_image_database(d: dict, database: Path):
    # Remove any existing database.
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists():
        shutil.rmtree(database)

    # For condor usage, we create a local database on the disk.
    tmp_dir = Path('/tmp') / f'TEMP_{time()}'
    tmp_dir.mkdir(parents=True)

    tmp_database = tmp_dir / f'{database.name}'

    # Create the database.
    with lmdb.open(path=f'{tmp_database}', map_size=2 ** 40) as env:

        # Add the protocol to the database.
        with env.begin(write=True) as txn:
            key = 'protocol'.encode('ascii')
            value = pickle.dumps(pickle.DEFAULT_PROTOCOL)
            txn.put(key=key, value=value, dupdata=False)

        # Add the keys to the database.
        with env.begin(write=True) as txn:
            key = pickle.dumps('keys')
            value = pickle.dumps(sorted(d.keys()))
            txn.put(key=key, value=value, dupdata=False)

        # Add the images to the database.
        for key, value in tqdm(sorted(d.items())):
            with env.begin(write=True) as txn:
                with value.open('rb') as file:
                    key = pickle.dumps(key)
                    txn.put(key=key, value=file.read(), dupdata=False)

    # Move the database to its destination.
    shutil.move(f'{tmp_database}', database)

    # Remove the temporary directories.
    shutil.rmtree(tmp_dir)


def write_label_database(d: dict, database: Path):
    # Remove any existing database.
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists():
        shutil.rmtree(database)

    # For condor usage, we create a local database on the disk.
    tmp_dir = Path('/tmp') / f'TEMP_{time()}'
    tmp_dir.mkdir(parents=True)

    tmp_database = tmp_dir / f'{database.name}'

    # Create the database.
    with lmdb.open(path=f'{tmp_database}', map_size=2 ** 40) as env:

        # Add the protocol to the database.
        with env.begin(write=True) as txn:
            key = 'protocol'.encode('ascii')
            value = pickle.dumps(pickle.DEFAULT_PROTOCOL)
            txn.put(key=key, value=value, dupdata=False)

        # Add the keys to the database.
        with env.begin(write=True) as txn:
            key = pickle.dumps('keys')
            value = pickle.dumps(sorted(d.keys()))
            txn.put(key=key, value=value, dupdata=False)

        # Add the values to the database.
        with env.begin(write=True) as txn:
            for key, value in tqdm(sorted(d.items())):
                key = pickle.dumps(key)
                value = pickle.dumps(value)
                txn.put(key=key, value=value, dupdata=False)

    # Move the database to its destination.
    shutil.move(f'{tmp_database}', database)

    # Remove the temporary directories.
    shutil.rmtree(tmp_dir)


def write_array_database(d: dict, database: Path):
    # Remove any existing database.
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists():
        shutil.rmtree(database)

    # For condor usage, we create a local database on the disk.
    tmp_dir = Path('/tmp') / f'TEMP_{time()}'
    tmp_dir.mkdir(parents=True)

    tmp_database = tmp_dir / f'{database.name}'

    # Create the database.
    with lmdb.open(path=f'{tmp_database}', map_size=2 ** 40) as env:

        # Add the protocol to the database.
        with env.begin(write=True) as txn:
            key = 'protocol'.encode('ascii')
            value = pickle.dumps(pickle.DEFAULT_PROTOCOL)
            txn.put(key=key, value=value, dupdata=False)

        # Add the keys to the database.
        with env.begin(write=True) as txn:
            key = pickle.dumps('keys')
            value = pickle.dumps(sorted(d.keys()))
            txn.put(key=key, value=value, dupdata=False)

        # Extract the shape and dtype of the values.
        value = next(iter(d.values()))
        shape = value.shape
        dtype = value.dtype

        # Add the shape to the database.
        with env.begin(write=True) as txn:
            key = pickle.dumps('shape')
            value = pickle.dumps(shape)
            txn.put(key=key, value=value, dupdata=False)

        # Add the dtype to the database.
        with env.begin(write=True) as txn:
            key = pickle.dumps('dtype')
            value = pickle.dumps(dtype)
            txn.put(key=key, value=value, dupdata=False)

        # Add the values to the database.
        with env.begin(write=True) as txn:
            for key, value in tqdm(sorted(d.items())):
                key = pickle.dumps(key)
                value = pickle.dumps(value)
                txn.put(key=key, value=value, dupdata=False)

    # Move the database to its destination.
    shutil.move(f'{tmp_database}', database)

    # Remove the temporary directories.
    shutil.rmtree(tmp_dir)
