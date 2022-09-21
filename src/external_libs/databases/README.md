# LMDatabase
## A simple Python interface for [LMDB](https://lmdb.readthedocs.io/) databases.

---

From https://github.com/GuillaumeRochette/LMDatabase

---

### Summary
1. [Getting Started](#getting-started)
2. [Reading from a Database](#reading-from-a-database)
3. [How Does It Work](#how-does-it-work)
4. [Specific Databases](#specific-databases)
5. [Creating a PyTorch Dataset](#creating-a-pytorch-dataset)
6. [Writing Databases](#writing-databases)
7. [Going Further](#going-further)

---

### Getting Started

Install the following packages to your environment:
```shell
pip install lmdb Pillow
```

---

### Reading from a Database
The [```Database```](database.py#L32) class mimics Python's ```dict``` structure, with the exception that is read-only.

```python
from database import Database

database = Database(path=f"/path/to/database")

# To iterate through the keys of the database.
for key in database:
    value = database[key]

# To retrieve one value. 
key = database.keys[69]
value = database[key]

# To retrieve a several values at once. (Similar to numpy.array mechanics). 
keys = database.keys[69:420]
values = database[keys]
```

---

### How Does It Work
[LMDB](https://lmdb.readthedocs.io/) operates with binary data for both keys and values to maintain its extremely high performance and memory efficiency.

Look at the method [```_fetch()```](database.py#L135):
1. An `lmdb.Transaction` handle is instantiated.
2. An `lmdb.Cursor` object is created.
3. The key is encoded with ```key = fencode(key)```.
4. The binary value is retrieved with ```value = cursor.get(key)```.
5. The value is decoded with ```value = fdecode(value)```.

By default, [```Database```](database.py#L14) uses ```pickle``` to encode keys and decode values.

---

### Specific Databases 
Several sub-classes exist already, [```ImageDatabase```](database.py#L229), [```LabelDatabase```](database.py#L236), [```ArrayDatabase```](database.py#L236), [```TensorDatabase```](database.py#L236).
- [```ImageDatabase._fdecode()```](database.py#L230) converts a value directly to a ```PIL.Image```.
- [```ArrayDatabase._fdecode()```](database.py#L230) converts a value directly to a ```np.ndarray```.
- [```TensorDatabase._fdecode()```](database.py#L230) converts a value directly to a ```torch.Tensor```.

Example:
```python
from database import ImageDatabase

database = ImageDatabase(path=f"/path/to/image/database")

# To retrieve one image. 
key = database.keys[69]
value = database[key]  # <- This is a PIL.Image.

# To retrieve a several values at once. (Similar to numpy.array's mechanics). 
keys = database.keys[69:420]
values = database[keys]  # <- This is a list of PIL.Image.
```

#### Important!

If you have specific needs in terms of I/O, you only have to sub-class [```_fdecode()```](database.py#L135), its behaviour should mimic behaviour opening a regular file, excepted that this one is binary.

---

### Creating a PyTorch Dataset
Integrating [```Databases```](database.py#L14) with PyTorch looks like this.
```python
from typing import Union
from torch.utils.data import Dataset
from pathlib import Path
from database import ImageDatabase, LabelDatabase


class MyDataset(Dataset):
    def __init__(self, path: Union[str, Path], transform=None):
        if not isinstance(path, Path):
            path = Path(path)
            
        images = path / f"Images.lmdb"
        labels = path / f"Labels.lmdb"
        
        self.images = ImageDatabase(path=images)
        self.labels = LabelDatabase(path=labels)
        self.keys = self._keys()
        self.transform = transform
    
    def _keys(self):
        # We assume that the keys are the same for the images and the labels.
        # Feel free to do something else if you fancy it.
        keys = sorted(set(self.images.keys).intersection(self.labels.keys))
        return keys
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, item):
        key = self.keys[item]
        data = {
            "image": self.images[key],
            "label": self.labels[key]
        }
        if self.transform:
            data = self.transform(data)
        return data
```

---

### Writing Databases
- To write a database for images located in a directory tree, execute [write_image_database.py](write_image_database.py):
```shell
python write_image_database.py --src_images SRC_IMAGES
                               --extension EXTENSION 
                               --dst_database DST_DATABASE
```
- To write a database of labels stored in a JSON file, execute [write_label_database.py](write_label_database.py):
```shell
python write_label_database.py --src_labels SRC_LABELS 
                               --dst_database DST_DATABASE
```

---

### Going Further
- Feel free to tailor the scripts for writing databases to suit your use-case/needs.
- You are **strongly encouraged** to read the LMDB docs, they're straightforward and simple.
- If you have questions, make sure to read the manual first.