import json
import os
import numpy as np
import pathlib

from recova.util import eprint

class FileCache:
    def __init__(self, root, max_size=100):
        self.root = pathlib.Path(root)
        self.memory_cache = {}
        self.max_size = max_size
        self._prefix = ''

        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=False)

    @property
    def prefix(self):
        return self._prefix

    @prefix.setter
    def set_prefix(self, new_prefix):
        self._prefix = new_prefix

    def prefixed_key(self, key):
        return self._prefix + key

    def __contains__(self, key):
        pkey = self.prefixed_key(key)
        return pkey in self.memory_cache or self.file_of_key_exists(key) or self.np_file_of_key(key).exists()

    def __getitem__(self, key):
        pkey = self.prefixed_key(key)
        demanded_file = self.root / (pkey + '.json')

        if pkey in self.memory_cache:
            value = self.memory_cache[pkey]
        elif self.np_file_of_key(key).exists():
            value = self.load_numpy(key)
        elif demanded_file.exists():
            value = self.load_from_file(key)
        else:
            raise ValueError('Key {} not found in this filecache instance at {}'.format(key, self.root))


        return value

    def load_from_file(self, key):
        pkey = self.prefixed_key(key)
        demanded_file = self.root / (pkey + '.json')

        with demanded_file.open() as f:
            loaded = json.load(f)

        self.memory_cache[pkey] = loaded

        return loaded


    def __setitem__(self, key, value):
        pkey = self.prefixed_key(key)

        if isinstance(value, np.ndarray):
            self.save_numpy(key, value)
        else:
            self.save_json(key, value)

        self.memory_cache[pkey] = value
        self.check_cache_size()


    def check_cache_size(self):
        if len(self.memory_cache) > self.max_size:
            _, _ = self.memory_cache.popitem()

    def save_json(self, key, value):
        pkey = self.prefixed_key(key)
        demanded_file = self.root / (pkey + '.json')

        with demanded_file.open('w') as f:
            json.dump(value, f)
            f.flush()

    def __delitem__(self, key):
        pkey = self.prefixed_key
        demanded_file = self.filename_of_key()

        if demanded_file.exists():
            os.remove(str(demanded_file))

        if key in self.memory_cache:
            del self.memory_cache[pkey]

    def filename_of_key(self, key):
        pkey = self.prefixed_key
        return self.root / (pkey + '.json')

    def get_or_generate(self, key, generator):
        pkey = self.prefixed_key(key)
        if pkey in self:
            return self.__getitem__(key)
        else:
            generated = generator()
            self.__setitem__(key, generated)

        return generated

    def file_of_key_exists(self, key):
        pkey = self.prefixed_key()
        file_of_key = self.root / (pkey + '.json')
        return file_of_key.exists()

    def file_of_key(self, key):
        pkey = self.prefixed_key()
        return self.root / (pkey + '.json')

    def np_file_of_key(self, key):
        pkey = self.prefixed_key()
        return self.root / (pkey + '.npy')

    def load_numpy(self, key):
        pkey = self.prefixed_key()
        ndarray = np.load(self.np_file_of_key(key))
        self.memory_cache[pkey] = ndarray
        self.check_cache_size()
        return ndarray

    def save_numpy(self, key, ndarray):
        np.save(self.np_file_of_key(key), ndarray)

