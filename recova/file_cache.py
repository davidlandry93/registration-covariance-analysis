import json
import os
import numpy as np
import pathlib

from recova.util import eprint

class FileCache:
    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.memory_cache = {}

        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=False)

    def __contains__(self, key):
        return key in self.memory_cache or self.file_of_key_exists(key) or self.np_file_of_key(key).exists()

    def __getitem__(self, key):
        demanded_file = self.root / (key + '.json')

        if key in self.memory_cache:
            value = self.memory_cache[key]
        elif self.np_file_of_key(key).exists():
            value = self.load_numpy(key)
        elif demanded_file.exists():
            value = self.load_from_file(key)
        else:
            raise ValueError('Key {} not found in this filecache instance at {}'.format(key, self.root))

        return value

    def load_from_file(self, key):
        demanded_file = self.root / (key + '.json')

        with demanded_file.open() as f:
            loaded = json.load(f)

        self.memory_cache[key] = loaded

        return loaded

    def __setitem__(self, key, value):
        if isinstance(value, np.ndarray):
            self.save_numpy(key, value)
        else:
            self.save_json(key, value)

        self.memory_cache[key] = value

    def save_json(self, key, value):
        demanded_file = self.root / (key + '.json')

        with demanded_file.open('w') as f:
            json.dump(value, f)
            f.flush()

    def __delitem__(self, key):
        demanded_file = self.filename_of_key

        if demanded_file.exists():
            os.remove(str(demanded_file))

        if key in self.memory_cache:
            del self.memory_cache[key]

    def filename_of_key(self, key):
        return self.root / (key + '.json')

    def get_or_generate(self, key, generator):
        if key in self:
            return self.__getitem__(key)
        else:
            generated = generator()
            self.__setitem__(key, generated)

        return generated

    def file_of_key_exists(self, key):
        file_of_key = self.root / (key + '.json')
        return file_of_key.exists()

    def file_of_key(self, key):
        return self.root / (key + '.json')

    def np_file_of_key(self, key):
        return self.root / (key + '.npy')

    def load_numpy(self, key):
        ndarray = np.load(self.np_file_of_key(key))
        self.memory_cache[key] = ndarray
        return ndarray

    def save_numpy(self, key, ndarray):
        np.save(self.np_file_of_key(key), ndarray)

