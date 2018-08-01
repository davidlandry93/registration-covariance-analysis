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
    def prefix(self, new_prefix):
        self._prefix = new_prefix

    def prefixed_key(self, key):
        return self._prefix + key

    def __contains__(self, key):
        pkey = self.prefixed_key(key)
        return self._contains(pkey)

    def _contains(self, key):
        return key in self.memory_cache or self._file_of_key_exists(key) or self._np_file_of_key(key).exists()

    def __getitem__(self, key):
        pkey = self.prefixed_key(key)
        return self._get(pkey)


    def _get(self, key):
        demanded_file = self._filename_of_key(key)

        if key in self.memory_cache:
            value = self.memory_cache[key]
        elif self._np_file_of_key(key).exists():
            value = self._load_numpy(key)
        elif demanded_file.exists():
            value = self._load_from_file(key)
        else:
            raise ValueError('Key {} not found in this filecache instance at {}'.format(key, self.root))

        return value

    def _load_from_file(self, key):
        demanded_file = self.root / (key + '.json')

        with demanded_file.open() as f:
            loaded = json.load(f)

        self.memory_cache[key] = loaded

        return loaded


    def __setitem__(self, key, value):
        pkey = self.prefixed_key(key)
        self._set(pkey, value)

    def _set(self, key, value):
        if isinstance(value, np.ndarray):
            self._save_numpy(key, value)
        else:
            self._save_json(key, value)

        self.memory_cache[key] = value
        self.check_cache_size()


    def check_cache_size(self):
        if len(self.memory_cache) > self.max_size:
            _, _ = self.memory_cache.popitem()

    def save_json(self, key, value):
        pkey = self.prefixed_key(key)
        self._save_json(pkey, value)

    def _save_json(self, key, value):
        demanded_file = self.root / (key + '.json')

        with demanded_file.open('w') as f:
            json.dump(value, f)
            f.flush()


    def __delitem__(self, key):
        pkey = self.prefixed_key(key)
        self._del(key)

    def _del(self, key):
        demanded_file = self.filename_of_key(key)

        if demanded_file.exists():
            os.remove(str(demanded_file))

        if key in self.memory_cache:
            del self.memory_cache[key]

    def filename_of_key(self, key):
        pkey = self.prefixed_key(key)
        return self._filename_of_key(pkey)

    def _filename_of_key(self, key):
        return self.root / (key + '.json')

    def get_or_generate(self, key, generator):
        pkey = self.prefixed_key(key)
        return self._get_or_generate(pkey, generator)

    def _get_or_generate(self, key, generator):
        if self._contains(key):
            return self._get(key)
        else:
            generated = generator()
            self._set(key, generated)

        return generated

    def get_no_prefix(self, key):
        return self._get(key)

    def _file_of_key_exists(self, key):
        file_of_key = self.root / (key + '.json')
        return file_of_key.exists()

    def np_file_of_key(self, key):
        pkey = self.prefixed_key(key)
        return self._np_file_of_key(pkey)

    def _np_file_of_key(self, key):
        return self.root / (key + '.npy')

    def load_numpy(self, key):
        pkey = self.prefixed_key(key)
        return self._load_numpy(pkey)

    def _load_numpy(self, key):
        ndarray = np.load(self._np_file_of_key(key))
        self.memory_cache[key] = ndarray
        self.check_cache_size()
        return ndarray


    def save_numpy(self, key, ndarray):
        pkey = self.prefixed_key(key)
        self._save_numpy(pkey, ndarray)

    def _save_numpy(self, key, ndarray):
        np.save(self._np_file_of_key(key), ndarray)

    def set_no_prefix(self, key, value):
        self._set(key, value)

