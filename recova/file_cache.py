import json
import os
import pathlib

from recova.util import eprint

class FileCache:
    def __init__(self, root):
        self.root = pathlib.Path(root)

        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=False)

    def __getitem__(self, key):
        demanded_file = self.root / (key + '.json')
        value = None

        if demanded_file.exists():
            try:
                with demanded_file.open() as f:
                    value = json.load(f)
            except ValueError:
                pass

        return value

    def __setitem__(self, key, value):
        demanded_file = self.root / (key + '.json')

        with demanded_file.open('w') as f:
            json.dump(value, f)

    def __delitem__(self, key):
        demanded_file = self.filename_of_key

        if demanded_file.exists():
            os.remove(str(demanded_file))

    def filename_of_key(self, key):
        return self.root / (key + '.json')
