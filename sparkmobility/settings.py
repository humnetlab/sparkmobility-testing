from .config.load_config import load_config

class ConfigDict:
    def __init__(self):
        self._values = load_config()

    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        self._values[key] = value

    def __repr__(self):
        return f"Config({self._values})"

    def as_dict(self):
        return dict(self._values)

    def reset(self):
        self._values = load_config()

config = ConfigDict()