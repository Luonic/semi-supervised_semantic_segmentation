import os
import sys
from importlib import import_module


def fromfile(filename):
    if not os.path.exists(filename):
        raise ValueError('Config path does not exist')

    if filename.endswith('.py'):
        module_name = os.path.basename(filename)[:-3]
        if '.' in module_name:
            raise ValueError('Dots are not allowed in config file path.')
        config_dir = os.path.dirname(filename)
        sys.path.insert(0, config_dir)
        mod = import_module(module_name)
        sys.path.pop(0)
        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith('__')
        }
    else:
        raise IOError('Only .py type are supported now!')
    return cfg_dict
