import configparser
import numpy as np
import ast
from tensordict import TensorDict
import torch as tr
from torchrl.data import ReplayBuffer, LazyTensorStorage, ListStorage, LazyMemmapStorage

device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')


def hyperparams_dict(section, path):
    config = configparser.ConfigParser()
    config.read(path)
    if not config.read(path):
        raise Exception("Could not read config file")
    
    params = config[section]
    typed_params = {}
    for key, value in params.items():
        try:
            # Attempt to evaluate the value to infer type
            typed_params[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Fallback to the original string value if evaluation fails
            typed_params[key] = value
    
    return typed_params


