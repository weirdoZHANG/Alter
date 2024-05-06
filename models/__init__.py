from typing import Union

import torch

from .Alter import alter


def get_model(model_type: str, **kwargs: Union[int, float]) -> torch.nn.Module:
    if model_type == 'alter':
        model = alter(inputs_feats=kwargs['inputs_feats'])
    else:
        raise ValueError(f"Unknown model type {model_type}")
    return model
