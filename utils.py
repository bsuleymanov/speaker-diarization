import omegaconf
from omegaconf import OmegaConf
from dataclasses import asdict

import torch

def list_to_tensor(list_of_tensors):
    return torch.stack(list_of_tensors, dim=0)


def map_loaded_weights(weights, model, model_name="rdino_funasr"):
    if model_name == "rdino_funasr":
        return {
            k.replace('module.', ''): v for k, v in weights.items()
            if k.replace('module.', '') in model.state_dict()
        }
    else:
        raise NotImplementedError(f"Unknown model name: {model_name}")


def asdict_filter_none(d):
    if isinstance(d, omegaconf.DictConfig):
        d = OmegaConf.to_container(d, resolve=True)
    else:
        d = asdict(d)
    return {k: v for k, v in d.items() if v is not None}