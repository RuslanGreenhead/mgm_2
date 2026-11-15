"""Factory for the shared sphere/torus training interface."""

from importlib import import_module


_MODULES = {
    "rmgm": "rmgm.methods.rmgm.model",
    "rwflow": "rmgm.methods.baselines.rwflow",
    "kgdm": "rmgm.methods.baselines.kgdm",
    "rmf_mt": "rmgm.methods.baselines.rmf_mt",
}


def create_method(name, ops, config, device):
    try:
        module = _MODULES[name]
    except KeyError as error:
        raise ValueError(f"Unknown sphere/torus method: {name}") from error
    return import_module(module).create_method(ops, config, device)
