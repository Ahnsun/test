# -*- coding: utf-8 -*-

from typing import Dict

from yacs.config import CfgNode

from videoanalyst.utils import merge_cfg_into_hps

from ..dataset import builder as dataset_builder
from ..filter import builder as filter_builder
from .sampler_base import TASK_SAMPLERS, DatasetBase


def build(task: str, cfg: CfgNode, seed: int = 0) -> DatasetBase:
    r"""
    Arguments
    ---------
    task: str
        task name (track|vos)
    cfg: CfgNode
        node name: sampler
    seed: int
        seed for rng initialization
    """
    assert task in TASK_SAMPLERS, "invalid task name"
    MODULES = TASK_SAMPLERS[task]

    submodules_cfg = cfg.submodules

    dataset_cfg = submodules_cfg.dataset
    datasets = dataset_builder.build(task, dataset_cfg)

    filter_cfg = getattr(submodules_cfg, "filter", None)
    filt = filter_builder.build(task,
                                filter_cfg) if filter_cfg is not None else None

    name = cfg.name
    module = MODULES[name](datasets, seed=seed, filt=filt)

    hps = module.get_hps()
    hps = merge_cfg_into_hps(cfg[name], hps)
    module.set_hps(hps)
    module.update_params()

    return module


def get_config() -> Dict[str, CfgNode]:
    cfg_dict = {name: CfgNode() for name in TASK_SAMPLERS.keys()}

    for cfg_name, modules in TASK_SAMPLERS.items():
        cfg = cfg_dict[cfg_name]
        cfg["name"] = ""

        for name in modules:
            cfg[name] = CfgNode()
            module = modules[name]
            hps = module.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]

        cfg["submodules"] = CfgNode()
        cfg["submodules"]["dataset"] = dataset_builder.get_config()[cfg_name]
        cfg["submodules"]["filter"] = filter_builder.get_config()[cfg_name]

    return cfg_dict
