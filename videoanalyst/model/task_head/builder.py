# -*- coding: utf-8 -*
import logging
from typing import Dict

from yacs.config import CfgNode

from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_head.taskhead_base import TRACK_HEADS, VOS_HEADS
from videoanalyst.utils import merge_cfg_into_hps

logger = logging.getLogger(__file__)

TASK_HEADS = dict(
    track=TRACK_HEADS,
    vos=VOS_HEADS,
)


def build(task: str, cfg: CfgNode):
    r"""
    Builder function.

    Arguments
    ---------
    task: str
        builder task name (track|vos)
    cfg: CfgNode
        buidler configuration

    Returns
    -------
    torch.nn.Module
        module built by builder
    """
    if task in TASK_HEADS:
        head_modules = TASK_HEADS[task]
    else:
        logger.error("no task model for task {}".format(task))
        exit(-1)

    name = cfg.name
    if task == "track":
        # head settings
        head_module = head_modules[name]()
        hps = head_module.get_hps()
        hps = merge_cfg_into_hps(cfg[name], hps)
        head_module.set_hps(hps)
        head_module.update_params()

        return head_module
    else:
        logger.error("task model {} is not completed".format(task))
        exit(-1)


def get_config() -> Dict[str, CfgNode]:
    r"""
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]
        config with list of available components
    """
    cfg_dict = {"track": CfgNode(), "vos": CfgNode()}
    for cfg_name, module in zip(["track", "vos"], [TRACK_HEADS, VOS_HEADS]):
        cfg = cfg_dict[cfg_name]
        cfg["name"] = "unknown"
        for name in module:
            cfg[name] = CfgNode()
            task_model = module[name]
            hps = task_model.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict
