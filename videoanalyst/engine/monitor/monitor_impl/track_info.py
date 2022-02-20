# -*- coding: utf-8 -*
import itertools
import logging
from typing import Dict

import cv2
import numpy as np
from tqdm import tqdm

import torch

from ..monitor_base import TRACK_MONITORS, MonitorBase

logger = logging.getLogger("global")


@TRACK_MONITORS.register
class TrackInfo(MonitorBase):
    r"""
    Print tracking information during training.
    Compatible with _RegularTrainer_

    Hyper-parameters
    ----------------
    """

    default_hyper_params = dict()

    def __init__(self, ):
        r"""
        Arguments
        ---------
        """
        super(TrackInfo, self).__init__()

    def init(self, engine_state: Dict):
        super(TrackInfo, self).init(engine_state)

    def update(self, engine_data: Dict):
        r"""
        """
        # etate
        engine_state = self._state["engine_state"]
        # data
        schedule_info = engine_data["schedule_info"]
        training_losses = engine_data["training_losses"]
        extras = engine_data["extras"]
        time_dict = engine_data["time_dict"]
        # schedule information
        epoch = engine_state["epoch"]
        print_str = 'epoch %d, ' % epoch
        for k in schedule_info:
            print_str += '%s: %.1e, ' % (k, schedule_info[k])
        # loss info
        for k in training_losses:
            l = training_losses[k]
            print_str += '%s: %.3f, ' % (k, l.detach().cpu().numpy())
        # extra info
        for extra in extras.values():
            for k in extra:
                l = extra[k]
                print_str += '%s: %.3f, ' % (k, l)
        # pring elapsed time
        for k in time_dict:
            print_str += "%s: %.1e, " % (k, time_dict[k])

        engine_state["print_str"] = print_str
