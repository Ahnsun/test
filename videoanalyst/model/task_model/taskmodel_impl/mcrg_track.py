# -*- coding: utf-8 -*

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from videoanalyst.model.common_opr.common_block import (conv_bn_relu,
                                                        xcorr_depthwise)
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_model.taskmodel_base import (TRACK_TASKMODELS,
                                                          VOS_TASKMODELS)

torch.set_printoptions(precision=8)

logger = logging.getLogger("global")


@TRACK_TASKMODELS.register
class McrgTrack(ModuleBase):
    r'''
    Mcrg model for tracking

    Hyper-Parameters
    ----------------
    pretrain_model_path: string
        path to parameter to be loaded into module
    head_width: int
        feature width in head structure
    '''

    default_hyper_params = dict(
        pretrain_model_path="",
        head_width=256,
        conv_weight_std=0.01,
    )

    def __init__(self, backbone, head, loss):
        super(McrgTrack, self).__init__()
        self.basemodel = backbone
        # head
        self.head = head
        # loss
        self.loss = loss

    def forward(self, *args, phase = 'train'):
        #phase train
        if phase == 'train':
            # resolve training data
            training_data = args[0]
            target_imgs = training_data["im_z"]
            search_imgs = training_data["im_x"]


