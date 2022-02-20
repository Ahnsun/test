# -*- coding: utf-8 -*
import copy
import itertools
import logging
import math
import os
from collections import OrderedDict
from multiprocessing import Process, Queue
from os.path import join

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import colorsys
import motmetrics as mm
import pandas as pd
import time

import torch
from torch.nn import functional as F
from reid import load_reid_model
from torchreid import metrics
from utils.dataloader import DataLoader

from videoanalyst.evaluation import vot_benchmark
from videoanalyst.utils import ensure_dir
from data_association.iou_matching import iou
from data_association.linear_assignment import LinearAssignment
from motion.kalman_tracker import LinearMotion, chi2inv95
from motion.ecc import ECC, AffinePoints

from ..tester_base import TRACK_TESTERS, TesterBase

vot_benchmark.init_log('global', logging.INFO)
logger = logging.getLogger("global")

