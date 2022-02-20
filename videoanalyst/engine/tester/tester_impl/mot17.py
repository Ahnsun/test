# -*- coding: utf-8 -*
import copy
import itertools
import logging
import math
import os
from collections import OrderedDict
from multiprocessing import Process, Queue
from os.path import join
import configparser

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

from videoanalyst.evaluation import vot_benchmark
from videoanalyst.utils import ensure_dir
from data_association.iou_matching import iou
from data_association.linear_assignment import LinearAssignment

from ..tester_base import TRACK_TESTERS, TesterBase

vot_benchmark.init_log('global', logging.INFO)
logger = logging.getLogger("global")

# parameters
# parameter = {'thresh': np.arange(0.1, 0.9, 0.01),
#              'duration': np.arange(0, self.track_len, 1)}
default = {'fading_memory': 1.0, 'dt': 0.3, 'std_weight_position': 0.03,
           'std_weight_velocity': 0.003, 'duration': 1, 'thresh': 0.1}
labels = ['mota', 'motp', 'idf1', 'num_switches']

@TRACK_TESTERS.register
class MOTTester(TesterBase):

    extra_hyper_params = dict(
        device_num=1,
        data_root={
            "MOT17": "datasets/MOT/MOT17",
            "MOT19": "datasets/VOT/MOT19"
        },
        dataset_names=[
            "MOT17",
        ],
    )

    def __init__(self, *args, **kwargs):
        super(MOTTester, self).__init__(*args, **kwargs)


    def test(self):

        # set dir
        self.tracker_name = self._hyper_params["exp_name"]

        # self.video_infos = ['01','02', '03','04', '05','06', '07', '08', '09', '10','11','12','13']
        # self.total_len = [451, 601, 1501,1051, 837,837,501, 626, 526, 655, 901,901, 751, 751]
        # self.video_id = ['01', '03', '06', '07', '08', '12', '14']
        # self.total_len = [451, 1501, 837, 1195, 501, 626, 901, 751]
        self.detector = ['SDP', 'DPM', 'FRCNN']
        self.txt_path = 'evaluation/'

        self.video_infos = ['10']
        self.total_len = [655]

        for video,video_len in zip(self.video_infos, self.total_len):
            for detector in self.detector:
                # print('video ID:{}'.format(video))

                path_base = 'datasets/MOT/MOT17/MOT17-' + video + '-' + detector + '/'
                im_path_base = path_base + 'img1/'
                det_path = path_base + 'det/det.txt'
                gt_path = path_base + 'gt/gt.txt'
                info_path = path_base + 'seqinfo.ini'
                self.txt_path = 'evaluation/'

                seq_info = configparser.ConfigParser()
                if os.path.exists(info_path):
                    seq_info.read(info_path)
                    self.video_info = {"name": seq_info.get('Sequence', 'name'),
                                  "shape": (int(seq_info.get('Sequence', 'imWidth')),
                                            int(seq_info.get('Sequence', 'imHeight')))}
                self.track_len = video_len
                print(self.video_info['name'])

                #prefetch
                self.images = []
                # self.track_len = 655
                for i in range(1, self.track_len):
                    im_path = im_path_base + ('00%04d.jpg' % i)
                    self.images.append(cv2.imread(im_path))

                dets = np.genfromtxt(det_path, delimiter=',')
                dets = dets[(dets[:, 0] < self.track_len) & (dets[:, 6] > 0.84), :]
                self.dets = dets.astype(np.int32)

                gt = np.genfromtxt(gt_path, delimiter=',')
                gt = gt[(gt[:, 0] < self.track_len) & (gt[:, 6] == 1), :]
                mask = (gt[:, 7] == 1) | (gt[:, 7] == 2) | (gt[:, 7] == 7)
                self.gt = gt[mask].astype(np.int32)

                time0 = time.time()
                self.global_tracker()
                print('Speed:{} FPS'.format(self.track_len /(time.time() - time0)))
                self.write_results()
                self.evaluation()
                #self.track_visualization()


    def run_tracker(self):

        # thresh = default['thresh'] if thresh is None else thresh
        # duration = default['duration'] if duration is None else duration

        self.record = []
        regions = []
        init_location = []

        num_gpu = 1
        all_devs = [torch.device("cuda:%d" % i) for i in range(num_gpu)]
        logging.info('runing test on devices {}'.format(all_devs))
        tracker = copy.deepcopy(self._pipeline)
        #用第一块GPU
        tracker.to_device(all_devs[0])


        start_frame, end_frame = 0, self.track_len
        #print(self.dets)
        for f,im in enumerate(self.images):
            im_show = im.copy().astype(np.uint8)

            if f == start_frame:
                #tracking init
                total_id = len(self.dets[self.dets[:, 0] == 1])
                for i, det in enumerate(self.dets[self.dets[:, 0] == 1]):
                    ix, iy, w, h = det[2], det[3], det[4], det[5]
                    cx = ix + det[4] / 2
                    cy = iy + det[5] / 2
                    self.record.append([1, i + 1, ix, iy, w, h])
                    init_location.append(vot_benchmark.cxy_wh_2_rect((cx, cy), (w, h)))


                #初始化跟踪器
                tracker.mot_init(im,init_location)
            elif f > start_frame:   # tracking
                tracks = tracker.mot_update(im)
                for track in tracks:
                    self.record.append([f+1, track['id'], track['box'][0], track['box'][1],
                                        track['box'][2], track['box'][3]])

    def one_tracker(self):
        r'''
        在多目标场景下，只对其中一个目标进行跟踪
        :return:
        '''
        self.record = []
        init_location = []

        num_gpu = 1
        all_devs = [torch.device("cuda:%d" % i) for i in range(num_gpu)]
        logging.info('runing test on devices {}'.format(all_devs))
        tracker = copy.deepcopy(self._pipeline)
        # 用第一块GPU
        tracker.to_device(all_devs[0])

        start_frame, end_frame = 0, self.track_len
        for f, im in enumerate(self.images):
            im_show = im.copy().astype(np.uint8)

            if f == start_frame:
                det = self.dets[6]
                ix, iy, w, h = det[2], det[3], det[4], det[5]
                cx = ix + det[4] / 2
                cy = iy + det[5] / 2
                self.record.append([1, 1, ix, iy, w, h])
                location = vot_benchmark.cxy_wh_2_rect((cx, cy), (w, h))
                tracker.init(im, location)

            elif f > start_frame:
                location = tracker.update(im)
                self.record.append([f + 1, 1, location[0], location[1],
                                    location[2], location[3]])

    def global_tracker(self, thresh = None, duration = None):
        r'''
        run the whole MOT tracker
        :param thresh:
        :param duration:
        :return:
        '''

        #set GPU
        num_gpu = 1
        all_devs = [torch.device("cuda:%d" % i) for i in range(num_gpu)]
        logging.info('runing test on devices {}'.format(all_devs))
        tracker = copy.deepcopy(self._pipeline)
        # choose the gpu to use
        tracker.to_device(all_devs[0])

        thresh = default['thresh'] if thresh is None else thresh
        duration = default['duration'] if duration is None else duration

        self.tracks = []
        self.record = []
        total_id = len(self.dets[self.dets[:, 0] == 1])
        start_frame,end_frame = 0,self.track_len - 1

        for f, im in enumerate(self.images):
            #get the frame image
            print(f)
            im_show = im.copy().astype(np.uint8)

            if f == start_frame:
                #record the first frame
                for i, det in enumerate(self.dets[self.dets[:, 0] == 1]):
                    self.tracks.append({'id': i + 1, 'pause': 0, 'box': det[2:6]})
                    self.record.append([1, i + 1, det[2], det[3], det[4], det[5]])
                #init the mot_tracker
                tracker.mot_init(im, self.tracks)

            elif f > start_frame:
                #update the tracker
                det = self.dets[self.dets[:, 0] == f + 1]

                if len(det) > 0 and len(self.tracks) > 0:
                    self.tracks,_ = tracker.mot_update(im, self.tracks)
                    #get the det and init the cost

                    cost = np.zeros((len(self.tracks), len(det)))
                    for i, track in enumerate(self.tracks):
                        cost[i, :] = 1 - iou(track['box'], det[:, 2:6])

                    #data association
                    row_idx, col_idx, unmatched_rows, unmatched_cols, _ = LinearAssignment(cost, threshold= 1-thresh, method = 'KM')
                else:
                    row_idx = []
                    col_idx = []
                    unmatched_rows = np.arange(len(self.tracks))
                    unmatched_cols = np.arange(len(det))

                for r, c in zip(row_idx, col_idx):
                    self.tracks[r]['box'] = det[c,2:6]
                    self.tracks[r]['pause'] = 0

                for r in np.flip(unmatched_rows, 0):
                    if self.tracks[r]['pause'] >= duration:
                        del self.tracks[r]
                    else:
                        self.tracks[r]['pause'] += 1

                new_targets = []
                for c in unmatched_cols:
                    total_id += 1
                    new_targets.append({'id': total_id, 'pause': 0, 'box': det[c, 2:6]})
                    self.tracks.append({'id': total_id, 'pause': 0, 'box': det[c, 2:6]})
                tracker.add_init(im, new_targets)

                # print('num of tracks:{}'.format(len(self.tracks)))
                for track in self.tracks:
                    self.record.append([f + 1, track['id'], track['box'][0], track['box'][1],
                                        track['box'][2], track['box'][3]])


    def track_visualization(self):
        n = self.track_len - 2
        self.record = np.array(self.record)
        #print(self.record)
        rgb = lambda x: colorsys.hsv_to_rgb((x * 0.41) % 1, 1., 1. - (int(x * 0.41) % 4) / 5.)
        colors = lambda x: (int(rgb(x)[0] * 255), int(rgb(x)[1] * 255), int(rgb(x)[2] * 255))
        draw = np.concatenate((self.images[0], self.images[n]), axis=1)
        sz = self.images[0].shape
        #print(self.record)
        boxes = self.record[self.record[:, 0] == 1]
        #print(boxes)
        id_list = list(boxes[:n, 1])
        #print(id_list)
        boxes = boxes[0:n, 2:6]

        track_boxes = []

        for i in id_list:
            t = self.record[(self.record[:, 0] == n + 1) & (self.record[:, 1] == i)]
            if t.size > 0:
                track_boxes.append(t[:, 2:6].squeeze())
            else:
                track_boxes.append(None)

        for i, (bbox, tracked_bbox) in enumerate(zip(boxes, track_boxes)):
            x_tl = (int(bbox[0]), int(bbox[1]))
            x_br = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            x_center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
            cv2.rectangle(draw, x_tl, x_br, colors(i), 5)

            if tracked_bbox is not None:
                y_tl = (int(tracked_bbox[0] + sz[1]), int(tracked_bbox[1]))
                y_br = (int(tracked_bbox[0] + tracked_bbox[2] + sz[1]), int(tracked_bbox[1] + tracked_bbox[3]))
                y_center = (int(tracked_bbox[0] + tracked_bbox[2] / 2 + sz[1]), int(tracked_bbox[1] + tracked_bbox[3] / 2))

                cv2.rectangle(draw, y_tl, y_br, colors(i), 5)
                cv2.line(draw, x_center, y_center, colors(i), 3)

        plt.imshow(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB))
        plt.title("MST Tracker")
        plt.show()





    def evaluation(self):

        gts = pd.DataFrame(self.gt[:, :6], columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])
        #gts.to_csv('datasets/MOT/evaluation_gt.txt', index=False)
        gts = gts.set_index(['FrameId', 'Id'])
        gts[['X', 'Y']] -= (1, 1)
        # gts.to_csv('datasets/MOT/evaluation_gt.txt',index = False)

        box = pd.DataFrame(np.array(self.record), columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])
        #box.to_csv('datasets/MOT/evaluation_box.txt', index=False)
        box = box.set_index(['FrameId', 'Id'])
        box[['X', 'Y']] -= (1, 1)
        box.to_csv('datasets/MOT/evaluation_box.txt', index=False)

        acc = mm.utils.compare_to_groundtruth(gts, box, 'iou', distth=0.5)
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, \
                             return_dataframe=False)
        print(max(0, summary['mota']), max(0, summary['motp']), max(0, summary['idf1']), max(0,summary['num_switches']))

    def write_results(self, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.txt_path, (self.video_info['name'] + '.txt'))

        with open(save_path, 'a') as f:
            for t in self.record:
                f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,-1,-1,-1,-1\n' %
                        (t[0], t[1], t[2], t[3], t[4], t[5]))
                # f.write('%d, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %d, %d, %d\n' %
                #        (t[0], -1, t[2], t[3], t[4], t[5], t[6], -1, -1, -1))

        f.close()


















