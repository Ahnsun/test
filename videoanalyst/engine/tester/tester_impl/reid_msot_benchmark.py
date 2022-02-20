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
from utils import integral_blocking

from videoanalyst.evaluation import vot_benchmark
from videoanalyst.utils import ensure_dir
from data_association.iou_matching import iou
from data_association.linear_assignment import LinearAssignment
from motion.kalman_tracker import LinearMotion, chi2inv95
from motion.ecc import ECC, AffinePoints

from ..tester_base import TRACK_TESTERS, TesterBase

vot_benchmark.init_log('global', logging.INFO)
logger = logging.getLogger("global")

# parameters
default = {'fading_memory': 1.14, 'dt': 0.15, 'std_weight_position': 0.02,
           'std_weight_velocity': 0.0005, 'duration': 4, 'predict': False,
           'preserve': True, 'width': 64, 'height': 160, 'model_name': 'hacnn_ibn_b',
           'weight_path': 'reid/model/model.pth.tar-150' ,'batch size': 20,
           'thresh': 0.1, 'dims': 1024}
labels = ['mota', 'motp', 'idf1', 'num_switches']

@TRACK_TESTERS.register
class ReidMSOTTester(TesterBase):

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
        super(ReidMSOTTester, self).__init__(*args, **kwargs)


    def test(self):
        # self.video_info = ['02','04','05','09','10','11','13']
        # self.total_len = [601,1051,837,526,655,901,751]
        self.video_info = ['10']
        self.total_len = [655]
        self.reid_thresh = 0.7
        # self.video_info = ['11','13']
        # self.total_len = [901,751]
        for video,video_len in zip(self.video_info, self.total_len):
            print('video ID:{}'.format(video))
            im_path_base = 'datasets/MOT/MOT17/MOT17-' + video + '-SDP/img1/'
            det_path = 'datasets/MOT/MOT17/MOT17-' + video + '-SDP/det/det.txt'
            gt_path = 'datasets/MOT/MOT17/MOT17-' + video + '-SDP/gt/gt.txt'
            self.track_len = video_len
            # set dir
            self.tracker_name = self._hyper_params["exp_name"]

            #prefetch
            self.images = []
            # self.track_len = 655
            for i in range(1, self.track_len):
                im_path = im_path_base + ('00%04d.jpg'%i)
                self.images.append(cv2.imread(im_path))

            dets = np.genfromtxt(det_path, delimiter = ',')
            dets = dets[(dets[:, 0] < self.track_len)&(dets[:, 6] > 0.84) , :]
            self.dets = dets.astype(np.int32)

            gt = np.genfromtxt(gt_path, delimiter = ',')
            gt = gt[(gt[:, 0] < self.track_len)&(gt[:, 6] == 1) , :]
            mask = (gt[:, 7] == 1) | (gt[:, 7] == 2) | (gt[:, 7] == 7)
            self.gt = gt[mask].astype(np.int32)

            # get reid model
            self.mean = torch.cuda.FloatTensor([0.485, 0.456, 0.406], device=torch.device('cuda: 0')).view(3, 1, 1)
            self.std = torch.cuda.FloatTensor([0.229, 0.224, 0.225], device=torch.device('cuda: 0')).view(3, 1, 1)

            self.reid_model = load_reid_model(default['model_name'], default['weight_path'],
                                              gpu_devices='0', input_size=(default['batch size'],
                                                                           3, default['height'], default['width']))
            # get data
            self.method = {'Tensor': lambda x: self.transfer(x['color']),
                           'Features': lambda x: self.get_features(x['Tensor'], x['index'])}
            # self.loader = DataLoader('datasets/MOT/MOT17/MOT17-10-img/', max_size=20,
            #                          color_space='RGB', save_list=['Tensor', 'Features'], **self.method)
            # self.loader.start()

            time0 = time.time()
            self.global_tracker()
            print('Speed:{} FPS'.format(self.track_len /(time.time() - time0)))
            self.evaluation()
            # self.track_visualization()
            # self.loader.stop()
    # def region_search(self, image, features, proposals = None):
    #     #print(proposals.shape)
    #     proposals = self.detector.generate_boxes(image, features, proposals)
    #
    #     results = self.detector.get_detections(image, features, proposals)
    #     detections = postprocess(results[0], self.video_info['shape'][1], self.video_info['shape'][0])
    #     boxes, scores, _ = instance_to_numpy(detections)
    #     #self.detector.visualize(self.image, detections)
    #     return boxes, scores
    #
    # def nms(self, bboxes, scores, nms_thresh):
    #
    #     boxes = deepcopy(bboxes)
    #     boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:] - 1
    #     filter_inds = np.zeros(boxes.shape[0], dtype = np.int)
    #
    #     keep = batched_nms(torch.from_numpy(boxes), torch.from_numpy(scores),
    #                        torch.from_numpy(filter_inds), nms_thresh)
    #     return keep.numpy()

    def transfer(self, img):
        img = torch.cuda.FloatTensor(img, device=torch.device('cuda:0'))
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.unsqueeze(0)

    def get_distance(self, input1, input2, metric = 'cosine'):
        input1_normed = F.normalize(input1, p=2, dim=1)
        input2_normed = F.normalize(input2, p=2, dim=1)
        distmat = 1 - input1_normed.unsqueeze(1) @ input2_normed.unsqueeze(2)

        return distmat.squeeze()


    def get_features(self, img, index=None, rois=None):

        if rois is None:
            rois = self.dets[self.dets[:, 0] == index + 1, 2: 6]

        data = torch.empty(len(rois), 3, default['height'],
                           default['width'], device=torch.device('cuda: 0'))
        for i, det in enumerate(rois):
            det[0] = max(0,det[0])
            det[1] = max(0,det[1])
            roi = img[:, :, det[1]: det[1] + det[3], det[0]: det[0] + det[2]] / 255.
            roi = torch.nn.functional.interpolate(roi, (default['height'],
                                                        default['width']), mode='bilinear', align_corners=False)[0]

            data[i] = (roi - self.mean) / self.std
        features = torch.empty(len(rois), default['dims'], device=torch.device('cuda: 0'))
        if data.size(0) == 0:
            return features

        if data.size(0) < default['batch size']:
            data = torch.cat([data, data[:(default['batch size'] - data.size(0))]], 0)
            features = self.reid_model(data).detach()[:len(rois)]
        else:
            for j in range(int(np.ceil(data.size(0) / default['batch size']))):
                idx1 = min(data.size(0), default['batch size'] * (j + 1))
                idx0 = idx1 - default['batch size']
                features[idx0: idx1] = self.reid_model(data[idx0: idx1]).detach()
        return F.normalize(features, p=2, dim=1)


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
            print('Frame Id:{}'.format(f))
            data = self.transfer(im)
            im_show = im.copy().astype(np.uint8)
            # data = self.loader.getData()

            if f == start_frame:
                #record the first frame
                for i, det in enumerate(self.dets[self.dets[:, 0] == 1]):
                    self.tracks.append({'id': i + 1, 'pause': 0, 'box': det[2:6]})
                    self.record.append([1, i + 1, det[2], det[3], det[4], det[5]])
                #init the mot_tracker
                tracker.mot_init(im, self.tracks)

            elif f > start_frame:
                #update the tracker
                # warp_matrix, _ = ECC(self.images[f - 1], self.images[f], warp_mode=cv2.MOTION_EUCLIDEAN,
                #                      eps=1e-2, max_iter=100, scale=0.1, align=False)
                det = self.dets[self.dets[:, 0] == f + 1]

                self.tracks, track_boxes = tracker.mot_update(im, self.tracks)
                track_boxes = track_boxes.astype(np.int32)
                #get the det and init the cost

                #dets_features = data['Features']
                # print('det:{}'.format(len(det)))
                # if(f >= 326):
                #     print(self.transfer(im))
                #print(data['Tensor'])

                # dets_features = self.get_features(data, rois=det[:,2:6])
                # track_features = self.get_features(data, rois=track_boxes)
                # cost = metrics.compute_distance_matrix(track_features, dets_features, 'cosine')
                # cost = cost.cpu().numpy()

                # print(dets_features.size())
                # print(track_features.size())
                # print(cost.shape())
                # keep = integral_blocking(track_boxes, det, (1920,1080), 2* track_boxes[:, 2:], (16,8))

                cost = np.zeros((len(self.tracks), len(det)))

                dets_features = self.get_features(data, rois=det[1:5, 2:6])
                track_features = self.get_features(data, rois=track_boxes)
                for j, track in enumerate(self.tracks):
                    # if True in keep[j]:
                    #     track_features = self.get_features(data, rois=track_boxes)
                    #     print(keep[j])
                    #     print(track_boxes)
                    #     print(track_boxes[keep[j]])
                    #     print(keep[j])
                    #     print(det[keep[j]])
                    track_feature = track_features[j].unsqueeze(1).transpose(0,1)
                    #print(track_feature.shape)
                    # track_features = self.get_features(data, rois=np.atleast_2d(track_boxes[j]))
                    reid_distance = 1 - self.get_distance(track_feature, dets_features).cpu().numpy()
                    reid_distance = np.atleast_1d(reid_distance)
                    sot_disance = 1 - iou(track['box'], det[:, 2:6])
                    sot_disance = np.atleast_1d(sot_disance)

                    rate = 0.95 ** track['pause']

                    # print(reid_distance, sot_disance, rate)

                    if track['pause'] > 2:
                        decrease_rate = 10 ** ((track['pause'] - 2) / duration)
                        reid_distance *= (decrease_rate / self.reid_thresh)
                    else:
                        reid_distance /= self.reid_thresh
                        #
                    #reid_distance[reid_distance > 50] = 50
                    cost[j,:] = rate * sot_disance + (1 - rate) * reid_distance
                # cost = np.zeros((len(self.tracks), len(det)))
                # for i, track in enumerate(self.tracks):
                #     cost[i, :] = 1 - iou(track['box'], det[:, 2:6])

                #data association
                row_idx, col_idx, unmatched_rows, unmatched_cols, _ = LinearAssignment(cost, threshold= 1 - thresh, method = 'KM')

                for r, c in zip(row_idx, col_idx):
                    self.tracks[r]['box'] = det[c,2:6]
                    self.tracks[r]['pause'] = 0

                for r in np.flip(unmatched_rows, 0):
                    if self.tracks[r]['pause'] >= duration:
                        del self.tracks[r]
                    else:

                        # points = np.array([self.tracks[r]['box'][:2],
                        #                    self.tracks[r]['box'][:2] + self.tracks[r]['box'][2:] - 1])
                        # points_aligned = AffinePoints(points.reshape(2, 2), warp_matrix)
                        # points_aligned = points_aligned.reshape(1, 4)
                        # boxes_aligned = np.c_[points_aligned[:, :2],
                        #                       points_aligned[:, 2:] - points_aligned[:, :2] + 1]
                        # boxes_aligned[:, 2:] = np.clip(boxes_aligned[:, 2:], 1, np.inf)
                        # self.tracks[r]['box'] = boxes_aligned.squeeze()
                        #
                        self.tracks[r]['pause'] += 1

                new_targets = []
                for c in unmatched_cols:
                    total_id += 1
                    new_targets.append({'id': total_id, 'pause': 0, 'box': det[c, 2:6]})
                    self.tracks.append({'id': total_id, 'pause': 0, 'box': det[c, 2:6]})
                tracker.add_init(im, new_targets)

                # print('num of tracks:{}'.format(len(self.tracks)))
                # for track in self.tracks:
                #     self.record.append([f + 1, track['id'], track['box'][0], track['box'][1],
                #                         track['box'][2], track['box'][3]])

                # for j, track in enumerate(self.tracks):
                    # tmp_track = np.abs(track['box'])
                    # tmp_track[2:] = np.clip(tmp_track[2:], 2, np.inf)
                    # tmp_track[2] = min(tmp_track[0] + tmp_track[2], 1919) - tmp_track[0]
                    # tmp_track[3] = min(tmp_track[1] + tmp_track[3], 1919) - tmp_track[1]
                    # if default['preserve']:
                    #     self.record.append([f + 1, track['id'], tmp_track[0], tmp_track[1],
                    #                    tmp_track[2], tmp_track[3]])
                    # else:
                    #     if track['pause'] == 0:
                    #         self.record.append([f + 1, track['id'], tmp_track[0], tmp_track[1],
                    #                        tmp_track[2], tmp_track[3]])
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
        plt.title("mot_SiamFC++ tracker")
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
        # box.to_csv('datasets/MOT/evaluation_box.txt', index=False)

        acc = mm.utils.compare_to_groundtruth(gts, box, 'iou', distth=0.5)
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, \
                             return_dataframe=False)
        print(max(0, summary['mota']), max(0, summary['motp']), max(0, summary['idf1']), max(0,summary['num_switches']))


















