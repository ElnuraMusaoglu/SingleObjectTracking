import cv2
import numpy as np
import os
from mosse_tracking.mosse import MOSSE
from hierarchical_conv_features_tracking.hcf import HCF_VGG19_Tracker as HCF_VGG19
from kernelized_correlation_filter.tracker import KernelizedCorrelationFilter
from opencv_trackers.opencv_tracker import OpenCV_Tracker
from kalman.kalman import KalmanFilter
from utils.utils import get_img_list
#from lib.eco.config import otb_deep_config, otb_hc_config
#from cftracker.config import staple_config,ldes_config,dsst_config,csrdcf_config,mkcf_up_config,mccth_staple_config
import time


class Tracker:
    def __init__(self, img_dir, tracker_type):
        self.img_dir = img_dir
        self.tracker_type = tracker_type
        self.frame_list = get_img_list(img_dir)
        self.frame_list.sort()

        if self.tracker_type == 'MOSSE':
            self.tracker = MOSSE()
        elif self.tracker_type == 'Kalman':
            self.tracker = KalmanFilter()

        elif self.tracker_type == 'OPENCV_CSRT':
            self.tracker = OpenCV_Tracker('CSRT')
        elif self.tracker_type == 'OPENCV_GOTURN':
            self.tracker = OpenCV_Tracker('GOTURN')
        elif self.tracker_type == 'OPENCV_BOOSTING':
            self.tracker = OpenCV_Tracker('BOOSTING')
        elif self.tracker_type == 'OPENCV_MEDIANFLOW':
            self.tracker = OpenCV_Tracker('MEDIANFLOW')
        elif self.tracker_type == 'OPENCV_MIL':
            self.tracker = OpenCV_Tracker('MIL')
        elif self.tracker_type == 'OPENCV_TLD':
            self.tracker = OpenCV_Tracker('TLD')
        elif self.tracker_type == 'OPENCV_MOSSE':
            self.tracker = OpenCV_Tracker('MOSSE')

        elif self.tracker_type == 'KCF_GRAY':
            self.tracker = KernelizedCorrelationFilter(correlation_type='gaussian', feature='gray')
        elif self.tracker_type == 'KCF_COLOR':
            self.tracker = KernelizedCorrelationFilter(correlation_type='gaussian', feature='color')
        elif self.tracker_type == 'KCF_HOG':
            self.tracker = KernelizedCorrelationFilter(correlation_type='gaussian', feature='hog')
        elif self.tracker_type == 'HCF':
            self.tracker = HCF_VGG19()
        elif self.tracker_type == 'HCF_C5':
            self.tracker = HCF_VGG19('C5')
        elif self.tracker_type == 'HCF_C4':
            self.tracker = HCF_VGG19('C4')
        elif self.tracker_type == 'HCF_C3':
            self.tracker = HCF_VGG19('C3')
        elif self.tracker_type == 'HCF_C2':
            self.tracker = HCF_VGG19('C2')
        elif self.tracker_type == 'HCF_C2':
            self.tracker = HCF_VGG19('C1')
        else:
            raise NotImplementedError

    def tracking(self, init_gt, show_active=True, video_path=None, det_length=None):
        poses = []
        poses.append(init_gt)
        init_frame = cv2.imread(self.frame_list[0])
        x1, y1, w, h = init_gt
        init_gt = tuple(init_gt)
        self.tracker.init(init_frame, init_gt)
        fr_length = len(self.frame_list)
        total_time = 0

        if det_length:
            if det_length < fr_length:
                fr_length = det_length
        writer = None
        if show_active is True and video_path is not None:
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (init_frame.shape[1], init_frame.shape[0]))

        for idx in range(fr_length):
            if idx != 0:
                current_frame = cv2.imread(self.frame_list[idx])
                start_time = time.time()
                bbox = self.tracker.update(current_frame, vis=show_active)
                finish_time = time.time()
                total_time = total_time + (finish_time - start_time)
                if bbox is not None:
                    x1, y1, w, h = bbox
                    if show_active is True:
                        if len(current_frame.shape) == 2:
                            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)
                        show_frame = cv2.rectangle(current_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                                                   (255, 0, 0), 1)
                        cv2.imshow('demo', show_frame)

                        if writer is not None:
                            writer.write(show_frame)
                        cv2.waitKey(1)
                else:
                    print('bbox is None')
                poses.append(np.array([int(x1), int(y1), int(w), int(h)]))

        return np.array(poses), (fr_length - 1) / total_time

