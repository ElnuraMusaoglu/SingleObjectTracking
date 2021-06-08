import numpy as np
import cv2
from base import BaseCF

import os


class OpenCV_Tracker(BaseCF):
    def __init__(self, tracker_type):
        super(OpenCV_Tracker).__init__()

        if tracker_type == 'BOOSTING':
            self.tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            self.tracker = cv2.TrackerMIL_create()
        if tracker_type == 'TLD':
            self.tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'CSRT':
            self.tracker = cv2.TrackerCSRT_create()
        if tracker_type == 'GOTURN':
            self.tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            self.tracker = cv2.TrackerMOSSE_create()

    def init(self, first_frame, bbox):
        self.tracker.init(first_frame, bbox)

    def update(self, current_frame, vis=False):
        # Update tracker
        ok, bbox = self.tracker.update(current_frame)
        return [bbox[0], bbox[1], bbox[2], bbox[3]]

def get_img_list(img_dir):
    frame_list = []
    for frame in sorted(os.listdir(img_dir)):
        if os.path.splitext(frame)[1] == '.jpg':
            frame_list.append(os.path.join(img_dir, frame))
    return frame_list
