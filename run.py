import os
from os.path import join

import numpy as np

from main_tracker import Tracker
from utils.utils import get_ground_truthes, calAUC, get_ground_truthes_pf_path, get_thresh_precision_pair, \
    get_thresh_success_pair
from utils.utils import plot_all_success_for_tracking_methods, \
    plot_all_precision_for_tracking_methods, write_gts, create_dir

'''

['MOSSE', 'Kalman',
'OPENCV_CSRT', 'OPENCV_GOTURN', 'OPENCV_BOOSTING', 'OPENCV_MEDIANFLOW', 'OPENCV_MIL', 'OPENCV_TLD', 'OPENCV_MOSSE', 
'KCF_GRAY', 'KCF_COLOR', 'KCF_HOG',
'HCF_C1', 'HCF_C2', 'HCF_C3', 'HCF_C4', 'HCF_C5', 'HCF']
'''


def run_all_trackers(data_dir, result_gt_dir):
    tracking_methods = ['HCF_C1', 'HCF', 'HCF_C5']

    for tracking_method in tracking_methods:
        run_tracker(tracking_method, data_dir, result_gt_dir)


def run_tracker(tracking_method, data_dir, result_gt_dir):
    data_names = sorted(os.listdir(data_dir))
    gt_path = join(result_gt_dir, tracking_method)
    create_dir(gt_path)
    total_fps = []

    for data_name in data_names:
        try:
            print(data_name)
            data_path = join(data_dir, data_name)
            gts = get_ground_truthes(data_path)

            img_dir = os.path.join(data_path, 'img')
            tracker = Tracker(img_dir, tracker_type=tracking_method)

            poses, fps = tracker.tracking(init_gt=gts[0], show_active=False, video_path='', det_length=10000)
            total_fps.append(fps)

            gt_path_result = join(gt_path, data_name + '.txt')
            if os.path.exists(gt_path_result):
                os.remove(gt_path_result)
            write_gts(gt_path_result, poses)
        except Exception as e:
            print(e)

    mean_fps = int(np.mean(total_fps))

    print('Mean FPS for tracking method ', tracking_method, mean_fps)


def evaluate_all_trackers(data_dir, result_gt_dir, plot_path):
    data_names = sorted(os.listdir(data_dir))

    tracking_methods = ['HCF', 'KCF_HOG', 'MOSSE', 'Kalman']
    # tracking_methods = ['HCF', 'CSRT', 'KCF_HOG', 'MIL', 'BOOSTING', 'MOSSE', 'TLD', 'MEDIANFLOW', 'Kalman']
    # tracking_methods = ['HCF', 'HCF_C4', 'HCF_C5', 'HCF_C3', 'KCF_HOG', 'HCF_C2', 'HCF_C1', 'KCF_COLOR', 'KCF_GRAY']

    all_precisions_for_tracking_method_global = []
    all_thresh_p_for_tracking_method_global = []
    all_success_for_tracking_method_global = []
    all_thresh_s_for_tracking_method_global = []

    for tracking_method in tracking_methods:
        all_precisions_for_tracking_method = []
        all_thresh_p_for_tracking_method = []
        all_success_for_tracking_method = []
        all_thresh_s_for_tracking_method = []

        for data_name in data_names:
            try:
                data_path = join(data_dir, data_name)
                tracker_gt_path = join(join(result_gt_dir, tracking_method), data_name + '.txt')
                actual_gts = get_ground_truthes(data_path)
                preds_gts = get_ground_truthes_pf_path(tracker_gt_path)

                threshes_p, precisions = get_thresh_precision_pair(actual_gts, preds_gts)
                all_thresh_p_for_tracking_method.append(threshes_p)
                all_precisions_for_tracking_method.append(precisions)
                idx20 = [i for i, x in enumerate(threshes_p) if x == 20][0]
                print(tracking_method, data_name, '20px: ', str(round(precisions[idx20], 3)))

                threshes_s, successes = get_thresh_success_pair(actual_gts, preds_gts)
                all_thresh_s_for_tracking_method.append(threshes_s)
                all_success_for_tracking_method.append(successes)
            except Exception as e:
                print(e)

        # compute average precision for tracking_method at 20px
        precision_mean = np.mean(all_precisions_for_tracking_method, axis=0)
        threshes_p_mean = np.mean(all_thresh_p_for_tracking_method, axis=0)
        idx20 = [i for i, x in enumerate(threshes_p_mean) if x == 20][0]
        print(tracking_method, ' Average precision: ', str(round(precision_mean[idx20], 3)))

        # compute average success for tracking_method at 0.5
        success_mean = np.mean(all_success_for_tracking_method, axis=0)
        threshes_s_mean = np.mean(all_thresh_s_for_tracking_method, axis=0)
        print(tracking_method, ' Average success AUC: ', str(round(calAUC(success_mean), 3)))

        all_precisions_for_tracking_method_global.append(precision_mean)
        all_thresh_p_for_tracking_method_global.append(threshes_p_mean)
        all_success_for_tracking_method_global.append(success_mean)
        all_thresh_s_for_tracking_method_global.append(threshes_s_mean)

    create_dir(plot_path)

    success_plot_path = join(plot_path, 'success_result.png')
    precision_plot_path = join(plot_path, 'precision_result.png')

    plot_all_success_for_tracking_methods(tracking_methods, all_thresh_s_for_tracking_method_global,
                                          all_success_for_tracking_method_global, success_plot_path)
    plot_all_precision_for_tracking_methods(tracking_methods, all_thresh_p_for_tracking_method_global,
                                            all_precisions_for_tracking_method_global, precision_plot_path)


if __name__ == '__main__':
    data_dir = ''  # OTB-100 dataset
    result_gt_dir = ''  # Tracking result GT write path
    plot_path = ''  # Tracking result plot path

    # run_all_trackers(data_dir, result_gt_dir)  #  1. step run trackers

    evaluate_all_trackers(data_dir, result_gt_dir, plot_path)  # 2. step evaluate trackers result
