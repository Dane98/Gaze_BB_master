"""
Outputs gaze annotation for video with path ./testvid/cut.mp4 and saves it to pitchjawcut.csv

run the script:

 python3 demo_annotate_inv.py \
 --snapshot models/L2CSNet_gaze360.pkl \
 --gpu 0 \
 --cam 0 
"""

import argparse
import numpy as np
import os
import cv2
import time
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from utils import *
from PIL import Image, ImageOps

from filterpy.kalman import KalmanFilter
from face_detection import RetinaFace
from model import L2CS
from readbbtxt import readbbtxt as readvis
from readbbtxt_inv import readbbtxt as readinv

NOFACE = 42

datafolder = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/data_fin/'
vis_datafile = 'pixel_position_vis.txt'
invis_datafile = 'pixel_position_invis_new.txt'

root = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/'
output_root = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/frames/Saved_frame/'
vis_frame_out_root = output_root + 'fig_vis/'
invis_frame_out_root = output_root + 'fig_invis/'
vis_l2cs_root = output_root + 'fig_vis_l2cs_filtered_m2prop/'
invis_l2cs_root = output_root + 'fig_invis_l2cs_filtered_m2prop/'

# vis_root_video = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/visible_bb_l2cs/'
# invis_root_video = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/invisible_bb_l2cs/'
vis_video_folder = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/visible_bb/'
invis_video_folder = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/invisible_bb/'

vis_data = readvis(datafolder + vis_datafile)
invis_data = readinv(datafolder + invis_datafile)
# remove .png extension from filenames
vis_data['file'] = vis_data['file'].apply(lambda a: a[:-4])
invis_data['file'] = invis_data['file'].apply(lambda a: a[:-4])

# video_file_list = []
vis_video_list = [vid for vid in os.listdir(vis_video_folder) if (vid.endswith(".MP4") or vid.endswith(".mp4"))]
vis_video_list = sorted(vis_video_list)
# print(video_list[0:3])
invis_video_list = [vid for vid in os.listdir(invis_video_folder) if (vid.endswith(".MP4") or vid.endswith(".mp4"))]
invis_video_list = sorted(invis_video_list)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='models_local/Gaze360/L2CSNet_gaze360.pkl', type=str)
    parser.add_argument(
        '--cam', dest='cam_id', help='Camera device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args


def getArch(arch, bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
    return model


def inbb(tlx, tly, brx, bry, gazex, gazey):
    return tlx <= gazex <= tlx + (brx - tlx) and tly <= gazey <= tly + (bry - tly)


def get_classname(id):
    if id == 0:
        return 'Pen & paper'
    elif id == 1:
        return 'robot'
    elif id == 2:
        return 'tablet'
    elif id == 3:
        return 'elsewhere'
    elif id == 4:
        return 'unknown'


# https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

# def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
#     """
#     This is a general function to calculate the intersection point between four points (two lines)
#
#     Returns the coordinates px, py. p is the intersection of the lines
#     defined by ((x1, y1), (x2, y2)) and ((x3, y3), (x4, y4))
#     """
#     px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
#             (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
#     py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
#             (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
#     return int(px), int(py)

def line_segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    This is a general function to calculate the intersection point between four points (two line segments)

    Returns the coordinate intersection_x, intersection_y.
    defined by ((x1, y1), (x2, y2)) and ((x3, y3), (x4, y4))
    if no intersection, return None
    """

    denom = (x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3)
    num1 = (x4 - x3) * (y1 - y3) - (x1 - x3) * (y4 - y3)
    num2 = (x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1)

    if denom == 0:  # two line segments are parallel or coincident
        return None

    t1 = num1 / denom
    t2 = num2 / denom

    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        intersection_x = round(x1 + t1 * (x2 - x1))
        intersection_y = round(y1 + t1 * (y2 - y1))
        return intersection_x, intersection_y
    else:
        return None


def intersection_proportion(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    This is a function to calculate the proportion of the intersected segments in the bounding box area

    Return a number that represent the percentage of the proportion
    """
    box_width = x2 - x1
    box_height = y2 - y1
    diagonal_length = (box_height ** 2 + box_width ** 2) ** 0.5

    delta_x = max(x3, x4) - min(x3, x4)
    delta_y = max(y3, y4) - min(y3, y4)
    intersection_length = (delta_y ** 2 + delta_y ** 2) ** 0.5

    proportion = intersection_length / diagonal_length * 100

    return proportion


def classify_line_box_intersection(a, b, c, d, image_in, pitch_pred, jaw_pred, face, vid_type, filename):
    """
    - This is the function to classify the intersection points between gaze and bounding boxs
    without considering the proportion of intersection.
    - The sequence of consideration is tablet, pen & paper, and then the robot. (based on their size in image. this
    might be able to increase the accuracy of the classification in the case of objects overlapping)

    Return one of five numbers (0-4) which represents the object label.
    """
    intersection_points_tablet = []
    intersection_points_pp = []
    intersection_points_robot = []
    # box_corners = [(1, 1), (3, 1), (3, 3), (1, 3)]

    # global box_corners_tablet
    # global box_corners_pp
    # global box_corners_robot
    # global intersection_x_t, intersection_y_t
    # global intersection_x_p, intersection_y_p
    # global intersection_x_r, intersection_y_r

    # if pitch_pred > 21 and jaw_pred > 21:
    #     return 4
    if face == 0:
        return 4

    (h, w) = image_in.shape[:2]
    length_norm = w / 2     # this is the original calculation for gaze line
    length_ext = w * 999    # this is to extend the gaze line
    pos = ((a + c / 2.0), (b + d / 2.0))    # central position of the head
    # print(f'pos is {pos}')
    dx_norm = -length_norm * np.sin(pitch_pred) * np.cos(jaw_pred)
    dy_norm = -length_norm * np.sin(jaw_pred)
    dx_ext = -length_ext * np.sin(pitch_pred) * np.cos(jaw_pred)
    dy_ext = -length_ext * np.sin(jaw_pred)
    gazex_norm = round(pos[0] + dx_norm)  # the end point of original gaze (x-axis)
    gazey_norm = round(pos[1] + dy_norm)  # the end point of original gaze (y-axis)
    gazex_ext = round(pos[0] + dx_ext)  # the end point of extended gaze (x-axis)
    gazey_ext = round(pos[1] + dy_ext)  # the end point of extended gaze (y-axis)
    # print(f'gazex, gazey = {(gazex, gazey)}')

    try:
        gaze_slope = float((gazey_norm - pos[1]) / (gazex_norm - pos[0]))
        # print(f"gaze slope: {gaze_slope}")
    except ZeroDivisionError:
        gaze_slope = 1.0
    gaze_intercept = float(pos[1] - gaze_slope * pos[0])
    # print(f"gaze intercept: {gaze_intercept}")

    image_edges = [(0, 0), (w, 0), (w, h), (0, h)]  # define the edges of the image (video frame)
    x1, y1 = round(pos[0]), round(pos[1])
    # x2_norm, y2_norm = gazex_norm, gazey_norm
    x2_ext, y2_ext = gazex_ext, gazey_ext
    # for edge in range(4):
    #     x3, y3 = image_edges[edge]
    #     x4, y4 = image_edges[(edge + 1) % 4]  # Next corner (wraps around to the first corner)
    #
    #     if line_segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4) is not None:
    #         gazex, gazey = line_segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
    #     else:
    #         gazex, gazey = x2, y2

    # # Check if gaze vector leaves the image
    # if gazex < 0 or gazex > w or gazey < 0 or gazey > h:
    #     # Do line intersection to find at which point the gaze vector line
    #     # leaves the image.
    #     image_edges = [(0, 0), (w, 0), (w, h), (0, h)]
    #
    #     # y3 = gazey if 0 <= gazey <= h else 0 if gazey < 0 else h
    #     # y4 = y3
    #
    #     x1, y1 = round(pos[0]), round(pos[1])
    #     x2, y2 = gazex, gazey
    #
    #     for edge in [1, 2, 3]:
    #         x3, y3 = image_edges[edge]
    #         x4, y4 = image_edges[(edge + 1) % 4]  # Next corner (wraps around to the first corner)
    #
    #         gazex, gazey = line_intersect(x1, y1, x2, y2, x3, y3, x4, y4)
    #
    #         if min(x1, x2) <= gazex <= max(x1, x2) and min(y1, y2) <= gazey <= max(y1, y2):
    #             # print(gazex, gazey)
    #             break

    # print(f"new gazex: {gazex}, new gazey: {gazey}")

    # Read the data file that contains the info of bounding box.
    if vid_type == 'vis':
        rec = vis_data[vis_data['file'] == filename].iloc[0]
    elif vid_type == 'invis':
        rec = invis_data[invis_data['file'] == filename].iloc[0]
    else:
        rec = None
        print('Video type error!')

    x_left_tab = int(rec['tl_tablet_x'] * w)
    x_right_tab = int(rec['br_tablet_x'] * w)
    y_bottom_tab = int(rec['br_tablet_y'] * h)
    y_top_tab = int(rec['tl_tablet_y'] * h)

    x_left_pp = int(rec['tl_pp_x'] * w)
    x_right_pp = int(rec['br_pp_x'] * w)
    y_bottom_pp = int(rec['br_pp_y'] * h)
    y_top_pp = int(rec['tl_pp_y'] * h)

    x_left_rbt = int(rec['tl_robot_x'] * w)
    x_right_rbt = int(rec['br_robot_x'] * w)
    y_bottom_rbt = int(rec['br_robot_y'] * h)
    y_top_rbt = int(rec['tl_robot_y'] * h)

    # Store the info of bounding boxes in four points with x-axis & y-axis
    box_corners_tablet = [(x_left_tab, y_bottom_tab), (x_right_tab, y_bottom_tab), (x_right_tab, y_top_tab),
                          (x_left_tab, y_top_tab)]
    box_corners_pp = [(x_left_pp, y_bottom_pp), (x_right_pp, y_bottom_pp), (x_right_pp, y_top_pp),
                      (x_left_pp, y_top_pp)]
    box_corners_robot = [(x_left_rbt, y_bottom_rbt), (x_right_rbt, y_bottom_rbt), (x_right_rbt, y_top_rbt),
                         (x_left_rbt, y_top_rbt)]

    # image_temp = Image.fromarray(image_in)
    # # im = cv2.imread(image_temp)
    # cv2.rectangle(image_temp, box_corners_tablet[3], box_corners_tablet[1], (0, 0, 255), 2)
    # cv2.rectangle(image_temp, box_corners_pp[3], box_corners_pp[1], (0, 255, 0), 2)
    # cv2.rectangle(image_temp, box_corners_robot[3], box_corners_robot[1], (255, 0, 0), 2)
    # print(box_corners_tablet)
    # print(box_corners_pp)
    # print(box_corners_robot)

    # to check if the original gaze end point is in the bbox or not, if not then check intersections for extended gaze
    if inbb(x_left_tab, y_top_tab, x_right_tab, y_bottom_tab, gazex_norm, gazey_norm):
        return 2
    elif inbb(x_left_pp, y_top_pp, x_right_pp, y_bottom_pp, gazex_norm, gazey_norm):
        return 0
    elif inbb(x_left_rbt, y_top_rbt, x_right_rbt, y_bottom_rbt, gazex_norm, gazey_norm):
        return 1
    else:
        # check if there is any the intersection point in tablet box
        for i in range(4):
            x1_t, y1_t = box_corners_tablet[i]
            x2_t, y2_t = box_corners_tablet[(i + 1) % 4]  # Move to next corner (wraps around to the first corner)
            # print((x1_t, y1_t))
            # print((x2_t, y2_t))
            # Check if the line and line segment are parallel
            try:
                # tablet_edge_slope = round((y2_t - y1_t) / (x2_t - x1_t))
                tablet_edge_slope = abs((y2_t - y1_t) / (x2_t - x1_t))
                # print(f"tablet edge slope: {tablet_edge_slope}")
            except ZeroDivisionError:
                tablet_edge_slope = 1.0
                # print(f"tablet edge slope: {tablet_edge_slope}")

            if gaze_slope == tablet_edge_slope:
                continue

            if line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_t, y1_t, x2_t, y2_t) is not None:
                # not none indicates there is at least one intersection
                intersection_x_t, intersection_y_t = line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_t, y1_t, x2_t,
                                                                               y2_t)

                # Check if the intersection point lies within the range of the line segment
                if min(x1_t, x2_t) <= abs(intersection_x_t) <= max(x1_t, x2_t) and min(y1_t, y2_t) <= abs(
                        intersection_y_t) <= max(y1_t, y2_t):
                    intersection_points_tablet.append((intersection_x_t, intersection_y_t))
                    cv2.circle(image_in, (intersection_x_t, intersection_y_t), 10, (0, 0, 255), 1)

            # if intersection_points_tablet:
            #     print(f"tablet: {intersection_points_tablet}")
        # return intersection_points_tablet

        # check if there is any the intersection point in pp box
        for i in range(4):
            x1_p, y1_p = box_corners_pp[i]
            x2_p, y2_p = box_corners_pp[(i + 1) % 4]  # Next corner (wraps around to the first corner)

            # Check if the line and line segment are parallel
            try:
                # pp_edge_slope = round((y2_p - y1_p) / (x2_p - x1_p))
                pp_edge_slope = (y2_p - y1_p) / (x2_p - x1_p)
            except ZeroDivisionError:
                pp_edge_slope = 1.0

            if gaze_slope == pp_edge_slope:
                continue

            if line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_p, y1_p, x2_p, y2_p) is not None:
                intersection_x_p, intersection_y_p = line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_p, y1_p, x2_p,
                                                                               y2_p)

                # Check if the intersection point lies within the range of the line segment
                if min(x1_p, x2_p) <= abs(intersection_x_p) <= max(x1_p, x2_p) and min(y1_p, y2_p) <= abs(
                        intersection_y_p) <= max(y1_p, y2_p):
                    intersection_points_pp.append((intersection_x_p, intersection_y_p))
                    cv2.circle(image_in, (intersection_x_p, intersection_y_p), 10, (255, 0, 0), 1)

            # if intersection_points_pp:
            #     print(f"pp: {intersection_points_pp}")
        # return intersection_points_tablet

        # check if there is any the intersection point in robot box
        for i in range(4):
            x1_r, y1_r = box_corners_robot[i]
            x2_r, y2_r = box_corners_robot[(i + 1) % 4]  # Next corner (wraps around to the first corner)

            # Check if the line and line segment are parallel
            try:
                # robot_edge_slope = round((y2_r - y1_r) / (x2_r - x1_r))
                robot_edge_slope = (y2_r - y1_r) / (x2_r - x1_r)
            except ZeroDivisionError:
                robot_edge_slope = 1.0

            if gaze_slope == robot_edge_slope:
                continue

            if line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_r, y1_r, x2_r, y2_r) is not None:
                intersection_x_r, intersection_y_r = line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_r, y1_r, x2_r,
                                                                               y2_r)

                # Check if the intersection point lies within the range of the line segment
                if min(x1_r, x2_r) <= abs(intersection_x_r) <= max(x1_r, x2_r) and min(y1_r, y2_r) <= abs(
                        intersection_y_r) <= max(y1_r, y2_r):
                    intersection_points_robot.append((intersection_x_r, intersection_y_r))
                    cv2.circle(image_in, (intersection_x_r, intersection_y_r), 10, (0, 255, 0), 1)

            # if intersection_points_robot:
            #     print(f"robot: {intersection_points_robot}")
            # return intersection_points_tablet

        # print(f'intersections with tablet are {len(intersection_points_tablet)}')
        # print(f'intersections with pp are {len(intersection_points_pp)}')
        # print(f'intersections with robot are {len(intersection_points_robot)}')

        if intersection_points_tablet:  # return 2 if list of intersection points is not empty
            return 2
        elif intersection_points_pp:
            return 0
        elif intersection_points_robot:
            return 1
        else:
            return 3


def classify_line_box_intersection_proportion(a, b, c, d, image_in, pitch_pred, jaw_pred, face, vid_type, filename):
    """
    - This is the function to classify the intersection points between gaze and bounding boxs
    *********  with considering the proportion of intersection!!! ***********
    - The sequence of consideration is tablet, pen & paper, and then the robot. (based on their size in image. this
    might be able to increase the accuracy of the classification in the case of objects overlapping)

    Return one of five numbers (0-4) which represents the object label.
    """

    intersection_points_tablet = []
    intersection_points_pp = []
    intersection_points_robot = []
    # box_corners = [(1, 1), (3, 1), (3, 3), (1, 3)]

    # global box_corners_tablet
    # global box_corners_pp
    # global box_corners_robot
    # global intersection_x_t, intersection_y_t
    # global intersection_x_p, intersection_y_p
    # global intersection_x_r, intersection_y_r

    # if pitch_pred > 21 and jaw_pred > 21:
    #     return 4
    if face == 0:
        return 4

    (h, w) = image_in.shape[:2]
    length_norm = w / 2
    length_ext = w * 999
    pos = ((a + c / 2.0), (b + d / 2.0))
    # print(f'pos is {pos}')
    dx_norm = -length_norm * np.sin(pitch_pred) * np.cos(jaw_pred)
    dy_norm = -length_norm * np.sin(jaw_pred)
    dx_ext = -length_ext * np.sin(pitch_pred) * np.cos(jaw_pred)
    dy_ext = -length_ext * np.sin(jaw_pred)
    gazex_norm = round(pos[0] + dx_norm)
    gazey_norm = round(pos[1] + dy_norm)
    gazex_ext = round(pos[0] + dx_ext)
    gazey_ext = round(pos[1] + dy_ext)
    # print(f'gazex, gazey = {(gazex, gazey)}')

    try:
        gaze_slope = float((gazey_norm - pos[1]) / (gazex_norm - pos[0]))
        # print(f"gaze slope: {gaze_slope}")
    except ZeroDivisionError:
        gaze_slope = 1.0
    gaze_intercept = float(pos[1] - gaze_slope * pos[0])
    # print(f"gaze intercept: {gaze_intercept}")

    image_edges = [(0, 0), (w, 0), (w, h), (0, h)]
    x1, y1 = round(pos[0]), round(pos[1])
    # x2_norm, y2_norm = gazex_norm, gazey_norm
    x2_ext, y2_ext = gazex_ext, gazey_ext
    # for edge in range(4):
    #     x3, y3 = image_edges[edge]
    #     x4, y4 = image_edges[(edge + 1) % 4]  # Next corner (wraps around to the first corner)
    #
    #     if line_segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4) is not None:
    #         gazex, gazey = line_segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
    #     else:
    #         gazex, gazey = x2, y2

    # # Check if gaze vector leaves the image
    # if gazex < 0 or gazex > w or gazey < 0 or gazey > h:
    #     # Do line intersection to find at which point the gaze vector line
    #     # leaves the image.
    #     image_edges = [(0, 0), (w, 0), (w, h), (0, h)]
    #
    #     # y3 = gazey if 0 <= gazey <= h else 0 if gazey < 0 else h
    #     # y4 = y3
    #
    #     x1, y1 = round(pos[0]), round(pos[1])
    #     x2, y2 = gazex, gazey
    #
    #     for edge in [1, 2, 3]:
    #         x3, y3 = image_edges[edge]
    #         x4, y4 = image_edges[(edge + 1) % 4]  # Next corner (wraps around to the first corner)
    #
    #         gazex, gazey = line_intersect(x1, y1, x2, y2, x3, y3, x4, y4)
    #
    #         if min(x1, x2) <= gazex <= max(x1, x2) and min(y1, y2) <= gazey <= max(y1, y2):
    #             # print(gazex, gazey)
    #             break

    # print(f"new gazex: {gazex}, new gazey: {gazey}")

    if vid_type == 'vis':
        rec = vis_data[vis_data['file'] == filename].iloc[0]
    elif vid_type == 'invis':
        rec = invis_data[invis_data['file'] == filename].iloc[0]
    else:
        rec = None
        print('Video type error!')

    x_left_tab = int(rec['tl_tablet_x'] * w)
    x_right_tab = int(rec['br_tablet_x'] * w)
    y_bottom_tab = int(rec['br_tablet_y'] * h)
    y_top_tab = int(rec['tl_tablet_y'] * h)

    x_left_pp = int(rec['tl_pp_x'] * w)
    x_right_pp = int(rec['br_pp_x'] * w)
    y_bottom_pp = int(rec['br_pp_y'] * h)
    y_top_pp = int(rec['tl_pp_y'] * h)

    x_left_rbt = int(rec['tl_robot_x'] * w)
    x_right_rbt = int(rec['br_robot_x'] * w)
    y_bottom_rbt = int(rec['br_robot_y'] * h)
    y_top_rbt = int(rec['tl_robot_y'] * h)

    box_corners_tablet = [(x_left_tab, y_bottom_tab), (x_right_tab, y_bottom_tab), (x_right_tab, y_top_tab),
                          (x_left_tab, y_top_tab)]
    box_corners_pp = [(x_left_pp, y_bottom_pp), (x_right_pp, y_bottom_pp), (x_right_pp, y_top_pp),
                      (x_left_pp, y_top_pp)]
    box_corners_robot = [(x_left_rbt, y_bottom_rbt), (x_right_rbt, y_bottom_rbt), (x_right_rbt, y_top_rbt),
                         (x_left_rbt, y_top_rbt)]

    # image_temp = Image.fromarray(image_in)
    # # im = cv2.imread(image_temp)
    # cv2.rectangle(image_temp, box_corners_tablet[3], box_corners_tablet[1], (0, 0, 255), 2)
    # cv2.rectangle(image_temp, box_corners_pp[3], box_corners_pp[1], (0, 255, 0), 2)
    # cv2.rectangle(image_temp, box_corners_robot[3], box_corners_robot[1], (255, 0, 0), 2)
    # print(box_corners_tablet)
    # print(box_corners_pp)
    # print(box_corners_robot)

    # to check if the original gaze end point is in the bbox or not, if not then check intersections for extended gaze
    if inbb(x_left_tab, y_top_tab, x_right_tab, y_bottom_tab, gazex_norm, gazey_norm):
        return 2
    elif inbb(x_left_pp, y_top_pp, x_right_pp, y_bottom_pp, gazex_norm, gazey_norm):
        return 0
    elif inbb(x_left_rbt, y_top_rbt, x_right_rbt, y_bottom_rbt, gazex_norm, gazey_norm):
        return 1
    else:
        # for i in range(4):
        #     x1_t, y1_t = box_corners_tablet[i]
        #     x2_t, y2_t = box_corners_tablet[(i + 1) % 4]  # Next corner (wraps around to the first corner)
        #     x1_p, y1_p = box_corners_pp[i]
        #     x2_p, y2_p = box_corners_pp[(i + 1) % 4]  # Next corner (wraps around to the first corner)
        #     x1_r, y1_r = box_corners_robot[i]
        #     x2_r, y2_r = box_corners_robot[(i + 1) % 4]  # Next corner (wraps around to the first corner)
        #
        #     try:
        #         tablet_edge_slope = abs((y2_t - y1_t) / (x2_t - x1_t))
        #         pp_edge_slope = (y2_p - y1_p) / (x2_p - x1_p)
        #         robot_edge_slope = (y2_r - y1_r) / (x2_r - x1_r)
        #     except ZeroDivisionError:
        #         tablet_edge_slope = 1.0
        #         pp_edge_slope = 1.0
        #         robot_edge_slope = 1.0
        #
        #     if gaze_slope == tablet_edge_slope or gaze_slope == pp_edge_slope or gaze_slope == robot_edge_slope:
        #         continue
        #
        #     if line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_t, y1_t, x2_t, y2_t) is not None:
        #         intersection_x_t, intersection_y_t = line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_t, y1_t, x2_t,
        #                                                                        y2_t)
        #
        #         # Check if the intersection point lies within the range of the line segment
        #         if min(x1_t, x2_t) <= abs(intersection_x_t) <= max(x1_t, x2_t) and min(y1_t, y2_t) <= abs(
        #                 intersection_y_t) <= max(y1_t, y2_t):
        #             intersection_points_tablet.append((intersection_x_t, intersection_y_t))
        #             cv2.circle(image_in, (intersection_x_t, intersection_y_t), 10, (0, 0, 255), 1)
        #
        #     if line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_p, y1_p, x2_p, y2_p) is not None:
        #         intersection_x_p, intersection_y_p = line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_p, y1_p, x2_p,
        #                                                                        y2_p)
        #
        #         # Check if the intersection point lies within the range of the line segment
        #         if min(x1_p, x2_p) <= abs(intersection_x_p) <= max(x1_p, x2_p) and min(y1_p, y2_p) <= abs(
        #                 intersection_y_p) <= max(y1_p, y2_p):
        #             intersection_points_pp.append((intersection_x_p, intersection_y_p))
        #             cv2.circle(image_in, (intersection_x_p, intersection_y_p), 10, (255, 0, 0), 1)
        #
        #     if line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_r, y1_r, x2_r, y2_r) is not None:
        #         intersection_x_r, intersection_y_r = line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_r, y1_r, x2_r,
        #                                                                        y2_r)
        #
        #         # Check if the intersection point lies within the range of the line segment
        #         if min(x1_r, x2_r) <= abs(intersection_x_r) <= max(x1_r, x2_r) and min(y1_r, y2_r) <= abs(
        #                 intersection_y_r) <= max(y1_r, y2_r):
        #             intersection_points_robot.append((intersection_x_r, intersection_y_r))
        #             cv2.circle(image_in, (intersection_x_r, intersection_y_r), 10, (0, 255, 0), 1)

        # check if there is any the intersection point in tablet box
        for i in range(4):
            x1_t, y1_t = box_corners_tablet[i]
            x2_t, y2_t = box_corners_tablet[(i + 1) % 4]  # Next corner (wraps around to the first corner)
            # print((x1_t, y1_t))
            # print((x2_t, y2_t))
            # Check if the line and line segment are parallel
            try:
                # tablet_edge_slope = round((y2_t - y1_t) / (x2_t - x1_t))
                tablet_edge_slope = abs((y2_t - y1_t) / (x2_t - x1_t))
                # print(f"tablet edge slope: {tablet_edge_slope}")
            except ZeroDivisionError:
                tablet_edge_slope = 1.0
                # print(f"tablet edge slope: {tablet_edge_slope}")

            if gaze_slope == tablet_edge_slope:
                continue

            if line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_t, y1_t, x2_t, y2_t) is not None:
                intersection_x_t, intersection_y_t = line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_t, y1_t, x2_t,
                                                                               y2_t)

                # Check if the intersection point lies within the range of the line segment
                if min(x1_t, x2_t) <= abs(intersection_x_t) <= max(x1_t, x2_t) and min(y1_t, y2_t) <= abs(
                        intersection_y_t) <= max(y1_t, y2_t):
                    intersection_points_tablet.append((intersection_x_t, intersection_y_t))
                    cv2.circle(image_in, (intersection_x_t, intersection_y_t), 10, (0, 0, 255), 1)

            # if intersection_points_tablet:
            #     print(f"tablet: {intersection_points_tablet}")
        # return intersection_points_tablet

        # check if there is any the intersection point in pp box
        for i in range(4):
            x1_p, y1_p = box_corners_pp[i]
            x2_p, y2_p = box_corners_pp[(i + 1) % 4]  # Next corner (wraps around to the first corner)

            # Check if the line and line segment are parallel
            try:
                # pp_edge_slope = round((y2_p - y1_p) / (x2_p - x1_p))
                pp_edge_slope = (y2_p - y1_p) / (x2_p - x1_p)
            except ZeroDivisionError:
                pp_edge_slope = 1.0

            if gaze_slope == pp_edge_slope:
                continue

            if line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_p, y1_p, x2_p, y2_p) is not None:
                intersection_x_p, intersection_y_p = line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_p, y1_p, x2_p,
                                                                               y2_p)

                # Check if the intersection point lies within the range of the line segment
                if min(x1_p, x2_p) <= abs(intersection_x_p) <= max(x1_p, x2_p) and min(y1_p, y2_p) <= abs(
                        intersection_y_p) <= max(y1_p, y2_p):
                    intersection_points_pp.append((intersection_x_p, intersection_y_p))
                    cv2.circle(image_in, (intersection_x_p, intersection_y_p), 10, (255, 0, 0), 1)

            # if intersection_points_pp:
            #     print(f"pp: {intersection_points_pp}")
        # return intersection_points_tablet

        # check if there is any the intersection point in robot box
        for i in range(4):
            x1_r, y1_r = box_corners_robot[i]
            x2_r, y2_r = box_corners_robot[(i + 1) % 4]  # Next corner (wraps around to the first corner)

            # Check if the line and line segment are parallel
            try:
                # robot_edge_slope = round((y2_r - y1_r) / (x2_r - x1_r))
                robot_edge_slope = (y2_r - y1_r) / (x2_r - x1_r)
            except ZeroDivisionError:
                robot_edge_slope = 1.0

            if gaze_slope == robot_edge_slope:
                continue

            if line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_r, y1_r, x2_r, y2_r) is not None:
                intersection_x_r, intersection_y_r = line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_r, y1_r, x2_r,
                                                                               y2_r)

                # Check if the intersection point lies within the range of the line segment
                if min(x1_r, x2_r) <= abs(intersection_x_r) <= max(x1_r, x2_r) and min(y1_r, y2_r) <= abs(
                        intersection_y_r) <= max(y1_r, y2_r):
                    intersection_points_robot.append((intersection_x_r, intersection_y_r))
                    cv2.circle(image_in, (intersection_x_r, intersection_y_r), 10, (0, 255, 0), 1)

            # if intersection_points_robot:
            #     print(f"robot: {intersection_points_robot}")
            # return intersection_points_tablet

        # print(f'intersections with tablet are {len(intersection_points_tablet)}')
        # print(f'intersections with pp are {len(intersection_points_pp)}')
        # print(f'intersections with robot are {len(intersection_points_robot)}')

        class_pred = 3
        prop_tab = 0
        prop_pp = 0
        prop_rbt = 0
        max_prop = 0
        if len(intersection_points_tablet) == 2:
            class_pred = 2
            prop_tab = intersection_proportion(x_left_tab, y_top_tab, x_right_tab, y_bottom_tab,
                                               intersection_points_tablet[0][0], intersection_points_tablet[0][1],
                                               intersection_points_tablet[1][0], intersection_points_tablet[1][1])
        elif len(intersection_points_tablet) == 1:
            class_pred = 2
        else:
            pass

        if len(intersection_points_pp) == 2:
            class_pred = 0
            prop_pp = intersection_proportion(x_left_pp, y_top_pp, x_right_pp, y_bottom_pp,
                                              intersection_points_pp[0][0], intersection_points_pp[0][1],
                                              intersection_points_pp[1][0], intersection_points_pp[1][1])
        elif len(intersection_points_pp) == 1:
            class_pred = 0
        else:
            pass

        if len(intersection_points_robot) == 2:
            class_pred = 1
            prop_rbt = intersection_proportion(x_left_rbt, y_top_rbt, x_right_rbt, y_bottom_rbt,
                                               intersection_points_robot[0][0], intersection_points_robot[0][1],
                                               intersection_points_robot[1][0], intersection_points_robot[1][1])
        elif len(intersection_points_tablet) == 1:
            class_pred = 1
        else:
            pass

        # to find the max proportion among the three intersections
        if prop_tab > max_prop:
            max_prop = prop_tab
            class_pred = 2
        if prop_pp > max_prop:
            max_prop = prop_pp
            class_pred = 0
        if prop_rbt > max_prop:
            max_prop = prop_rbt
            class_pred = 1

        # print(max_prop)
        return class_pred


def klm(data):
    """
    This is a filter function using kalman filter to filter out the potential noises.

    Return a matrix table which is similar with the input.
    """
    # Create a simple Kalman filter
    kf = KalmanFilter(dim_x=1, dim_z=1)  # 1-dimensional state and measurement

    # Define the state transition matrix
    kf.F = np.array([[1.]])  # Identity matrix since we're dealing with a single value

    # Define the measurement matrix
    kf.H = np.array([[1.]])  # Identity matrix since we're directly measuring the state

    # Define the process noise covariance
    kf.Q = np.array([[0.05]])  # Adjust the value based on the noise level in the system

    # Define the measurement noise covariance
    kf.R = np.array([[1.]])  # Adjust the value based on the noise level in the measurements

    # Initialize the state and covariance matrix
    kf.x = np.array([[0.]])  # Initial state estimate
    kf.P = np.array([[1.]])  # Initial state covariance

    # Create an array of measurements
    measurements = np.array(data)  # Replace with your own array

    # Perform the filtering
    filtered_values = []
    for measurement in measurements:
        kf.predict()  # Predict the next state
        kf.update(measurement)  # Update the state based on the measurement
        filtered_values.append(kf.x[0, 0])  # Store the filtered value

    # print(filtered_values)
    return filtered_values


def get_largest_face(faces):
    """
    Returns the face closest to the camera (based on size of the face relative to the image size)
    """
    largest_face_idx = 0
    largest_face = 0
    for idx, face in enumerate(faces):
        box, _, _ = face
        x_min = int(box[0])
        if x_min < 0:
            x_min = 0
        y_min = int(box[1])
        if y_min < 0:
            y_min = 0
        x_max = int(box[2])
        y_max = int(box[3])
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        fsize = bbox_width * bbox_height
        if fsize > largest_face:
            largest_face = fsize
            largest_face_idx = idx

    return faces[largest_face_idx]


def run(vid_type):
    """
    - This is the main function to run the whole script.
    - Input "vid_type" to run whether "vis" videos or "invis" videos.
    - The respective output folders will be created automatically if they do not exist.
    - If any of the folders exists, the respective folder of input video will be skipped for this run. (for example,
    if output folder "50001_taskrobotengagement" exists, this video will be skipped and will not update the output)

    Return frames that contain head bounding box, gaze line, objects bounding boxs, intersection points with each box edge.
    And output files contain ["yaw", "pitch", 'm2_class', 'm2prop_class', 'flt_m2_class', 'flt_prop_class', 'face'].
    yaw: original gaze yaw
    pitch: original gaze pitch
    m2_class: the classified class label using [method2] (line-segments intersection without proportion)
    m2prop_class: the classified class label using [method2 + proportion] (line-segments intersection with proportion)
    flt_m2_class: the classified class label using only [method2] and the output is filtered
    flt_prop_class: the classified class label using [method2 + proportion] and the output is filtered
    face: the face status in 0 or 1 (0 is no face shown, 1 is face shown)
    """
    if vid_type == 'vis':
        video_list = vis_video_list
        video_folder = vis_video_folder
        # frame_out_root = vis_frame_out_root
        l2cs_root = vis_l2cs_root
        face_folder_l2cs = './input/face_l2cs/face_vis/'
        face_folder_base = './input/face_baseline/face_vis/'
        result_folder = root + '/pitchjaw/visible_filtered_m2prop/'
        # pred_values_folder = root + 'pitchjaw/visible/'
    elif vid_type == 'invis':
        video_list = invis_video_list
        video_folder = invis_video_folder
        # frame_out_root = invis_frame_out_root
        l2cs_root = invis_l2cs_root
        face_folder_l2cs = './input/face_l2cs/face_invis/'
        face_folder_base = './input/face_baseline/face_invis/'
        result_folder = root + '/pitchjaw/invisible_filtered_m2prop/'
        # pred_values_folder = root + 'pitchjaw/invisible/'
    else:
        video_list = None
        video_folder = None
        # frame_out_root = None
        l2cs_root = None
        face_folder_l2cs = './input/face_l2cs/face_vis/'
        face_folder_base = './input/face_baseline/face_vis/'
        result_folder = root + '/pitchjaw/visible_filtered_m2prop/'
        # pred_values_folder = root + 'pitchjaw/visible/'
        print("Video type error in run()!")

    for video in video_list:
        filename = os.path.splitext(video)[0]

        face_file = filename + '.txt'
        # frame_out_folder = frame_out_root + filename + '/'
        l2cs_out = l2cs_root + filename + '/'
        l2cs_out_temp = l2cs_root + filename + '_temp/'
        # video_name = output_root + filename + '.avi'

        if os.path.exists(l2cs_out):
            print(f'{filename} is passed...')
            continue
        else:
            print(f"{filename} is in processing...")
            os.makedirs(l2cs_out)
            # if not os.path.exists(frame_out_folder):
            #     os.makedirs(frame_out_folder)
            if not os.path.exists(l2cs_out_temp):
                os.makedirs(l2cs_out_temp)
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            # if not os.path.exists(face_folder):
            #     os.makedirs(face_folder)
            # if not os.path.exists(invis_root_video):
            #     os.makedirs(invis_root_video)

            face_l2cs_info = pd.read_csv(os.path.join(face_folder_l2cs, face_file),
                                         names=['file', 'left', 'top', 'right', 'bottom', 'face'])
            face_base_info = pd.read_csv(os.path.join(face_folder_base, face_file),
                                         names=['file', 'left', 'top', 'right', 'bottom', 'face'])

            cap = cv2.VideoCapture(video_folder + video)

            # Check if the file is opened correctly
            if not cap.isOpened():
                raise IOError("Could not read the video file")

            pitch_predicted_ = []
            yaw_predicted_ = []
            gaze_class_list = []
            gaze_class_prop_list = []
            face_status_list = []

            c = 0
            with torch.no_grad():
                while True:
                    success, frame = cap.read()
                    if not success:
                        print('All frames are processed')
                        break
                    if frame is not None:
                        # cv2.imwrite(frame_out_folder + "%05d.jpg" % c, frame)
                        start_fps = time.time()

                        faces_base = face_base_info[face_base_info['file'] == "%05d.jpg" % c].iloc[0]
                        # print(f'baseline_face: img = {"%05d.jpg" % c}')
                        if faces_base['face']:
                            face_status = 1
                            x_min_base = int(faces_base['left'])
                            y_min_base = int(faces_base['top'])
                            x_max_base = int(faces_base['right'])
                            y_max_base = int(faces_base['bottom'])

                            # # # this block is for l2cs to detect faces.
                            # faces = detector(frame)
                            # faces = [face for face in faces if face[2] >= 0.85]
                            # if len(faces) > 0:
                            #     face_status = True
                            #     # Assume the biggest face in scene is the child's face. This is the only face relevant
                            #     # for the gaze estimation
                            #     box, landmarks, score = get_largest_face(faces)
                            #     x_min = int(box[0])
                            #     if x_min < 0:
                            #         x_min = 0
                            #     y_min = int(box[1])
                            #     if y_min < 0:
                            #         y_min = 0
                            #     x_max = int(box[2])
                            #     y_max = int(box[3])
                            #     bbox_width = x_max - x_min
                            #     bbox_height = y_max - y_min
                            #
                            #     face_box = [x_min, y_min, x_max, y_max]

                            # Crop image
                            img = frame[y_min_base:y_max_base, x_min_base:x_max_base]
                            img = cv2.resize(img, (224, 224))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            im_pil = Image.fromarray(img)
                            img = transformations(im_pil)
                            img = Variable(img).cuda(gpu)
                            img = img.unsqueeze(0)

                            # gaze prediction
                            gaze_pitch, gaze_yaw = model(img)

                            pitch_predicted = softmax(gaze_pitch)
                            yaw_predicted = softmax(gaze_yaw)

                            # Get continuous predictions in degrees.
                            pitch_predicted = torch.sum(
                                pitch_predicted.data[0] * idx_tensor) * 4 - 180
                            yaw_predicted = torch.sum(
                                yaw_predicted.data[0] * idx_tensor) * 4 - 180

                            pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
                            yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0
                        else:
                            face_status = 0
                            pitch_predicted = NOFACE
                            yaw_predicted = NOFACE
                            x_min_base, y_min_base, x_max_base, y_max_base = 0, 0, 0, 0

                        bbox_width_base = x_max_base - x_min_base
                        bbox_height_base = y_max_base - y_min_base

                        pitch_predicted_.append(pitch_predicted)
                        yaw_predicted_.append(yaw_predicted)

                        draw_gaze_norm(x_min_base, y_min_base, bbox_width_base, bbox_height_base, frame,
                                       (pitch_predicted, yaw_predicted), color=(0, 0, 255))  # l2cs output
                        draw_gaze_extended(x_min_base, y_min_base, bbox_width_base, bbox_height_base, frame,
                                           (pitch_predicted, yaw_predicted), color=(0, 0, 255))  # l2cs output
                        cv2.rectangle(frame, (x_min_base, y_min_base), (x_max_base, y_max_base), (0, 0, 255), 2)

                        gaze_class = classify_line_box_intersection(x_min_base, y_min_base, bbox_width_base,
                                                                    bbox_height_base,
                                                                    frame, pitch_predicted_[-1], yaw_predicted_[-1],
                                                                    face_status,
                                                                    vid_type, filename)
                        gaze_class_prop = classify_line_box_intersection_proportion(x_min_base, y_min_base,
                                                                                    bbox_width_base,
                                                                                    bbox_height_base, frame,
                                                                                    pitch_predicted_[-1],
                                                                                    yaw_predicted_[-1],
                                                                                    face_status, vid_type, filename)

                        # # # this block is to write face info into txt files
                        # with open(os.path.join(face_folder, face_file), 'a') as f:
                        #     f.write("%05d.jpg" % c)
                        #     f.write(',')
                        #     f.write(','.join(str(b) for b in face_box))
                        #     f.write(',')
                        #     f.write(str(face_status))
                        #     f.write('\n')

                        gaze_class_list.append(gaze_class)
                        gaze_class_prop_list.append(gaze_class_prop)
                        face_status_list.append(face_status)

                        # cv2.putText(frame, '(Face: {})'.format(face_status), (5, 20),
                        #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
                        myFPS = 1.0 / (time.time() - start_fps)
                        # print(f"myFPS: {myFPS}")
                        cv2.putText(frame, 'class: {} cls_p: {} (face: {})'.format(get_classname(gaze_class),
                                                                                   get_classname(gaze_class_prop),
                                                                                   face_status), (10, 20),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)

                        cv2.imshow("Demo", frame)
                        # cv2.imwrite(invis_root + filename + "/" + "%05d.jpg" % count, frame)
                        cv2.imwrite(l2cs_out_temp + "%05d.jpg" % c, frame)  # here
                        c += 1
                        if cv2.waitKey(1) & 0xFF == 27:
                            break
                    else:
                        print(f'image{c} is None! Skip it...')
                        pass

                print(f"l2cs prediction for {filename} is done!")
                print(f"Applying filter on {filename}...")

                pitch_filtered = klm(np.array(pitch_predicted_))
                yaw_filtered = klm(np.array(yaw_predicted_))

                gaze_class_filtered = []
                gaze_class_prop_filtered = []
                # face_list_filtered = []
                count = 0

                images = [img for img in os.listdir(l2cs_out_temp) if
                          (img.endswith(".jpg") and img.startswith("0"))]
                images = sorted(images)

                for img in images:
                    frame = cv2.imread(os.path.join(l2cs_out_temp, img))

                    faces_l2cs = face_l2cs_info[face_l2cs_info['file'] == img].iloc[0]
                    # print(f'l2cs_face: img = {img}')
                    if faces_l2cs['face']:
                        x_min_l2cs = int(faces_l2cs['left'])
                        y_min_l2cs = int(faces_l2cs['top'])
                        x_max_l2cs = int(faces_l2cs['right'])
                        y_max_l2cs = int(faces_l2cs['bottom'])
                        face_status_f = 1
                    else:
                        x_min_l2cs, y_min_l2cs, x_max_l2cs, y_max_l2cs = 0, 0, 0, 0
                        face_status_f = 0

                    bbox_width_l2cs = x_max_l2cs - x_min_l2cs
                    bbox_height_l2cs = y_max_l2cs - y_min_l2cs

                    draw_gaze_norm(x_min_l2cs, y_min_l2cs, bbox_width_l2cs, bbox_height_l2cs, frame,
                                   (pitch_filtered[count], yaw_filtered[count]), color=(0, 255, 0))
                    draw_gaze_extended(x_min_l2cs, y_min_l2cs, bbox_width_l2cs, bbox_height_l2cs, frame,
                                       (pitch_filtered[count], yaw_filtered[count]), color=(0, 255, 0))
                    cv2.rectangle(frame, (x_min_l2cs, y_min_l2cs), (x_max_l2cs, y_max_l2cs), (0, 255, 0),
                                  2)  # using l2cs' face
                    gaze_class_f = classify_line_box_intersection(x_min_l2cs, y_min_l2cs, bbox_width_l2cs,
                                                                  bbox_height_l2cs, frame,
                                                                  pitch_filtered[count], yaw_filtered[count],
                                                                  face_status_f,
                                                                  vid_type, filename)
                    gaze_class_prop_f = classify_line_box_intersection_proportion(x_min_l2cs, y_min_l2cs,
                                                                                  bbox_width_l2cs, bbox_height_l2cs,
                                                                                  frame,
                                                                                  pitch_filtered[count],
                                                                                  yaw_filtered[count],
                                                                                  face_status_f,
                                                                                  vid_type, filename)

                    gaze_class_filtered.append(gaze_class_f)
                    gaze_class_prop_filtered.append(gaze_class_prop_f)
                    # face_list_filtered.append(face_status_f)

                    cv2.putText(frame, '(flt: {} flt_p: {})'.format(get_classname(gaze_class_f),
                                                                    get_classname(gaze_class_prop_f)),
                                (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow("Demo", frame)
                    cv2.imwrite(l2cs_out + "%05d.jpg" % count, frame)  # here
                    count += 1
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                dataframe_filter = pd.DataFrame(
                    data=np.concatenate(
                        [np.array(pitch_filtered, ndmin=2), np.array(yaw_filtered, ndmin=2),
                         np.array(gaze_class_list, ndmin=2), np.array(gaze_class_prop_list, ndmin=2),
                         np.array(gaze_class_filtered, ndmin=2), np.array(gaze_class_prop_filtered, ndmin=2),
                         np.array(face_status_list, ndmin=2)]).T,
                    columns=["yaw", "pitch", 'm2_class', 'm2prop_class', 'flt_m2_class', 'flt_prop_class', 'face'])
                dataframe_filter.to_csv(result_folder + filename + '.csv', index=False)  # here

                cv2.destroyAllWindows()
                cap.release()
                print(f"{filename} is done!")
    print("--- Complete excecution = %s seconds ---" % (time.time() - start_time))
    print("All done!")


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()

    cudnn.enabled = True
    arch = args.arch
    batch_size = 16
    # cam = args.cam_id
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    snapshot_path = args.snapshot

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    model = getArch(arch, 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    x = 0

    # ext = '.mp4'
    # filename = 'Proefpersoon22016_Sessie1'
    # file = filename + ext
    # cap = cv2.VideoCapture('testvid/' + file)

    vid_type = 'vis'
    run(vid_type)

    vid_type = 'invis'
    run(vid_type)
