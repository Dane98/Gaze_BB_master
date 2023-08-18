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
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from face_detection import RetinaFace
from model import L2CS
from readbbtxt import readbbtxt as readvis
from readbbtxt_inv import readbbtxt as readinv

datafolder = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/data_fin/'
datafile = 'pixel_position_vis.txt'

data = readvis(datafolder + datafile)
# remove .png extension from filenames
data['file'] = data['file'].apply(lambda x: x[:-4])


def inbb(tlx, tly, brx, bry, gazex, gazey):
    return tlx <= gazex <= tlx + (brx - tlx) and tly <= gazey <= tly + (bry - tly)


def get_classname(id):
    if id == 0:
        return 'Pen and paper'
    elif id == 1:
        return 'robot'
    elif id == 2:
        return 'tablet'
    elif id == 3:
        return 'elsewhere'
    elif id == 4:
        return 'unknown'


def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Returns the coordinates px, py. p is the intersection of the lines
    defined by ((x1, y1), (x2, y2)) and ((x3, y3), (x4, y4))
    """
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return int(px), int(py)


def line_segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
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


def classify_line_box_intersection(a, b, c, d, pitch_pred, jaw_pred):
    intersection_points_tablet = []
    intersection_points_pp = []
    intersection_points_robot = []
    # box_corners = [(1, 1), (3, 1), (3, 3), (1, 3)]

    if pitch_pred == 42 and jaw_pred == 42:
        return 4

    h, w = 720, 1280
    pos = (int(a + c / 2.0), int(b + d / 2.0))
    gazex = round(pitch_pred)
    gazey = round(jaw_pred)

    image_edges = [(0, 0), (w, 0), (w, h), (0, h)]
    x1, y1 = round(pos[0]), round(pos[1])
    x2, y2 = gazex, gazey

    try:
        gaze_slope = round((gazey - pos[1]) / (gazex - pos[0]))
        # print(f"gaze slope: {gaze_slope}")
    except ZeroDivisionError:
        gaze_slope = 1.0
    gaze_intercept = round(pos[1] - gaze_slope * pos[0])

    for edge in range(4):
        x3, y3 = image_edges[edge]
        x4, y4 = image_edges[(edge + 1) % 4]  # Next corner (wraps around to the first corner)

        if line_segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4) is not None:
            gazex, gazey = line_segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
        else:
            gazex, gazey = x2, y2

    rec = data[data['file'] == 'Proefpersoon22016_Sessie1'].iloc[0]

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

        if line_segment_intersection(x1, y1, x2, y2, x1_t, y1_t, x2_t, y2_t) is not None:
            intersection_x_t, intersection_y_t = line_segment_intersection(x1, y1, x2, y2, x1_t, y1_t, x2_t, y2_t)

            # Check if the intersection point lies within the range of the line segment
            if min(x1_t, x2_t) <= abs(intersection_x_t) <= max(x1_t, x2_t) and min(y1_t, y2_t) <= abs(
                    intersection_y_t) <= max(y1_t, y2_t):
                intersection_points_tablet.append((intersection_x_t, intersection_y_t))

        if intersection_points_tablet:
            print(f"tablet: {intersection_points_tablet}")
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

        if line_segment_intersection(x1, y1, x2, y2, x1_p, y1_p, x2_p, y2_p) is not None:
            intersection_x_p, intersection_y_p = line_segment_intersection(x1, y1, x2, y2, x1_p, y1_p, x2_p, y2_p)

            # Check if the intersection point lies within the range of the line segment
            if min(x1_p, x2_p) <= abs(intersection_x_p) <= max(x1_p, x2_p) and min(y1_p, y2_p) <= abs(
                    intersection_y_p) <= max(y1_p, y2_p):
                intersection_points_pp.append((intersection_x_p, intersection_y_p))

        if intersection_points_pp:
            print(f"pp: {intersection_points_pp}")
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

        if line_segment_intersection(x1, y1, x2, y2, x1_r, y1_r, x2_r, y2_r) is not None:
            intersection_x_r, intersection_y_r = line_segment_intersection(x1, y1, x2, y2, x1_r, y1_r, x2_r, y2_r)

            # Check if the intersection point lies within the range of the line segment
            if min(x1_r, x2_r) <= abs(intersection_x_r) <= max(x1_r, x2_r) and min(y1_r, y2_r) <= abs(
                    intersection_y_r) <= max(y1_r, y2_r):
                intersection_points_robot.append((intersection_x_r, intersection_y_r))

        if intersection_points_robot:
            print(f"robot: {intersection_points_robot}")
            # return intersection_points_tablet
    if intersection_points_tablet:  # return 2 if list of intersection points is not empty
        return 2
    elif intersection_points_pp:
        return 0
    elif intersection_points_robot:
        return 1
    else:
        return 3


if __name__ == '__main__':
    gaze_class_ = []

    datafile = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/test_py.csv'
    data = pd.read_csv(datafile)
    pitch = data['pitch']
    yaw = data['yaw']

    face_file = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/Proefpersoon22016_Sessie1.txt'
    columns = ['frame', 'left', 'top', 'right', 'bottom']
    face_data = pd.read_csv(face_file, names=columns, index_col=0)

    x_min = face_data[0]
    y_min = face_data[1]
    x_max = face_data[2]
    y_max = face_data[3]

    for j in range(len(data.index)):
        if not face_data.loc[j, 'left'] == 'None':
            x_min[j]
            # pos = (int(x_min[i] + x_max[i] / 2.0), int(y_min[i] + y_max[i] / 2.0))
            gaze_class = classify_line_box_intersection(x_min[j], y_min[j], x_max[j], y_max[j], pitch[j], yaw[j])
            gaze_class_.append(gaze_class)

    column_values = pd.Series(gaze_class_)
    data.insert(loc=0, column='class', value=column_values)
