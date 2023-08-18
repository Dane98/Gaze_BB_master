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
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from face_detection import RetinaFace
from model import L2CS
from readbbtxt import readbbtxt

NOFACE = 42

datafolder = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/data_fin/'
datafile = 'pixel_position_vis.txt'
root = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/'
output_root = root + 'frames/Saved_frame/'
vis_frame_out_root = output_root + 'fig_vis/'
vis_l2cs_root = output_root + 'fig_vis_l2cs/'
vis_root_video = root + 'visible_bb_l2cs/'
video_folder = root + 'visible_bb/'

data = readbbtxt(datafolder + datafile)
# remove .png extension from filenames
data['file'] = data['file'].apply(lambda x: x[:-4])

# video_file_list = []
video_list = [vid for vid in os.listdir(video_folder) if (vid.endswith(".MP4") or vid.endswith(".mp4"))]
video_list = sorted(video_list)
print(video_list[0:3])

# for vid_file in video_list:
#     file_name = os.path.basename(vid_file)
#     video_file_list.append(os.path.splitext(file_name)[0])


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
        return 'Pen and paper'
    elif id == 1:
        return 'robot'
    elif id == 2:
        return 'tablet'
    elif id == 3:
        return 'elsewhere'
    elif id == 4:
        return 'unknown'


# https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

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


def classify_line_box_intersection(a, b, c, d, image_in, pitch_pred, jaw_pred):
    intersection_points_tablet = []
    intersection_points_pp = []
    intersection_points_robot = []
    # box_corners = [(1, 1), (3, 1), (3, 3), (1, 3)]

    global box_corners_tablet
    global box_corners_pp
    global box_corners_robot
    global intersection_x_t, intersection_y_t
    global intersection_x_p, intersection_y_p
    global intersection_x_r, intersection_y_r

    if pitch_pred == NOFACE and jaw_pred == NOFACE:
        return 4

    (h, w) = image_in.shape[:2]
    length = w * 999
    # length = w /2
    pos = ((a + c / 2.0), (b + d / 2.0))
    # print(f'pos is {pos}')
    dx = -length * np.sin(pitch_pred) * np.cos(jaw_pred)
    dy = -length * np.sin(jaw_pred)
    gazex = round(pos[0] + dx)
    gazey = round(pos[1] + dy)
    # print(f'gazex, gazey = {(gazex, gazey)}')

    try:
        gaze_slope = float((gazey - pos[1]) / (gazex - pos[0]))
        # print(f"gaze slope: {gaze_slope}")
    except ZeroDivisionError:
        gaze_slope = 1.0
    gaze_intercept = float(pos[1] - gaze_slope * pos[0])
    print(f"gaze intercept: {gaze_intercept}")

    image_edges = [(0, 0), (w, 0), (w, h), (0, h)]
    x1, y1 = round(pos[0]), round(pos[1])
    x2, y2 = gazex, gazey
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

    # # Calculate the y-values at the x-coordinates of the line segment's endpoints
    # y1_intersect = gaze_slope * x1 + gaze_intercept
    # y2_intersect = gaze_slope * x2 + gaze_intercept
    #
    # # Check if the y-values lie within the range of the line segment
    # if (y1_intersect * y2_intersect <= 0) or (y1_intersect == 0) or (y2_intersect == 0):
    #     return True
    # else:
    #     return False

    rec = data[data['file'] == filename].iloc[0]

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

    # roi_tablet = image_in[y_top_tab:y_bottom_tab, x_left_tab:x_right_tab]
    # roi_pp = image_in[y_top_pp:y_bottom_pp, x_left_pp:x_right_pp]
    # roi_robot = image_in[y_top_rbt:y_bottom_rbt, x_left_rbt:x_right_rbt]

    # roi_tablet = Image.fromarray(roi_tablet)
    # roi_tablet.show()
    # roi_pp = Image.fromarray(roi_pp)
    # roi_pp.show()
    # roi_robot = Image.fromarray(roi_robot)
    # roi_robot.show()

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

            # Calculate the intersection point
            # try:
            # intersection_x_t = int((gaze_intercept - y1_t + tablet_edge_slope * x1_t) / (tablet_edge_slope - gaze_slope))
            # intersection_x_t = (gaze_intercept - y1_t + tablet_edge_slope * x1_t) / (tablet_edge_slope - gaze_slope)
            # print(f"int x: {intersection_x_t}")
            # except ZeroDivisionError:
            #     intersection_x_t = (gaze_intercept - y1_t + tablet_edge_slope * x1_t) / (tablet_edge_slope - gaze_slope)
            #     print("DZ error: tablet")
            # intersection_y_t = int(gaze_slope * intersection_x_t + gaze_intercept)
            # intersection_y_t = gaze_slope * intersection_x_t + gaze_intercept
            # print(f"int y: {intersection_y_t}")

            # Check if the intersection point lies within the range of the line segment
            if min(x1_t, x2_t) <= abs(intersection_x_t) <= max(x1_t, x2_t) and min(y1_t, y2_t) <= abs(
                    intersection_y_t) <= max(y1_t, y2_t):
                intersection_points_tablet.append((intersection_x_t, intersection_y_t))
                cv2.circle(frame, (intersection_x_t, intersection_y_t), 10, (0, 0, 255), 1)

        # else:
        #     print(f"there is no intersection with tablet line {i}")
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
            # Calculate the intersection point
            # try:
            # intersection_x_p = int((gaze_intercept - y1_p + pp_edge_slope * x1_p) / (pp_edge_slope - gaze_slope))
            # intersection_x_p = (gaze_intercept - y1_t + tablet_edge_slope * x1_t) / (tablet_edge_slope - gaze_slope)
            # except ZeroDivisionError:
            #     intersection_x_p = 1
            # intersection_y_p = int(gaze_slope * intersection_x_p + gaze_intercept)
            # intersection_y_p = gaze_slope * intersection_x_p + gaze_intercept

            # Check if the intersection point lies within the range of the line segment
            if min(x1_p, x2_p) <= abs(intersection_x_p) <= max(x1_p, x2_p) and min(y1_p, y2_p) <= abs(
                    intersection_y_p) <= max(y1_p, y2_p):
                intersection_points_pp.append((intersection_x_p, intersection_y_p))
                cv2.circle(frame, (intersection_x_p, intersection_y_p), 10, (255, 0, 0), 1)
        # else:
        #     print(f"there is not intersection with pp line {i}")
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
            # Calculate the intersection point
            # try:
            # intersection_x_r = int((gaze_intercept - y1_r + robot_edge_slope * x1_r) / (robot_edge_slope - gaze_slope))
            # intersection_x_r = (gaze_intercept - y1_r + robot_edge_slope * x1_r) / (robot_edge_slope - gaze_slope)
            # except ZeroDivisionError:
            #     intersection_x_r = 1
            # intersection_y_r = int(gaze_slope * intersection_x_r + gaze_intercept)
            # intersection_y_r = gaze_slope * intersection_x_r + gaze_intercept

            # Check if the intersection point lies within the range of the line segment
            if min(x1_r, x2_r) <= abs(intersection_x_r) <= max(x1_r, x2_r) and min(y1_r, y2_r) <= abs(
                    intersection_y_r) <= max(y1_r, y2_r):
                intersection_points_robot.append((intersection_x_r, intersection_y_r))
                cv2.circle(frame, (intersection_x_r, intersection_y_r), 10, (0, 255, 0), 1)
        # else:
        #     print(f"there is not intersection with robot line {i}")
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


def classify_gaze(a, b, c, d, image_in, pitch_pred, jaw_pred):
    """
    returns 
    0 : pen & paper
    1 : robot
    2 : tablet
    3 : elsewhere
    4 : unknown
    """

    if pitch_pred == NOFACE and jaw_pred == NOFACE:
        return 4

    (h, w) = image_in.shape[:2]
    length = w * 999  # extend the length of the gaze arrow
    pos = (int(a + c / 2.0), int(b + d / 2.0))
    dx = -length * np.sin(pitch_pred) * np.cos(jaw_pred)
    dy = -length * np.sin(jaw_pred)
    extended_gazex = round(pos[0] + dx)
    extended_gazey = round(pos[1] + dy)

    delta_x = extended_gazex - pitch_pred
    delta_y = extended_gazey - jaw_pred
    try:
        gaze_slope = delta_y / delta_x
    except ZeroDivisionError:
        gaze_slope = 1.0

    # Check if gaze vector leaves the image
    if extended_gazex < 0 or extended_gazex > w or extended_gazey < 0 or extended_gazey > h:
        # Do line intersection to find at which point the gaze vector line
        # leaves the image.
        x3, y3, x4, y4 = 0, 0, w, 0

        y3 = extended_gazey if 0 <= extended_gazey <= h else 0 if extended_gazey < 0 else h
        y4 = y3

        x1, y1 = pos[0], pos[1]
        x2, y2 = extended_gazex, extended_gazey

        extended_gazex, extended_gazey = line_intersect(x1, y1, x2, y2, x3, y3, x4, y4)

    rec = data[data['file'] == filename].iloc[0]
    # Check small objects first because if e.g. tablet bb intersects with bb of robot and
    # gaze is towards the parts of intersection, changes are higher the child is indeed looking at the
    # smaller bb, thus the tablet.

    # images = [img for img in os.listdir(frame_out_folder) if
    #           (img.endswith(".jpg") and img.startswith("00"))]
    # images = sorted(images)
    # print(images[0:5])
    # frame_out = cv2.imread(os.path.join(frame_out_folder, images[0]))
    # hh, ww, ll = frame_out.shape
    # for image in images:
    #     # print(image)
    #     # image = cv2.imread(os.path.join(image_folder, "frame" + str(image) + ".jpg"))
    #     # print(image[2:])
    #     image = cv2.imread(os.path.join(frame_out_folder, image))

    # image_in = image_in[y_min:y_max, x_min:x_max]
    # im_pil = Image.fromarray(image_in)
    # image_in = transformations(im_pil)
    # image_in = Variable(image_in).cuda(gpu)
    # image_in = image_in.unsqueeze(0)
    # image = cv2.imread(image_in)
    gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

    roi_tablet = gray[int(rec['tl_tablet_y'] * h):int(rec['br_tablet_y'] * h),
                 int(rec['tl_tablet_x'] * w):int(rec['br_tablet_x'] * w)]
    roi_pp = gray[int(rec['tl_pp_y'] * h):int(rec['br_pp_y'] * h), int(rec['tl_pp_x'] * w):int(rec['br_pp_x'] * w)]
    roi_robot = gray[int(rec['tl_robot_y'] * h):int(rec['br_robot_y'] * h),
                int(rec['tl_robot_x'] * w):int(rec['br_robot_x'] * w)]

    edges_tablet = cv2.Canny(roi_tablet, threshold1=50, threshold2=150)
    edges_pp = cv2.Canny(roi_pp, threshold1=50, threshold2=150)
    edges_robot = cv2.Canny(roi_robot, threshold1=50, threshold2=150)

    lines_tablet = cv2.HoughLinesP(edges_tablet, rho=1, theta=np.pi / 180, threshold=100, minLineLength=5,
                                   maxLineGap=10)
    lines_pp = cv2.HoughLinesP(edges_pp, rho=1, theta=np.pi / 180, threshold=100, minLineLength=5, maxLineGap=10)
    lines_robot = cv2.HoughLinesP(edges_robot, rho=1, theta=np.pi / 180, threshold=100, minLineLength=5, maxLineGap=10)

    if lines_tablet is not None:
        for line in lines_tablet:
            x2, y2, x1, y1 = line[0]
            line_slope = abs((y2 - y1) / (x2 - x1))
            if line_slope == gaze_slope:
                print(f"tablet: {line_slope}")
                break
            # break
            # else:
            #     print(f"tablet not equ: {line_slope}")
            #     continue
        return 2
    elif lines_pp is not None:
        for line in lines_pp:
            x2, y2, x1, y1 = line[0]
            line_slope = abs((y2 - y1) / (x2 - x1))
            if line_slope == gaze_slope:
                print(f"pp: {line_slope}")
                break
            # break
            # else:
            #     print(f"pp not equ: {lines_pp}")
            #     continue
        return 0
    elif lines_robot is not None:
        for line in lines_robot:
            x2, y2, x1, y1 = line[0]
            line_slope = abs((y2 - y1) / (x2 - x1))
            if line_slope == gaze_slope:
                print(f"robot: {line_slope}")
                break
            # break
            # else:
            #     print(f"robot not equ: {lines_robot}")
            #     continue
        return 1
    else:
        return 3


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


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()

    cudnn.enabled = True
    arch = args.arch
    batch_size = 1
    # cam = args.cam_id
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    snapshot_path = args.snapshot

    # processing from local
    # image_folder = './output/test_bb_l2cs/'
    # outputname = 'output/test_bb_l2cs/'

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

    for video in video_list:
        filename = os.path.splitext(video)[0]
        print(f"filename: {filename}")

        frame_out_folder = vis_frame_out_root + filename + '/'
        vis_out_l2cs = vis_l2cs_root + filename + '/'
        video_name = vis_root_video + filename + '.avi'

        if not os.path.exists(frame_out_folder):
            os.makedirs(frame_out_folder)
        if not os.path.exists(vis_out_l2cs):
            os.makedirs(vis_out_l2cs)
        if not os.path.exists(vis_root_video):
            os.makedirs(vis_root_video)

        cap = cv2.VideoCapture(video_folder + video)

        # Check if the file is opened correctly
        if not cap.isOpened():
            raise IOError("Could not read the video file")

        pitch_predicted_ = []
        yaw_predicted_ = []
        gaze_class_ = []

        count = 0
        # img_list = []
        with torch.no_grad():
            while True:
                success, frame = cap.read()

                # cv2.imwrite(frame_out_folder + "%05d.jpg" % count, frame)
                # if cv2.waitKey(1) & 0xFF == 27:
                #     break

                if not success:
                    print('All frames are processed')
                    break
                start_fps = time.time()

                faces = detector(frame)

                faces = [face for face in faces if face[2] >= 0.95]

                if len(faces) > 0:
                    # Assume the biggest face in scene is the child's face. This is the only face relevant
                    # for the gaze estimation
                    box, landmarks, score = get_largest_face(faces)
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

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
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

                    pitch_predicted_.append(pitch_predicted)
                    yaw_predicted_.append(yaw_predicted)

                    draw_gaze(x_min, y_min, bbox_width, bbox_height, frame,
                              (pitch_predicted, yaw_predicted), color=(0, 0, 255))
                    cv2.rectangle(frame, (x_min, y_min),
                                  (x_max, y_max), (0, 255, 0), 1)
                    gaze_class = classify_line_box_intersection(x_min, y_min, bbox_width, bbox_height, frame,
                                                                pitch_predicted_[-1], yaw_predicted_[-1])

                    # cv2.circle(frame, (intersection_x_t, intersection_y_t), 10, (0,0,255), 1)
                    # cv2.circle(frame, (intersection_x_p, intersection_y_p), 10, (255, 0, 0), 1)
                    # cv2.circle(frame, (intersection_x_r, intersection_y_r), 10, (0, 255, 0), 1)
                    (h, w) = frame.shape[:2]
                    # frame_in = frame[y_min:y_max, x_min:x_max]
                    # im_pil = Image.fromarray(frame_in)
                    # # image_in = transformations(im_pil)
                    # # image_in = Variable(image_in).cuda(gpu)
                    # # image_in = image_in.unsqueeze(0)
                    # im_pil.save(frame_out_folder + "%05d.jpg" % count)
                    # # cv2.imwrite(frame_out_folder + "%05d.jpg" % count, frame)
                    # images = [img for img in os.listdir(frame_out_folder) if
                    #           (img.endswith(".jpg") and img.startswith("00"))]  # from pepper experiment
                    # images = sorted(images)

                else:
                    # No face detected, pitch, jaw annotated with 42, 42 to specify that
                    pitch_predicted_.append(NOFACE)
                    yaw_predicted_.append(NOFACE)
                    gaze_class = classify_line_box_intersection(0, 0, 0, 0, frame, pitch_predicted_[-1],
                                                                yaw_predicted_[-1])

                gaze_class_.append(gaze_class)
                myFPS = 1.0 / (time.time() - start_fps)
                # print(f"myFPS: {myFPS}")
                # cv2.putText(frame, 'FPS: {:.1f}'.format(
                #     myFPS), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
                # cv2.putText(frame, 'FPS: {} (class {})'.format(get_classname(gaze_class), gaze_class), (10, 20),
                # cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, 'FPS: {:.1f} (class {})'.format(myFPS, get_classname(gaze_class)), (10, 20),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

                # cv2.rectangle(frame, box_corners_tablet[3], box_corners_tablet[1], (0, 0, 255), 2)
                # cv2.rectangle(frame, box_corners_pp[3], box_corners_pp[1], (255, 0, 0), 2)
                # cv2.rectangle(frame, box_corners_robot[3], box_corners_robot[1], (0, 255, 0), 2)

                # cv2.circle(frame, (), radius, color, thickness)

                cv2.imshow("Demo", frame)
                cv2.imwrite(vis_out_l2cs + "%05d.jpg" % count, frame)
                count += 1
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            images = [img for img in os.listdir(vis_out_l2cs) if
                      (img.endswith(".jpg") and img.startswith("00"))]
            images = sorted(images)
            # # print(images[0:5])
            frame_out = cv2.imread(os.path.join(vis_out_l2cs, images[0]))
            hh, ww, ll = frame_out.shape
            video = cv2.VideoWriter(video_name, 0, 10, (ww, hh))

            for im in images:
                img_out = cv2.imread(os.path.join(vis_out_l2cs, im))
                video.write(img_out)

            # from time import sleep
            # sleep(0.2)

            dataframe = pd.DataFrame(
                data=np.concatenate(
                    [np.array(pitch_predicted_, ndmin=2), np.array(yaw_predicted_, ndmin=2),
                     np.array(gaze_class_, ndmin=2)]).T,
                columns=["yaw", "pitch", 'class'])
            dataframe.to_csv(root + 'pitchjaw/visible/' + filename + '.csv', index=False)
            cv2.destroyAllWindows()
            video.release()

            print("--- Complete excecution = %s seconds ---" % (time.time() - start_time))
    print("All done!")
