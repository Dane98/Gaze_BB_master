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

import pandas as pd
from readbbtxt_inv import readbbtxt

import cv2

NOFACE = 42

datafolder = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/data_fin/'
datafile = 'pixel_position_invis_new.txt'
vis_root = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/frames/Saved_frame/fig_vis_l2cs'
invis_root = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/frames/Saved_frame/fig_invis_l2cs/'
vis_root_video = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/visible_bb_l2cs/'
invis_root_video = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/invisible_bb_l2cs/'

data = readbbtxt(datafolder + datafile)
# remove .png extension from filenames
data['file'] = data['file'].apply(lambda x: x[:-4])

filename = '51016_sessie3_taskrobotEngagement'
ext = '.MP4'
file = filename + ext


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
    length = w * 999
    pos = (int(a + c / 2.0), int(b + d / 2.0))
    dx = -length * np.sin(pitch_pred) * np.cos(jaw_pred)
    dy = -length * np.sin(jaw_pred)
    gazex = round(pos[0] + dx)
    gazey = round(pos[1] + dy)
    print(f"gazex: {gazex}, gazey: {gazey}")

    # Check if gaze vector leaves the image
    if gazex < 0 or gazex > w or gazey < 0 or gazey > h:
        # Do line intersection to find at which point the gaze vector line
        # leaves the image.
        x3, y3, x4, y4 = 0, 0, w, 0

        y3 = gazey if 0 <= gazey <= h else 0 if gazey < 0 else h
        y4 = y3

        x1, y1 = pos[0], pos[1]
        x2, y2 = gazex, gazey

        gazex, gazey = line_intersect(x1, y1, x2, y2, x3, y3, x4, y4)
        print(f"outside gazex: {gazex}, gazey: {gazey}")

    rec = data[data['file'] == filename].iloc[0]
    # Check small objects first because if e.g. tablet bb intersects with bb of robot and
    # gaze is towards the parts of intersection, changes are higher the child is indeed looking at the
    # smaller bb, thus the tablet.

    if inbb(int(rec['tl_tablet_x'] * w), int(rec['tl_tablet_y'] * h), int(rec['br_tablet_x'] * w),
            int(rec['br_tablet_y'] * h), gazex, gazey):
        return 2
    elif inbb(int(rec['tl_pp_x'] * w), int(rec['tl_pp_y'] * h), int(rec['br_pp_x'] * w), int(rec['br_pp_y'] * h), gazex,
              gazey):
        return 0
    elif inbb(int(rec['tl_robot_x'] * w), int(rec['tl_robot_y'] * h), int(rec['br_robot_x'] * w),
              int(rec['br_robot_y'] * h), gazex, gazey):
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
    args = parse_args()

    cudnn.enabled = True
    arch = args.arch
    batch_size = 1
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

    cap = cv2.VideoCapture('testvid/' + file)

    # Check if the file is opened correctly
    if not cap.isOpened():
        raise IOError("Could not read the video file")

    pitch_predicted_ = []
    yaw_predicted_ = []
    gaze_class_ = []

    count = 0
    with torch.no_grad():
        while True:
            success, frame = cap.read()
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
                gaze_class = classify_gaze(x_min, y_min, bbox_width, bbox_height, frame, pitch_predicted_[-1],
                                           yaw_predicted_[-1])
            else:
                # No face detected, pitch, jaw annotated with 42, 42 to specify that
                pitch_predicted_.append(NOFACE)
                yaw_predicted_.append(NOFACE)
                gaze_class = classify_gaze(0, 0, 0, 0, frame, pitch_predicted_[-1], yaw_predicted_[-1])

            gaze_class_.append(gaze_class)
            myFPS = 1.0 / (time.time() - start_fps)
            # cv2.putText(frame, 'FPS: {:.1f}'.format(
            #     myFPS), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'FPS: {} (class {})'.format(
                get_classname(gaze_class), gaze_class), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1,
                        cv2.LINE_AA)

            cv2.imshow("Demo", frame)
            # cv2.imwrite(invis_root + "%05d.jpg" % count, frame)
            count += 1
            if cv2.waitKey(1) & 0xFF == 27:
                break

            # from time import sleep
            # sleep(0.2)

        dataframe = pd.DataFrame(
            data=np.concatenate(
                [np.array(pitch_predicted_, ndmin=2), np.array(yaw_predicted_, ndmin=2),
                 np.array(gaze_class_, ndmin=2)]).T,
            columns=["yaw", "pitch", 'class'])
        # dataframe.to_csv('pitchjaw/' + filename + '.csv', index=False)
