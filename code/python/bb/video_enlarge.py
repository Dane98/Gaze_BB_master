import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from moviepy.decorators import apply_to_mask
from moviepy.video.VideoClip import ImageClip
from moviepy.video.fx.margin import margin
# from moviepy.editor import ipython_display
from moviepy.video.io.VideoFileClip import VideoFileClip

vis_vid_folder = './visible/org/'
invis_vid_folder = './invisible/org/'

vis_frm_out = './frames/frame_vis/'
invis_frm_out = './frames/frame_invis/'


def visible_video_margin(video):
    clip = VideoFileClip(os.path.join(vis_vid_folder, video))
    clip1 = margin(clip, left=106, right=106, top=0, bottom=360)
    clip1.write_videofile(os.path.join('./frames/', video))


def invisible_video_margin(video):
    clip = VideoFileClip(os.path.join(invis_vid_folder, video))
    clip1 = margin(clip, left=106, right=106, top=0, bottom=360)
    clip1.write_videofile(os.path.join('./frames/', video))


for vid in os.listdir(invis_vid_folder):
    print(vid)
    if vid.endswith('mp4') or vid.endswith('MP4'):
        video_margin(vid)
"""
for vid in os.listdir(invis_vid_folder):
    print(vid)
    vidcap = cv2.VideoCapture(vid)
    if not vidcap.isOpened():
        vidcap = cv2.VideoCapture(vid)
        if not vidcap.isOpened():
            raise IOError("Could not read the video file")

    # width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    success, frame = vidcap.read()
    output = cv2.VideoWriter('test2.avi', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))

    count = 0
    while success:
        success, frame = vidcap.read()
        if frame is not None:
            # im_resize = cv2.resize(frame, (1064, 840))
            im_resize = cv2.copyMakeBorder(src=frame, top=0, bottom=360, left=106, right=106, borderType=cv2.BORDER_CONSTANT)
            cv2.imwrite(os.path.join(invis_frm_out, vid[0:-4]), "%05d.jpg" % count, im_resize)
            output.write(im_resize)
        else:
            pass
        if cv2.waitkey(10) == 27:
            break
        count += 1
    output.release()
    cap.release()
    cv2.destroyAllWindows()"""

# 852, 480
# test1 = './invisible/org/Proefpersoon51007_sessie1.mp4'
# test2 = './invisible/Proefpersoon11029_sessie2.mp4'
#
# frm_out = os.path.join('./frames/', 'frames_invis/test/')
