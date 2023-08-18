# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from PIL import Image
import os
import cv2
import pandas as pd
import numpy as np

root = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/'
vis_img_root = root + 'frames/enlarged_frame/frames_vis/'
invis_img_root = root + 'frames/enlarged_frame/frame_invis/'
# saved_vis_root = root + 'frames/Saved_frame/fig_vis/'
# saved_invis_root = root + 'frames/Saved_frame/fig_invis/'
save_inv_fig = 'frames/temp/51007/'
img_name = '00000.jpg'

bb_image_path = './frames/Saved_frame/fig_invis/51007_sessie2_taskrobotEngagement/'
l2cs_image_path = './frames/Saved_frame/fig_invis_l2cs_filtered/51007_sessie2_taskrobotEngagement_temp/'
baseline_image_path = './frames/test_face/'

# makes output window scalable
cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)

vis_img = vis_img_root + img_name
invis_img = invis_img_root + img_name
bb_img = bb_image_path + img_name
l2cs_img = l2cs_image_path + img_name
base_img = baseline_image_path + img_name

# Read RGB image
# im = cv2.imread(vis_img)
im = cv2.imread(l2cs_img)
height, width, channels = im.shape

# width, height = 852, 480
# print(f"({width}, {height})")   # (852,480) (1280,720) (1920, 1080)
# 381,142,442,221 # face ######## 610,176,743,328
# (225, 437), (412, 480), (83, 162), (268, 480), (602, 420), (852, 480)###########(730, 480), (910, 640), (410, 480), (730, 840), (130, 480), (340, 660)
# (0.264, 0.91), (0.484, 1.0), (0.097, 0.338), (0.315, 1.0), (0.707, 0.875), (1.0, 1.0)
face_SP, face_EP = (610, 176), (743, 328)

tablet_SP, tablet_EP = (730, 480), (910, 640)
robot_SP, robot_EP = (410, 480), (730, 840)
table_SP, table_EP = (130, 480), (340, 660)
# tablet_SP, tablet_EP = (round(0.264*width), round(0.91*height)), (round(0.484*width), round(1.0*height))
# robot_SP, robot_EP = (round(0.097*width), round(0.338*height)), (round(0.315*width), round(1.0*height))
# table_SP, table_EP = (round(0.707*width), round(0.875*height)), (round(1.0*width), round(1.0*height))
# Draw rectangles (SP=start point, EP=End point)
# (690, 480), (870, 640), (320, 480), (640, 840), (30, 470), (240, 650)
# (730, 480), (910, 640), (410, 480), (730, 840), (130, 480), (340, 660)
# tablet_SP, tablet_EP = (730, 480), (910, 640)
# robot_SP, robot_EP = (410, 480), (730, 840)
# table_SP, table_EP = (130, 480), (340, 660)

# Red rectangle (tablet)
tablet_invis = True
cv2.rectangle(im, tablet_SP, tablet_EP,
              (0, 0, 255), 2)
# face red
cv2.rectangle(im, face_SP, face_EP,
              (0, 0, 255), 2)

# Green rectangle (robot)
robot_invis = True
cv2.rectangle(im, robot_SP, robot_EP,
              (0, 255, 0), 2)
# Blue rectangle (table)
table_invis = True
cv2.rectangle(im, table_SP, table_EP,
              (255, 0, 0), 2)

# data = np.array(['Video_name', 'Tablet_SP', 'Tablet_EP', 'Robot_SP', 'Robot_EP', 'Table_SP', 'Table_EP', 'Invisible_obj'])

invis_obj = []
if tablet_invis == robot_invis == table_invis is False:
    invis_obj = None

while tablet_invis:
    invis_obj.append("Tablet")
    break
while robot_invis:
    invis_obj.append("Robot")
    break
while table_invis:
    invis_obj.append("Table")
    break

# Normalize
norm_tablet_SP = (round(tablet_SP[0] / width, 3), round(tablet_SP[1] / height, 3))
norm_tablet_EP = (round(tablet_EP[0] / width, 3), round(tablet_EP[1] / height, 3))
norm_robot_SP = (round(robot_SP[0] / width, 3), round(robot_SP[1] / height, 3))
norm_robot_EP = (round(robot_EP[0] / width, 3), round(robot_EP[1] / height, 3))
norm_table_SP = (round(table_SP[0] / width, 3), round(table_SP[1] / height, 3))
norm_table_EP = (round(table_EP[0] / width, 3), round(table_EP[1] / height, 3))
# print(f"{norm_tablet_SP}, {norm_tablet_EP}, {norm_robot_SP}, {norm_robot_EP}, {norm_table_SP}, {norm_table_EP}")

# Output img with window name as image_name
cv2.imshow(img_name, im)

cv2.imwrite(os.path.join(save_inv_fig, img_name), im)
# uncomment this part to write the image and text files
# cv2.imwrite(os.path.join(saved_invis_root, img_name), im)
# with open("pixel_position_invis_original.txt", "a") as file:
#     file.write(f"{img_name}, {tablet_SP}, {tablet_EP}, {robot_SP}, "
#                f"{robot_EP}, {table_SP}, {table_EP}, {invis_obj} \n")
# with open("pixel_position_invis.txt", "a") as file:
#     file.write(f"{img_name}, {norm_tablet_SP}, {norm_tablet_EP}, {norm_robot_SP}, "
#                f"{norm_robot_EP}, {norm_table_SP}, {norm_table_EP}, {invis_obj} \n")

# Maintain output window until user presses a key
cv2.waitKey(2 * 500)
# Destroying present windows on screen
cv2.destroyAllWindows()
