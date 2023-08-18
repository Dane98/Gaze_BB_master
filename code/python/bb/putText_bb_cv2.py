import os
import cv2
import pandas as pd
from readbbtxt_inv import readbbtxt

datafolder = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/data_fin/'
datafile = 'pixel_position_invis_original_new.txt'
image_path = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/frames/Saved_frame/fig_invis_org/'
# test_img = '51007_sessie1_taskrobotEngagement.png'
saved_inv_root = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/frames/Saved_frame/fig_invis/'

data = readbbtxt(datafolder + datafile)

for i in data.index:
    im = cv2.imread(image_path + data['file'][i])
    # height, width, channels = im.shape
    cv2.putText(im, 'Tablet', (int(data['tl_tablet_x'][i]), int(data['tl_tablet_y'][i]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255), 2)
    cv2.putText(im, 'Robot', (int(data['tl_robot_x'][i]), int(data['tl_robot_y'][i]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0), 2)
    cv2.putText(im, 'Table', (int(data['tl_pp_x'][i]), int(data['tl_pp_y'][i]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 0, 0), 2)

    cv2.imshow(data['file'][i], im)
    cv2.imwrite(os.path.join(saved_inv_root, data['file'][i]), im)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


