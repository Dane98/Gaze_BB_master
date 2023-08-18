import pandas as pd
from readbbtxt import readbbtxt

import cv2

datafolder = './data_fin/'
datafile = 'pixel_position_vis.txt'

data = readbbtxt(datafolder+datafile)

root = './visible/'
for f in data['file']:
    video = f[:-3]+'MP4'
    cap = cv2.VideoCapture(root+video)
    if not cap.isOpened():
        video = f[:-3]+'mp4'
        cap = cv2.VideoCapture(root+video)
        if not cap.isOpened():
            raise IOError("Could not read the video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output = cv2.VideoWriter(
        './visible_bb/'+video, cv2.VideoWriter_fourcc(*'MPEG'), 
      fps, (width, height))
    
    while(True):
        ret, frame = cap.read()
        if(ret):
            # adding bb on each frame
            d = data[data['file']==f]
       
            # Red rectangle (tablet)
            cv2.rectangle(frame, (int(d['tl_tablet_x'][0]*width), int(d['tl_tablet_y'][0]*height)), (int(d['br_tablet_x'][0]*width), int(d['br_tablet_y'][0]*height)),
                          (0, 0, 255), 0)
            # Green rectangle (robot)
            cv2.rectangle(frame, (int(d['tl_robot_x'][0]*width), int(d['tl_robot_y'][0]*height)), (int(d['br_robot_x'][0]*width), int(d['br_robot_y'][0]*height)),
                          (0, 255, 0), 0)
            # Blue rectangle (table)
            cv2.rectangle(frame, (int(d['tl_pp_x'][0]*width), int(d['tl_pp_y'][0]*height)), (int(d['br_pp_x'][0]*width), int(d['br_pp_y'][0]*height)),
                          (255, 0, 0), 0)
            
    
            # original coordinates
            # Red rectangle (tablet)
            # cv2.rectangle(frame, (int(d['tl_tablet_x']), int(d['tl_tablet_y'])), (int(d['br_tablet_x']), int(d['br_tablet_y'])),
            #               (0, 0, 255), 0)
            # # Green rectangle (robot)
            # cv2.rectangle(frame, (int(d['tl_robot_x']), int(d['tl_robot_y'])), (int(d['br_robot_x']), int(d['br_robot_y'])),
            #               (0, 255, 0), 0)
            # # Blue rectangle (table)
            # cv2.rectangle(frame, (int(d['tl_pp_x']), int(d['tl_pp_y'])), (int(d['br_pp_x']), int(d['br_pp_y'])),
            #               (255, 0, 0), 0)
              
              
            # writing the new frame in output
            output.write(frame)
            cv2.imshow(f, frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break

    cv2.destroyAllWindows()
    output.release()
    cap.release()
    
    # comment break to output all videos
    # break