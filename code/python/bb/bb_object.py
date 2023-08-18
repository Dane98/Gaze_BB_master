from PIL import Image, ImageDraw, ImageFilter
from readbbtxt_inv import readbbtxt
import os
import cv2

root = './invisible/'
frame_root = './frames/frame_invis/'
datafolder = './data_fin/'
datafile = 'pixel_position_invis_new.txt'
frame_bb_object = './invisible_bb_object/'
data = readbbtxt(datafolder + datafile)

# im = Image.open('frames/00000_bb.jpg')
robot = Image.open('new_robot.png')
table = Image.open('new_table.png')
tablet = Image.open('new_tablet.png')

for f in data['file']:
    filepath = os.path.join(frame_bb_object, f[:-4])
    frame_path = os.path.join(frame_root, f[:-4])
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    video = f[:-3] + 'MP4'
    cap = cv2.VideoCapture(root + video)
    if not cap.isOpened():
        video = f[:-3] + 'mp4'
        cap = cv2.VideoCapture(root + video)
        if not cap.isOpened():
            raise IOError("Could not read the video file")

    count = 0
    while (True):
        ret, frame = cap.read()
        if (ret):
            # adding bb on each frame
            d = data[data['file'] == f].iloc[0]
            h, w, c = frame.shape
            # print(height_h)
            frame = cv2.copyMakeBorder(frame, 0, round(h * 0.75), round(w / 8), round(w / 8),
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))

            cv2.imwrite(os.path.join(frame_path, "%05d.jpg" % count), frame)

            height, width, channels = frame.shape

            # im = Image.open()
            # .paste(robot, (410, 480))
            # Red rectangle (tablet)
            cv2.rectangle(frame, (int(d['tl_tablet_x'] * width), int(d['tl_tablet_y'] * height)),
                          (int(d['br_tablet_x'] * width), int(d['br_tablet_y'] * height)), (0, 0, 255), -1)
            cv2.putText(frame, 'Tablet', (int(d['tl_tablet_x'] * width), int(d['tl_tablet_y'] * height) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # Green rectangle (robot)
            cv2.rectangle(frame, (int(d['tl_robot_x'] * width), int(d['tl_robot_y'] * height)),
                          (int(d['br_robot_x'] * width), int(d['br_robot_y'] * height)), (0, 255, 0), -1)
            cv2.putText(frame, 'Robot', (int(d['tl_robot_x'] * width), int(d['tl_robot_y'] * height) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Blue rectangle (table)
            cv2.rectangle(frame, (int(d['tl_pp_x'] * width), int(d['tl_pp_y'] * height)),
                          (int(d['br_pp_x'] * width), int(d['br_pp_y'] * height)), (255, 0, 0), -1)
            cv2.putText(frame, 'Table', (int(d['tl_pp_x'] * width), int(d['tl_pp_y'] * height) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # writing the new frame in output
            # output.write(frame)
            cv2.imwrite(os.path.join(filepath, "%05d.jpg" % count), frame)
            # print(count)
            count += 1
            cv2.imshow(f, frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break

    cv2.destroyAllWindows()
    # output.release()
    cap.release()

    # comment break to output all videos
    # break


back_im = im.copy()

back_im.paste(robot, (410, 480))
back_im.paste(table, (130, 480))
back_im.paste(tablet, (730, 480))
back_im.save('frames/00000_object.jpg', quality=95)

# tablet_SP, tablet_EP = (730, 480), (910, 640)
# robot_SP, robot_EP = (410, 480), (730, 840)
# table_SP, table_EP = (130, 480), (340, 660)
# robot_width = 300
# robot_height = 330
# new_robot = robot.resize((robot_width, robot_height), Image.ANTIALIAS)
# new_robot.save('new_robot.png')
#
# table_width = 200
# table_height = 160
# new_table = table.resize((table_width, table_height), Image.ANTIALIAS)
# new_table.save('new_table.png')
#
# tablet_width = 180
# tablet_height = 160
# new_tablet = tablet.resize((tablet_width, tablet_height), Image.ANTIALIAS)
# new_tablet.save('new_tablet.png')
