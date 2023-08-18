import cv2

video_path = 'Input_videos/Research_videos/'
frame_path = 'input/objectbox_test/'

# Read video
video_name = 'Proefpersoon11024_sessie2.MP4'
vidcap = cv2.VideoCapture(video_path + video_name)

success, image = vidcap.read()
count = 0
while success:
    print(count)
    # image = cv2.rotate(image, cv2.ROTATE_180)
    # cv2.imwrite("Test_frames/Video_5/frame_res3/frame%d.jpg" % count, image)     # save frame as JPEG file
    cv2.imwrite(frame_path + "%05d.jpg" % count, image)  # save frame as JPEG file
    success, image = vidcap.read()
    # print('Read a new frame: ', success)
    count += 1
