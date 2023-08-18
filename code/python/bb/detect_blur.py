import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image


def is_blurry(frame, threshold=100):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


"""
# single file processing
face_file = './input/face_baseline/face_vis/33002_sessie2_taskrobotEngagement.txt'
face_info = pd.read_csv(face_file, names=['file', 'left', 'top', 'right', 'bottom', 'face'])

video_path = './testvid/33002_sessie2_taskrobotEngagement.MP4'

gpu = select_device("0", batch_size=16)

transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

count = 0
count_blur = 0
blurry_list_temp = []

cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image_name = "%05d.jpg" % count
    # for f in face_info['file'].unique():
    face_base_info = face_info[face_info['file'] == image_name].iloc[0]
    # print(face_base_info)
    # print(f)
    if face_base_info['face']:
        x_min = int(face_base_info['left'])
        y_min = int(face_base_info['top'])
        x_max = int(face_base_info['right'])
        y_max = int(face_base_info['bottom'])

        img = frame[y_min:y_max, x_min:x_max]
        # img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # im_pil = Image.fromarray(img)
        # img = transformations(im_pil)
        # img = Variable(img).cuda(gpu)
        # img = img.unsqueeze(0)

        if is_blurry(img):
            blurry_list_temp.append(1)
            # Display the frame (optional)
            cv2.imshow('Frame', frame)
            cv2.imwrite(os.path.join('./output/test/face_blur_100/', '%05d.jpg' % count), frame)
            count_blur += 1
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            blurry_list_temp.append(0)
            # print('not blurred')
            if cv2.waitKey(1) & 0xFF == 27:
                break
    else:
        blurry_list_temp.append(2)
        print('face is false')
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue
    count += 1
cap.release()
cv2.destroyAllWindows()

blur_count = pd.Series(blurry_list_temp).value_counts()
print(blur_count)
"""

"""
# gpu = select_device("0", batch_size=16)
#
# transformations = transforms.Compose([
#         transforms.Resize(448),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#     ])

# count = 0
# count_blur = 0

# visible case
face_root = './data/face_baseline/face_vis/'
vid_root = os.path.join('./visible_bb/')
blurry_list = []
count_all_blur = 0

df_vis = pd.read_csv('../manual_annotation/results/ml_vis_filtered_face_th2.csv', header=0)

for f in df_vis['file'].unique():
    face_info = pd.read_csv(face_root + f + '.txt', names=['file', 'left', 'top', 'right', 'bottom', 'face'])

    count_blur = 0
    blurry_list_temp = []

    if f.startswith('P'):
        vid_name = f + '.mp4'
    else:
        vid_name = f + '.MP4'
    video_path = os.path.join(vid_root, vid_name)

    count = 0
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_name = "%05d.jpg" % count
        # for f in face_info['file'].unique():
        face_base_info = face_info[face_info['file'] == image_name].iloc[0]
        # print(face_base_info)
        # print(f)
        if face_base_info['face']:
            x_min = int(face_base_info['left'])
            y_min = int(face_base_info['top'])
            x_max = int(face_base_info['right'])
            y_max = int(face_base_info['bottom'])

            img = frame[y_min:y_max, x_min:x_max]
            # img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # im_pil = Image.fromarray(img)
            # img = transformations(im_pil)
            # img = Variable(img).cuda(gpu)
            # img = img.unsqueeze(0)

            if is_blurry(img):
                blurry_list_temp.append(1)
                # Display the frame (optional)
                cv2.imshow('Frame', frame)
                cv2.imwrite(os.path.join(f'./frames/Saved_frame/face_blur_100/{f}/', '%05d.jpg' % count), frame)
                count_blur += 1
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                blurry_list_temp.append(0)
                # print('not blurred')
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        else:
            blurry_list_temp.append(2)
            # print('face is false')
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
        count += 1
    cap.release()
    cv2.destroyAllWindows()

    # print()
    # blur_count = pd.Series(blurry_list_temp).value_counts()
    # blur_count_temp = [zip(blur_count.index.value_count())]
    # with open('./frames/Saved_frame/face_blur_100.txt', 'a+') as file:
    #     file.write(','.join(str(b) for b in blur_count))
    # print(blur_count)
    count_all_blur += count_blur
    blurry_list.extend(blurry_list_temp)

print()
print(f'all blurs are {count_all_blur}')
print(f'blur list is {len(blurry_list)}')
df_vis['is_blurry'] = blurry_list
df_vis.to_csv(os.path.join('../manual_annotation/results/mlb_face_vis_merge_th2_blur.csv'), index=False)
"""

# df = pd.read_csv('../manual_annotation/results/mlb_face_vis_merge_blur.csv', header=0)
# print(df.loc[df['is_blurry'] == 1])
# df_blur = df['is_blurry'].tolist()
# blur_count = pd.Series(df_blur).value_counts()
# print(blur_count)

# invisible case
face_root = './data/face_baseline/face_invis/'
vid_root = os.path.join('./invisible_bb/')
blurry_list = []
count_all_blur = 0

df_invis = pd.read_csv('../manual_annotation/results/ml_invis_filtered_face_th2.csv', header=0)

for f in df_invis['file'].unique():
    face_info = pd.read_csv(face_root + f + '.txt', names=['file', 'left', 'top', 'right', 'bottom', 'face'])

    count_blur = 0
    blurry_list_temp = []

    if f.startswith('P'):
        vid_name = f + '.mp4'
    else:
        vid_name = f + '.MP4'
    video_path = os.path.join(vid_root, vid_name)

    count = 0
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_name = "%05d.jpg" % count
        # for f in face_info['file'].unique():
        face_base_info = face_info[face_info['file'] == image_name].iloc[0]
        # print(face_base_info)
        # print(f)
        if face_base_info['face']:
            x_min = int(face_base_info['left'])
            y_min = int(face_base_info['top'])
            x_max = int(face_base_info['right'])
            y_max = int(face_base_info['bottom'])

            img = frame[y_min:y_max, x_min:x_max]
            # img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # im_pil = Image.fromarray(img)
            # img = transformations(im_pil)
            # img = Variable(img).cuda(gpu)
            # img = img.unsqueeze(0)

            if is_blurry(img):
                blurry_list_temp.append(1)
                # Display the frame (optional)
                cv2.imshow('Frame', frame)
                # cv2.imwrite(os.path.join(f'./frames/Saved_frame/face_blur_100/{f}/', '%05d.jpg' % count), frame)
                count_blur += 1
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                blurry_list_temp.append(0)
                # print('not blurred')
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        else:
            blurry_list_temp.append(2)
            # print('face is false')
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
        count += 1
    cap.release()
    cv2.destroyAllWindows()

    # print()
    # blur_count = pd.Series(blurry_list_temp).value_counts()
    # blur_count_temp = [zip(blur_count.index.value_count())]
    # with open('./frames/Saved_frame/face_blur_100.txt', 'a+') as file:
    #     file.write(','.join(str(b) for b in blur_count))
    # print(blur_count)
    count_all_blur += count_blur
    blurry_list.extend(blurry_list_temp)

print()
print(f'all blurs are {count_all_blur}')
print(f'blur list is {len(blurry_list)}')
df_invis['is_blurry'] = blurry_list
df_invis.to_csv(os.path.join('../manual_annotation/results/ml_face_invis_th2_blur.csv'), index=False)
