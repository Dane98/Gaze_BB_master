import os
import numpy as np
import pandas as pd
import cv2
from scipy import stats


# input_folder = './frames/Saved_frame/fig_vis/33003_sessie3_robotEngagement1/'
# output_folder = './frames/test_face/'
# file = '33003_sessie3_robotEngagement1'

# output_folder_base = './frames/test_face/org_face/base/'
# output_folder_l2cs = './frames/test_face/org_face/l2cs/'
# output_folder_base_f = './frames/test_face/z_face/base/'
# output_folder_l2cs_f = './frames/test_face/z_face/l2cs/'

def detect_outliers(data, threshold=2):
    # Convert the list of tuples to separate lists for x and y coordinates
    x_coords, y_coords = zip(*data)

    # Calculate the Z-scores for both x and y coordinates
    try:
        z_scores_x = np.abs((x_coords - np.mean(x_coords)) / (np.std(x_coords) + 0.00001))
        z_scores_y = np.abs((y_coords - np.mean(y_coords)) / (np.std(y_coords) + 0.00001))
    except ZeroDivisionError:
        z_scores_x = 0.0
        z_scores_y = 0.0

    # Find the indices of the outliers based on the threshold
    outlier_indices = np.where((z_scores_x > threshold) | (z_scores_y > threshold))

    # Get the tuples representing the outliers
    outliers = [data[i] for i in outlier_indices[0]]

    return outliers, outlier_indices[0]


def remove_outliers(data, threshold=2):
    outliers = detect_outliers(data, threshold)[0]

    non_outliers = [point for point in data if point not in outliers]
    count = 0
    for i in range(len(outliers)):
        if not outliers[i] == (0, 0):
            count += 1
    print(f'removed outliers are {count}')
    return non_outliers


def remove_outliers_successive(data, window_size=5, threshold=2):
    filtered_data = []
    outliers_norm, outlier_indices_norm = detect_outliers(data, threshold=2)
    for i in range(0, len(data), window_size):
        window_data = data[i:i + window_size]
        outliers, outlier_indices = detect_outliers(window_data, threshold)
        non_outliers = [point for j, point in enumerate(window_data) if
                        j not in outlier_indices and j not in outlier_indices_norm]
        filtered_data.extend(non_outliers)

    print(f'removed outliers are {len(data)-len(filtered_data)}')
    return filtered_data


def replace_outliers(df, data, threshold=2):
    outliers, outlier_indices = detect_outliers(data, threshold)

    outlier_indices_without0 = []
    for i in range(len(outliers)):
        if not outliers[i] == (0, 0):
            outlier_indices_without0.append(outlier_indices[i])

    print(f'length of outliers in {file} is {len(outlier_indices_without0)}')
    for id in outlier_indices_without0:
        # data[id] = (0, 0)
        df[id] = 2.0

    return df


def replace_outliers_successive(df, data, window_size=5, threshold=2):
    outlier_indices_without0 = []
    for i in range(0, len(data), window_size):
        window_data = data[i:i + window_size]
        outliers, outlier_indices = detect_outliers(window_data, threshold)
        for j in range(len(outliers)):
            if not outliers[j] == (0, 0):
                outlier_indices_without0.append(outlier_indices[j])

    # print(f'after windows checking, length of outliers in {file} is {len(outlier_indices_without0)}')

    outliers, outlier_indices = detect_outliers(data, threshold=2)
    for k in range(len(outliers)):
        if not outliers[k] == (0, 0):
            outlier_indices_without0.append(outlier_indices[k])

    print(f'after normal checking, length of outliers in {file} is {len(outlier_indices_without0)}')
    prop = len(outlier_indices_without0) / data_point * 100
    print(f'outliers proportion is {prop}%')
    print()
    for idx in outlier_indices_without0:
        # data[id] = (0, 0)
        df[idx] = 2.0
    return df


# def replace_out(data, threshold=2):
#     outliers, outlier_indices = detect_outliers(data, threshold)
#
#     outlier_indices_without0 = []
#     for i in range(len(outliers)):
#         if not outliers[i] == (0, 0):
#             outlier_indices_without0.append(outlier_indices[i])
#
#     print(f'length of outliers in {file} is {len(outlier_indices_without0)}')
#     # return data


# this block is to draw face center point in images
def get_face_point_list(image_list):
    bf_list = []
    lf_list = []
    single_image = image_list[0]
    sim_base = cv2.imread(os.path.join(image_folder, single_image))
    sim_l2cs = cv2.imread(os.path.join(image_folder, single_image))
    sim_filtered_base = cv2.imread(os.path.join(image_folder, single_image))
    sim_filtered_l2cs = cv2.imread(os.path.join(image_folder, single_image))

    for image in image_list:
        left_base = base_face_info[base_face_info['file'] == image]['left'].iloc[0]
        right_base = base_face_info[base_face_info['file'] == image]['right'].iloc[0]
        top_base = base_face_info[base_face_info['file'] == image]['top'].iloc[0]
        bottom_base = base_face_info[base_face_info['file'] == image]['bottom'].iloc[0]
        bb_width_base = right_base - left_base
        bb_height_base = bottom_base - top_base
        pos_base = (int(left_base + bb_width_base / 2.0), int(top_base + bb_height_base / 2.0))
        bf_list.append(pos_base)

        left_l2cs = l2cs_face_info[l2cs_face_info['file'] == image]['left'].iloc[0]
        right_l2cs = l2cs_face_info[l2cs_face_info['file'] == image]['right'].iloc[0]
        top_l2cs = l2cs_face_info[l2cs_face_info['file'] == image]['top'].iloc[0]
        bottom_l2cs = l2cs_face_info[l2cs_face_info['file'] == image]['bottom'].iloc[0]
        bb_width_l2cs = right_l2cs - left_l2cs
        bb_height_l2cs = bottom_l2cs - top_l2cs
        pos_l2cs = (int(left_l2cs + bb_width_l2cs / 2.0), int(top_l2cs + bb_height_l2cs / 2.0))
        lf_list.append(pos_l2cs)

        cv2.circle(sim_base, (pos_base[0], pos_base[1]), 2, (0, 255, 0), -1)
        cv2.circle(sim_l2cs, (pos_l2cs[0], pos_l2cs[1]), 2, (0, 0, 255), -1)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # filtered_base = remove_outliers_successive(bf_list, threshold=2, window_size=wd)
    # filtered_l2cs = remove_outliers_successive(lf_list, threshold=2, window_size=wd)
    filtered_base = remove_outliers(bf_list, threshold=2)
    filtered_l2cs = remove_outliers(lf_list, threshold=2)
    for base_point in filtered_base:
        cv2.circle(sim_filtered_base, base_point, 2, (0, 255, 0), -1)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    for l2cs_point in filtered_l2cs:
        cv2.circle(sim_filtered_l2cs, l2cs_point, 2, (0, 0, 255), -1)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.imwrite(output_folder + file + '_baseface.jpg', sim_base)
    cv2.imwrite(output_folder + file + '_l2csface.jpg', sim_l2cs)
    cv2.imwrite(output_folder + file + '_baseface_f.jpg', sim_filtered_base)
    cv2.imwrite(output_folder + file + '_l2csface_f.jpg', sim_filtered_l2cs)

    return bf_list, lf_list


# outlier_list_base = detect_outliers(base_face_list, threshold=2)
# outlier_list_l2cs = detect_outliers(l2cs_face_list, threshold=2)
# print(outlier_list_base)
# print(outlier_list_l2cs)
# print(len(outlier_list_base))
# print(len(outlier_list_l2cs))


# base_face = np.array(df_vis[df_vis['file'] == file]['base_face'])
# l2cs_face = np.array(df_vis[df_vis['file'] == file]['l2cs_face'])
# mean = np.mean(base_face)
# std_dev = np.std(base_face)
# z_threshold = 2
# z_scores = (base_face - mean) / std_dev
# outliers = base_face[np.abs(z_scores) > z_threshold]
# filtered_data = base_face[np.abs(z_scores) <= z_threshold]


if __name__ == '__main__':
    """
    # visible case
    root_folder = os.path.join('./frames/Saved_frame/fig_vis/')
    # output_folder = os.path.join('./frames/Saved_frame/fig_vis_face_filtered/z_combine_method/window25_threshold2/')

    df_vis = pd.read_csv('../manual_annotation/results/mlb_face_vis_merge.csv', header=0)

    base_face_list_new = []
    l2cs_face_list_new = []
    for file in df_vis['file'].unique():
        if file.startswith('P'):
            wd = 30
        else:
            wd = 25
        image_folder = os.path.join(root_folder, file)
        img_list = [img for img in os.listdir(image_folder)]
        image_list = sorted(img_list)

        base_face_info = pd.read_csv('./data/face_baseline/face_vis/' + file + '.txt',
                                     names=['file', 'left', 'top', 'right', 'bottom', 'face'])
        l2cs_face_info = pd.read_csv('./data/face_l2cs/face_vis/' + file + '.txt',
                                     names=['file', 'left', 'top', 'right', 'bottom', 'face'])

        base_face_list, l2cs_face_list = get_face_point_list(image_list)

        # this block is to filter the face
        base_face = df_vis[df_vis['file'] == file]['base_face'].tolist()
        l2cs_face = df_vis[df_vis['file'] == file]['l2cs_face'].tolist()
        data_point = len(base_face)

        # df_vis_base_f = replace_outliers_successive(base_face, base_face_list, window_size=wd)
        # df_vis_l2cs_f = replace_outliers_successive(l2cs_face, l2cs_face_list, window_size=wd)
        df_vis_base_f = replace_outliers(base_face, base_face_list)
        df_vis_l2cs_f = replace_outliers(l2cs_face, l2cs_face_list)
        base_face_list_new.extend(df_vis_base_f)
        l2cs_face_list_new.extend(df_vis_l2cs_f)
        # replace_out(base_face_list)
        # replace_out(l2cs_face_list)

    df_vis['base_ff'] = base_face_list_new
    df_vis['l2cs_ff'] = l2cs_face_list_new
    df_vis.to_csv(os.path.join('../manual_annotation/results/', 'ml_vis_filtered_face_th2.csv'), index=False)"""

    # invisible case
    root_folder = os.path.join('./frames/Saved_frame/fig_invis/')
    output_folder = os.path.join('./frames/Saved_frame/fig_invis_face_filtered/z_org/threshold2/')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df_invis = pd.read_csv('../manual_annotation/results/ml_face_invis_merge.csv', header=0)

    inv_base_face_list_new = []
    inv_l2cs_face_list_new = []
    for file in df_invis['file'].unique():
        # if file.startswith('P'):
        #     wd = 30
        # else:
        #     wd = 25
        image_folder = os.path.join(root_folder, file)
        img_list = [img for img in os.listdir(image_folder)]
        image_list = sorted(img_list)

        base_face_info = pd.read_csv('./data/face_baseline/face_invis/' + file + '.txt',
                                     names=['file', 'left', 'top', 'right', 'bottom', 'face'])
        l2cs_face_info = pd.read_csv('./data/face_l2cs/face_invis/' + file + '.txt',
                                     names=['file', 'left', 'top', 'right', 'bottom', 'face'])

        base_face_list, l2cs_face_list = get_face_point_list(image_list)

        # this block is to filter the face
        base_face = df_invis[df_invis['file'] == file]['base_face'].tolist()
        l2cs_face = df_invis[df_invis['file'] == file]['l2cs_face'].tolist()
        data_point = len(base_face)

        # df_vis_base_f = replace_outliers_successive(base_face, base_face_list, window_size=wd)
        # df_vis_l2cs_f = replace_outliers_successive(l2cs_face, l2cs_face_list, window_size=wd)
        df_invis_base_f = replace_outliers(base_face, base_face_list)
        df_invis_l2cs_f = replace_outliers(l2cs_face, l2cs_face_list)
        inv_base_face_list_new.extend(df_invis_base_f)
        inv_l2cs_face_list_new.extend(df_invis_l2cs_f)
        # replace_out(base_face_list)
        # replace_out(l2cs_face_list)

    df_invis['base_ff'] = inv_base_face_list_new
    df_invis['l2cs_ff'] = inv_l2cs_face_list_new
    df_invis.to_csv(os.path.join('../manual_annotation/results/', 'ml_invis_filtered_face_th2.csv'), index=False)