import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from itertools import groupby


def get_classname(id):
    if id == 0:
        return 'pen & paper'
    elif id == 1:
        return 'robot'
    elif id == 2:
        return 'tablet'
    elif id == 3:
        return 'elsewhere'
    elif id == 4:
        return 'unknown'


def get_color(id):
    if id == 0:
        return 'skyblue'
    elif id == 1:
        return 'olive'
    elif id == 2:
        return 'violet'
    elif id == 3:
        return 'darkorange'
    elif id == 4:
        return 'sandybrown'


def mapping_to_event_level(df, gold_column='class'):
    manual_event_group = []
    # count = 1
    for f in df['file'].unique():
        list_manual = df[df['file'] == f][gold_column].tolist()
        # list_l2cs = df_vis[df_vis['file'] == f]['l2cs_class'].tolist()

        count_dups = [sum(1 for _ in group) for _, group in groupby(list_manual)]
        count = 1
        for cd in count_dups:
            # manual_event_grouop_single.extend([count_dups.index(cd)+1] * cd)
            manual_event_group.extend([count] * cd)
            count += 1
    return manual_event_group


# def plotfile(filename):
#     df = pd.read_csv(filename)
#
#     df['m2prop_class'] = df['m2prop_class'].astype(int)
#     classes = df['m2prop_class'].astype(int).unique()
#
#     df_vid = pd.read_csv('vidstat.csv')
#     # nr_frames = df_vid.loc[df_vid['file'] == filename, 'frames'].iloc[0].astype(int)
#     # fps = df_vid.loc[df_vid['file'] == filename, 'rate'].iloc[0].astype(int)
#     # fps = df_vid.query('file == Proefpersoon22016_Sessie1')['rate'].astype(int)
#
#     # bin = fps * 3
#     nr_frames = len(df.index)
#     x = range(0, nr_frames)
#
#     # print(f'bin: {bin}')
#     # print(type(bin))
#
#     aggregated_class = df['m2prop_class'].copy()
#     df_class = aggregated_class.groupby(aggregated_class.index // 1).transform(lambda a: a.value_counts().idxmax())
#     # most = df['class'][0:30].mode()[0]
#
#     bars = []
#     bottoms = []
#     for c in classes:
#         arr = 1 * np.array(df_class == c)
#         bars.append(arr)
#         bottom = np.ones(nr_frames).astype(int)
#         bottoms.append(bottom)
#
#     plt.rcParams['xtick.labelsize'] = 8
#     plt.rcParams['ytick.labelsize'] = 8
#     fig, ax = plt.subplots()
#
#     labels = [get_classname(i) for i in classes]
#     colors = [get_color(i) for i in classes]
#
#     for b, bot, col in zip(bars, bottoms, colors):
#         ax.bar(x, height=b, width=1, bottom=bot, color=col)
#
#     # ax.hlines(1.5, 0, nr_frames, color='lightgrey', linestyles='solid', linewidth=0.5)
#
#     ax.set_ylim((1, 4))
#     ax.set_xlim((0, nr_frames))
#     ax.set_yticks([1.5])
#     ax.set_yticklabels(["classes"])
#     ax.set_ylabel('gazed upon object', fontsize=12)
#     ax.set_xlabel('frame', fontsize=12)
#     fig.tight_layout()
#     plt.legend(labels)
#     plt.savefig(join(plot_path, f[:-4] + '_gaze.png'), dpi=300)
#     # plt.savefig(join('./plot/33004_sessie1_taskrobotEngagement_gaze_aggr.png'), dpi=300)
#     # plt.show()
#     plt.close()


def plotdata_event_vis(filepath, f):
    df = pd.read_csv(filepath, header=0)
    df_file = df[df['file'] == f].copy() # this is raw data
    # df_file = df[
    #     (df['file'] == f) & (df['base_ff'] == 1) & (df['class'].isin([0, 1, 2, 3]))].copy()  # this is cleaned data

    event_group = mapping_to_event_level(df_file)  # mapping the cleaned data to event level
    df_file['event_group'] = event_group  # add event group into dataframe

    # Group-by event and transform the grouped labels with the most frequent values (will keep the original data length)
    df_group_single = df_file.groupby('event_group')[
        ['class', 'base_class', 'm2prop_class']].transform(lambda x: x.mode().iloc[0])

    # df_group_single = df_group[df_group['file'] == filename]
    manual_class = df_group_single['class'].astype(int).copy()
    l2cs_class = df_group_single['m2prop_class'].astype(int).copy()
    base_class = df_group_single['base_class'].astype(int).copy()
    # framenumber = df_group_single['framenumber'].astype(int).copy()
    classes_manual = df_group_single['class'].astype(int).unique().tolist()
    classes_l2cs = df_group_single['m2prop_class'].astype(int).unique().tolist()
    classes_manual_set = set(classes_manual)
    classes = classes_manual + [x for x in classes_l2cs if x not in classes_manual_set]

    nr_frames = len(df_file[df_file['file'] == f])
    x = range(0, nr_frames)

    # aggregated_class = df['m2prop_class'].copy()
    # df_class = aggregated_class.groupby(aggregated_class.index // 1).transform(lambda a: a.value_counts().idxmax())
    # most = df['class'][0:30].mode()[0]

    bars_base = []
    bottoms_base = []
    for c in classes:
        arr = 1 * np.array(base_class == c)
        bars_base.append(arr)
        bottom = np.ones(nr_frames).astype(int)
        bottoms_base.append(bottom)

    bars_manual = []
    bottoms_manual = []
    for c in classes:
        arr = 1 * np.array(manual_class == c)
        bars_manual.append(arr)
        bottom = np.ones(nr_frames).astype(int) * 2
        bottoms_manual.append(bottom)

    bars_l2cs = []
    bottoms_l2cs = []
    for c in classes:
        arr = 1 * np.array(l2cs_class == c)
        bars_l2cs.append(arr)
        bottom = np.ones(nr_frames).astype(int) * 3
        bottoms_l2cs.append(bottom)

    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams["legend.loc"] = 'upper right'
    plt.rcParams['lines.color'] = 'lightgrey'
    fig, ax = plt.subplots()

    labels = [get_classname(i) for i in classes]
    colors = [get_color(i) for i in classes]

    for b, bot, col in zip(bars_base, bottoms_base, colors):
        ax.bar(x, height=b, width=1, bottom=bot, color=col)

    for b, bot, col in zip(bars_manual, bottoms_manual, colors):
        ax.bar(x, height=b, width=1, bottom=bot, color=col)

    for b, bot, col in zip(bars_l2cs, bottoms_l2cs, colors):
        ax.bar(x, height=b, width=1, bottom=bot, color=col)

    ax.set_ylim((1, 5))
    ax.set_xlim((0, nr_frames))
    ax.set_yticks([1.5, 2.5, 3.5])
    ax.set_yticklabels(["baseline", "manual", "l2cs"])
    ax.set_ylabel('methods output', fontsize=12)
    ax.set_xlabel('frames on event level', fontsize=12)
    fig.tight_layout()
    plt.legend(labels)
    ax.hlines([2, 3], 0, nr_frames, linestyles='solid', linewidth=0.5)
    plt.savefig(join(plot_path_event, f + '_event.png'), dpi=300)
    # plt.savefig(join(f'./plot/{f}_gaze_aggr_event.png'), dpi=300)
    # plt.show()
    plt.close()


def plotdata_frame_vis(filepath, f):
    df = pd.read_csv(filepath, header=0)
    df_file = df[df['file'] == f].copy()    # this is raw data
    # df_file = df[
    #     (df['file'] == f) & (df['base_ff'] == 1) & (df['class'].isin([0, 1, 2, 3]))].copy()  # this is cleaned data

    manual_class = df_file['class'].astype(int).copy()
    l2cs_class = df_file['m2prop_class'].astype(int).copy()
    base_class = df_file['base_class'].astype(int).copy()
    # framenumber = df_file['framenumber'].astype(int).copy()
    classes_manual = df_file['class'].astype(int).unique().tolist()
    classes_l2cs = df_file['m2prop_class'].astype(int).unique().tolist()
    classes_manual_set = set(classes_manual)
    classes = classes_manual + [x for x in classes_l2cs if x not in classes_manual_set]

    nr_frames = len(df_file[df_file['file'] == f])
    x = range(0, nr_frames)

    bars_base = []
    bottoms_base = []
    for c in classes:
        arr = 1 * np.array(base_class == c)
        bars_base.append(arr)
        bottom = np.ones(nr_frames).astype(int)
        bottoms_base.append(bottom)

    bars_manual = []
    bottoms_manual = []
    for c in classes:
        arr = 1 * np.array(manual_class == c)
        bars_manual.append(arr)
        bottom = np.ones(nr_frames).astype(int) * 2
        bottoms_manual.append(bottom)

    bars_l2cs = []
    bottoms_l2cs = []
    for c in classes:
        arr = 1 * np.array(l2cs_class == c)
        bars_l2cs.append(arr)
        bottom = np.ones(nr_frames).astype(int) * 3
        bottoms_l2cs.append(bottom)

    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['lines.color'] = 'lightgrey'
    fig, ax = plt.subplots()

    labels = [get_classname(i) for i in classes]
    colors = [get_color(i) for i in classes]

    for b, bot, col in zip(bars_base, bottoms_base, colors):
        ax.bar(x, height=b, width=1, bottom=bot, color=col)

    for b, bot, col in zip(bars_manual, bottoms_manual, colors):
        ax.bar(x, height=b, width=1, bottom=bot, color=col)

    for b, bot, col in zip(bars_l2cs, bottoms_l2cs, colors):
        ax.bar(x, height=b, width=1, bottom=bot, color=col)

    ax.set_ylim((1, 5))
    ax.set_xlim((0, nr_frames))
    ax.set_yticks([1.5, 2.5, 3.5])
    ax.set_yticklabels(["baseline", "manual", "l2cs"])
    ax.set_ylabel('methods output', fontsize=12)
    ax.set_xlabel('frames', fontsize=12)
    fig.tight_layout()
    plt.legend(labels)
    ax.hlines([2, 3], 0, nr_frames, linestyles='solid', linewidth=0.5)
    plt.savefig(join(plot_path_frame, f + '_frame.png'), dpi=300)
    # plt.savefig(join(f'./plot/{f}_gaze_aggr_frame.png'), dpi=300)
    # plt.show()
    plt.close()


def plotdata_event_invis(filepath, f):
    df = pd.read_csv(filepath, header=0)
    df_file = df[df['file'] == f].copy()    # this is raw data
    # df_file = df[
    #     (df['file'] == f) & (df['base_ff'] == 1) & (df['class'].isin([0, 1, 2, 3]))].copy()  # this is cleaned data

    event_group = mapping_to_event_level(df_file)  # mapping the cleaned data to event level
    df_file['event_group'] = event_group  # add event group into dataframe

    # Group-by event and transform the grouped labels with the most frequent values (will keep the original data length)
    df_group_single = df_file.groupby('event_group')[['class', 'm2prop_class']].transform(lambda x: x.mode().iloc[0])

    manual_class = df_group_single['class'].astype(int).copy()
    l2cs_class = df_group_single['m2prop_class'].astype(int).copy()
    # framenumber = df_group_single['framenumber'].astype(int).copy()
    classes_manual = df_group_single['class'].astype(int).unique().tolist()
    classes_l2cs = df_group_single['m2prop_class'].astype(int).unique().tolist()
    classes_manual_set = set(classes_manual)
    classes = classes_manual + [x for x in classes_l2cs if x not in classes_manual_set]

    nr_frames = len(df_file[df_file['file'] == f])
    x = range(0, nr_frames)

    bars_manual = []
    bottoms_manual = []
    for c in classes:
        arr = 1 * np.array(manual_class == c)
        bars_manual.append(arr)
        bottom = np.ones(nr_frames).astype(int)
        bottoms_manual.append(bottom)

    bars_l2cs = []
    bottoms_l2cs = []
    for c in classes:
        arr = 1 * np.array(l2cs_class == c)
        bars_l2cs.append(arr)
        bottom = np.ones(nr_frames).astype(int) * 2
        bottoms_l2cs.append(bottom)

    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['lines.color'] = 'lightgrey'
    fig, ax = plt.subplots()

    labels = [get_classname(i) for i in classes]
    colors = [get_color(i) for i in classes]

    for b, bot, col in zip(bars_manual, bottoms_manual, colors):
        ax.bar(x, height=b, width=1, bottom=bot, color=col)

    for b, bot, col in zip(bars_l2cs, bottoms_l2cs, colors):
        ax.bar(x, height=b, width=1, bottom=bot, color=col)

    ax.set_ylim((1, 4))
    ax.set_xlim((0, nr_frames))
    ax.set_yticks([1.5, 2.5])
    ax.set_yticklabels(["manual", "l2cs"])
    ax.set_ylabel('methods output', fontsize=12)
    ax.set_xlabel('frames on event level', fontsize=12)
    fig.tight_layout()
    plt.legend(labels)
    ax.hlines(2, 0, nr_frames, linestyles='solid', linewidth=0.5)
    plt.savefig(join(plot_path_event, f + '_event.png'), dpi=300)
    # plt.savefig(join(f'./plot/{f}_gaze_aggr_event.png'), dpi=300)
    # plt.show()
    plt.close()


def plotdata_frame_invis(filepath, f):
    df = pd.read_csv(filepath, header=0)
    df_file = df[df['file'] == f].copy()    # this is raw data
    # df_file = df[
    #     (df['file'] == f) & (df['base_ff'] == 1) & (df['class'].isin([0, 1, 2, 3]))].copy()  # this is cleaned data

    manual_class = df_file['class'].astype(int).copy()
    l2cs_class = df_file['m2prop_class'].astype(int).copy()
    # framenumber = df_file['framenumber'].astype(int).copy()
    classes_manual = df_file['class'].astype(int).unique().tolist()
    classes_l2cs = df_file['m2prop_class'].astype(int).unique().tolist()
    classes_manual_set = set(classes_manual)
    classes = classes_manual + [x for x in classes_l2cs if x not in classes_manual_set]

    nr_frames = len(df_file[df_file['file'] == f])
    x = range(0, nr_frames)

    bars_manual = []
    bottoms_manual = []
    for c in classes:
        arr = 1 * np.array(manual_class == c)
        bars_manual.append(arr)
        bottom = np.ones(nr_frames).astype(int)
        bottoms_manual.append(bottom)

    bars_l2cs = []
    bottoms_l2cs = []
    for c in classes:
        arr = 1 * np.array(l2cs_class == c)
        bars_l2cs.append(arr)
        bottom = np.ones(nr_frames).astype(int) * 2
        bottoms_l2cs.append(bottom)

    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['lines.color'] = 'lightgrey'
    fig, ax = plt.subplots()

    labels = [get_classname(i) for i in classes]
    colors = [get_color(i) for i in classes]

    for b, bot, col in zip(bars_manual, bottoms_manual, colors):
        ax.bar(x, height=b, width=1, bottom=bot, color=col)

    for b, bot, col in zip(bars_l2cs, bottoms_l2cs, colors):
        ax.bar(x, height=b, width=1, bottom=bot, color=col)

    ax.set_ylim((1, 4))
    ax.set_xlim((0, nr_frames))
    ax.set_yticks([1.5, 2.5])
    ax.set_yticklabels(["manual", "l2cs"])
    ax.set_ylabel('methods output', fontsize=12)
    ax.set_xlabel('frames', fontsize=12)
    fig.tight_layout()
    plt.legend(labels)
    ax.hlines([2, 3], 0, nr_frames, linestyles='solid', linewidth=0.5)
    plt.savefig(join(plot_path_frame, f + '_frame.png'), dpi=300)
    # plt.savefig(join(f'./plot/{f}_gaze_aggr_frame.png'), dpi=300)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    # plotfile('./pitchjaw/visible_filtered_m2prop/33004_sessie1_taskrobotEngagement.csv')

    # # # this block is for single file processing
    # file_root = '../manual_annotation/results/'
    # file_path = file_root + 'mlm2_event_vis_merge.csv'
    # # file_path = file_root + 'ml_invis_merge_all.csv'
    # # filename = 'Proefpersoon11012_sessie1' # invis
    # # filename = '33007_sessie2_taskrobotEngagement'  # vis
    # filename = '33001_sessie1_taskrobotEngagement' # vis
    # plot_path_event = './plot/'
    # plot_path_frame = './plot/'
    # plotdata_event_vis(file_path, filename)
    # plotdata_frame_vis(file_path, filename)
    # # plotdata_event_invis(file_path, filename)
    # # plotdata_frame_invis(file_path, filename)

    # # #
    # df = pd.read_csv(file_path, header=0)
    # print(len(df[df['file'] == filename]))
    # print(os.path.basename(file_path)[:-4])
    # df_file = df[df['file'] == filename].copy()
    # df_group_single = df_file.groupby('event_group')[['class', 'l2cs_pred_class', 'base_class', 'm2prop_class']].transform(lambda x: x.mode().iloc[0])
    # df_group = df.groupby('event_group')[
    #     ['file', 'framenumber', 'class', 'l2cs_pred_class', 'base_class', 'm2prop_class']].apply(lambda x: x.mode().iloc[0])
    # print(df_group)
    # print(len(df_file))
    # print(len(df_group_single))
    #
    # l2cs_class = df_group_single['m2prop_class'].astype(int).copy()
    # print(len(l2cs_class))
    # print(l2cs_class)
    # l2cs_class.to_csv('test_l2cs.csv')

    # df_group = df_file.groupby('event_group')[['framenumber', 'm2prop_class']].transform(lambda x: x.mode().iloc[0])
    # df_file['m2prop_class'] = df_file.groupby('event_group')['m2prop_class'].transform(lambda x: x.mode().iloc[0])
    # df_group_single = df_group[df_group['file'] == filename][['framenumber', 'class', 'l2cs_pred_class', 'base_class', 'm2prop_class']].astype(int)
    # df_group_single = df_group[df_group['file'] == filename]
    # print(len(df_file['m2prop_class']))
    # print(df_file['m2prop_class'])
    # print(len(df))
    # print(df_group)

    # aggregated_class = df[df['file'] == filename][['file', 'class', 'event_group', 'm2prop_class']].copy()
    # df_class = aggregated_class[['event_group', 'm2prop_class']].groupby(['event_group'])['m2prop_class']
    # df_class_c = df_class.transform(lambda a: )
    # df_class = aggregated_class[['event_group', 'm2prop_class']].groupby(['event_group']).transform(lambda a: a.value_counts().idxmax())
    # m2_idxmax = df_class['m2prop_class'].tolist()
    # print(m2_idxmax)
    # df_class = aggregated_class.groupby(aggregated_class.index // 1).transform(lambda a: a.value_counts().idxmax())

    # """
    # visible case
    file_root = '../manual_annotation/results/'
    file_path = file_root + 'mlm2_event_vis_merge.csv'
    # plot_path_event = './plot/visible/event_mlb_cleaned/'
    # plot_path_frame = './plot/visible/frame_mlb_cleaned/'
    plot_path_event = './plot/visible/event_mlb/'
    plot_path_frame = './plot/visible/frame_mlb/'

    df = pd.read_csv(file_path, header=0)
    # print(len(df['file'].unique()))
    for f in df['file'].unique():
        plotdata_event_vis(file_path, f)
        plotdata_frame_vis(file_path, f)

    # invisible case
    file_root = '../manual_annotation/results/'
    file_path = file_root + 'ml_invis_merge_all.csv'
    # plot_path_event = './plot/invisible/event_ml_cleaned/'
    # plot_path_frame = './plot/invisible/frame_ml_cleaned/'
    plot_path_event = './plot/invisible/event_ml/'
    plot_path_frame = './plot/invisible/frame_ml/'

    df = pd.read_csv(file_path, header=0)
    # print(len(df['file'].unique()))
    for f in df['file'].unique():
        plotdata_event_invis(file_path, f)
        plotdata_frame_invis(file_path, f)

    # file_path = './pitchjaw/visible_filtered_m2prop/'
    # plot_path = './plot/visible/m2prop_org/'
    #
    # vis_files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    #
    # for f in vis_files:
    #     plot(file_path + f)
    #
    # file_path = './pitchjaw/invisible_filtered_m2prop/'
    # plot_path = './plot/invisible/m2prop_org/'
    #
    # invis_files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    #
    # for f in invis_files:
    #     plot(file_path + f)
# """
# df_vid = pd.read_csv('vidstat.csv')
# nr_frames = df_vid.loc[df_vid['file'] == 'Proefpersoon22016_Sessie1', 'frames'].iloc[0]
# print(type(nr_frames))
# fps = df_vid.loc[df_vid['file'] == 'Proefpersoon22016_Sessie1', 'rate'].iloc[0]
# print(type(fps))

# df = pd.read_csv('./pitchjaw/visible/Proefpersoon22016_Sessie1.csv')
# df['class'] = df['class'].astype(int)
# most = df['class'][0:30].mode()[0]
# print(most)
# print(type(most))

# dftest = pd.read_csv('testtest.csv')
# aggregated_class = dftest['class'].copy()
# new_df_class = aggregated_class.groupby(aggregated_class.index // 5).transform(lambda a: a.value_counts().idxmax())
# print(new_df_class)
