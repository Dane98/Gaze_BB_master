import os
import re
import pandas as pd
from itertools import groupby


def domain_class_info(list1, list2):
    list1 = list1.copy()
    list2 = list2.copy()

    i = 0
    for j in count_dups:
        manual_class = list1[i:j]
        l2cs_class = list2[i:j]

        manual_count = pd.Series(manual_class).value_counts()
        l2cs_count = pd.Series(l2cs_class).value_counts()
        #     print(l2cs_count)

        manual_label = manual_count.index.tolist()[0]  # a value
        l2cs_label = l2cs_count.index.tolist()  # a list of values
        l2cs_label_count = l2cs_count.tolist()  # counts for list of values
        print(f"manual label is {manual_label}")

        # threshold = 0.6
        max_prop = 0
        domain_label = 0
        temp = 0
        prop_list = []
        for c in l2cs_label:
            count = l2cs_count[c]
            prop = round(count / j, 2)
            prop_list.append((c, prop))
            if prop > max_prop:
                max_prop = prop
                domain_label = c
        #         if prop >= threshold:
        #             temp = prop
        #             l2cs_class = [c] * j
        #         else:
        #             continue
        #     if temp == 0:
        #         l2cs_class = [9] * j
        print(prop_list)
        print(max_prop)
        print(domain_label)
        print()
    return domain_label, max_prop


def domain_class_info(list1, list2):
    list1 = list1.copy()
    list2 = list2.copy()

    i = 0
    for j in count_dups:
        manual_class = list1[i:j]
        l2cs_class = list2[i:j]

        manual_count = pd.Series(manual_class).value_counts()
        l2cs_count = pd.Series(l2cs_class).value_counts()
        #     print(l2cs_count)

        manual_label = manual_count.index.tolist()[0]  # a value
        l2cs_label = l2cs_count.index.tolist()  # a list of values
        l2cs_label_count = l2cs_count.tolist()  # counts for list of values
        print(f"manual label is {manual_label}")

        # threshold = 0.6
        max_prop = 0
        domain_label = 0
        temp = 0
        prop_list = []
        for c in l2cs_label:
            count = l2cs_count[c]
            prop = round(count / j, 2)
            prop_list.append((c, prop))
            if prop > max_prop:
                max_prop = prop
                domain_label = c
        #         if prop >= threshold:
        #             temp = prop
        #             l2cs_class = [c] * j
        #         else:
        #             continue
        #     if temp == 0:
        #         l2cs_class = [9] * j
        print(prop_list)
        print(max_prop)
        print(domain_label)
        print()
    return domain_label, max_prop


def check_if_matching(df, list1, list2, threshold=0.6):
    # label = [0, 1, 2, 3, 4]
    list1 = list1.copy()
    list2 = list2.copy()

    i = 0
    is_match = 0
    matching_list = []
    for j in count_dups:
        manual_class = list1[i:j]
        l2cs_class = list2[i:j]

        manual_count = pd.Series(manual_class).value_counts()
        l2cs_count = pd.Series(l2cs_class).value_counts()
        #     print(l2cs_count)

        manual_label = manual_count.index.tolist()[0]
        l2cs_label = l2cs_count.index.tolist()
        l2cs_label_count = l2cs_count.tolist()
        # print(manual_label)
        # print(l2cs_label)

        # threshold = 0.6
        if manual_label in l2cs_label:
            count = l2cs_count[manual_label]
            prop = round(count / j, 2)
            print(f"prop for label {manual_label} is {prop}")
            if prop >= threshold:
                # l2cs_class = [manual_label] * j
                is_match = 1
                print(f"l2cs matches manual, the proportion of label {manual_label} is up to {prop}")
            else:
                is_match = 0
                print(f"There's no label over {threshold * 100}%.")
        else:
            is_match = 0
            print(f"l2cs disagrees with manual.")

    return df


# # visible case
# df_vis = pd.read_csv('../manual_annotation/results/mlb_face_vis_merge_th2_blur.csv', header=0)
#
# manual_event_grouop = []
# # count = 1
# for f in df_vis['file'].unique():
#
#     list_manual = df_vis[df_vis['file'] == f]['class'].tolist()
#     # list_l2cs = df_vis[df_vis['file'] == f]['l2cs_class'].tolist()
#
#     count_dups = [sum(1 for _ in group) for _, group in groupby(list_manual)]
#     count = 1
#     for cd in count_dups:
#         # manual_event_grouop_single.extend([count_dups.index(cd)+1] * cd)
#         manual_event_grouop.extend([count] * cd)
#         count += 1
#
# df_vis['event_group'] = manual_event_grouop
# df_vis.to_csv('../manual_annotation/results/ml_vis_filtered_face_th2_blur_event.csv', index=False)


# invisible case
df_invis = pd.read_csv('../manual_annotation/results/ml_face_invis_th2_blur.csv', header=0)


def mapping_to_event_level(df):
    manual_event_group = []
    # count = 1
    for f in df_invis['file'].unique():
        list_manual = df_invis[df_invis['file'] == f]['class'].tolist()
        # list_l2cs = df_vis[df_vis['file'] == f]['l2cs_class'].tolist()

        count_dups = [sum(1 for _ in group) for _, group in groupby(list_manual)]
        count = 1
        for cd in count_dups:
            # manual_event_grouop_single.extend([count_dups.index(cd)+1] * cd)
            manual_event_group.extend([count] * cd)
            count += 1
    return manual_event_group


event_list = mapping_to_event_level(df_invis)
print(len(event_list))
df_invis['event_group'] = event_list
df_invis.to_csv('../manual_annotation/results/ml_invis_filtered_face_th2_blur_event.csv', index=False)

"""
filepath = './data/ml_merge.csv'
f = '33013_sessie3_robotEngagement3'
df = pd.read_csv(filepath)
print(df.head())

df_temp = df[df['file'] == f].copy()

list_manual = df_temp['class'].tolist()
list_l2cs = df_temp['l2cs_class'].tolist()
count_dups = [sum(1 for _ in group) for _, group in groupby(list_manual)]
print(count_dups)

count_dups_idx = len(count_dups)
print(count_dups_idx)

manual_event_grouop = []
count = 1
for cd in count_dups:
    manual_event_grouop.extend([count] * cd)
    count += 1

df_temp['event_group'] = manual_event_grouop"""
