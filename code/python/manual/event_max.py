import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import groupby
from collections import Counter


# Function to calculate the counts for each interval of 5%
def calculate_counts(percentage_list, interval=5):
    counts = [0] * 21  # Initialize a list of counts for each interval (20 intervals from 0-100)
    for percentage in percentage_list:
        interval_area = int(percentage // interval)  # Determine the interval for the current percentage
        counts[interval_area] += 1  # Increment the count for the corresponding interval
    return counts


# """
file_path_vis = './results/mlm2_event_vis_merge.csv'
file_path_invis = './results/ml_invis_merge_all.csv'
file_path_double = './results/annotations_double_frame.csv'

df_vis = pd.read_csv(file_path_vis, header=0)
df_invis = pd.read_csv(file_path_invis, header=0)
df_double = pd.read_csv(file_path_double, header=0)
# df_part_vis = df_vis[
#     (df_vis['base_ff'] == 1) & (df_vis['is_blurry'] == 0) & (df_vis['class'].isin([0, 1, 2, 3]))].copy()
# df_part_vis = df_vis[
#     (df_vis['base_face'] == 1) & (df_vis['is_blurry'] == 0) & (df_vis['class'].isin([0, 1, 2, 3]))].copy()
df_part_vis = df_vis[
    (df_vis['base_ff'] == 1) & (df_vis['class'].isin([0, 1, 2, 3]))].copy() # 33008_s1(m2p=0) 22018_s2(base=0)

df_part_invis = df_invis[
    (df_invis['base_ff'] == 1) & (df_invis['class'].isin([0, 1, 2, 3]))].copy()
# df_part_invis = df_invis[
#     (df_invis['base_ff'] == 1) & (df_invis['is_blurry'] == 0) & (df_invis['class'].isin([0, 1, 2, 3]))].copy()
print(len(df_part_vis['file'].unique().tolist()))
print(len(df_part_invis['file'].unique().tolist()))

df_double_lin = df_double[df_double['annotator'] == 'Lin']
df_double_daen = df_double[df_double['annotator'] == 'Daen']
df_double_ronald = df_double[df_double['annotator'] == 'Ronald']
# print(len(df_double['file'].unique().tolist()))
# print(len(df_double_lin))
# print(len(df_double_lin['file'].unique().tolist()))
# print(len(df_double_daen))
# print(len(df_double_daen['file'].unique().tolist()))
# print(len(df_double_ronald))
# print(len(df_double_ronald['file'].unique().tolist()))

double_file = df_double['file'].unique().tolist()
df_all = pd.concat([df_part_vis, df_part_invis], axis=0)

df_group_vis = df_part_vis.groupby(['file', 'event_group'])[[
    'file', 'class', 'l2cs_pred_class', 'base_class', 'm2prop_class']].apply(lambda x: x.mode().iloc[0])
print(df_group_vis)
df_group_invis = df_part_invis.groupby(['file', 'event_group'])[[
    'file', 'class', 'l2cs_pred_class', 'm2prop_class']].apply(lambda x: x.mode().iloc[0])
# print(df_group_invis)
# df_group = df_all.groupby(['file', 'event_group'])['class'].max()"""

"""
num_matches_l2cs = 0
num_matches_m2p = 0
l2cs_list = []
m2p_list = []
# Iterate through the rows of the DataFrame
for f in df_group_invis['file'].unique():
    df_group_invis_single = df_group_invis[df_group_invis['file'] == f]
    for i in range(len(df_group_invis_single)):
        # Compare the "class" value with the corresponding "group" value
        if df_group_invis_single["class"].iloc[i] == df_group_invis_single["l2cs_pred_class"].iloc[i]:
            num_matches_l2cs += 1
        if df_group_invis_single["class"].iloc[i] == df_group_invis_single["m2prop_class"].iloc[i]:
            num_matches_m2p += 1

    # Calculate the matching percentage
    matching_percentage_l2cs = round((num_matches_l2cs / len(df_group_invis)) * 100, 2)
    matching_percentage_m2p = round((num_matches_m2p / len(df_group_invis)) * 100, 2)
    l2cs_list.append(matching_percentage_l2cs)
    m2p_list.append(matching_percentage_m2p)

df_result_invis = pd.DataFrame(columns=['file', 'l2cs_pct', 'm2p_pct'])
df_result_invis['file'] = df_group_invis['file'].unique().tolist()
df_result_invis['l2cs_pct'] = l2cs_list
df_result_invis['m2p_pct'] = m2p_list
df_result_invis.to_csv('df_test_result.csv', index=False)

print(f"l2cs Matching Percentage: {matching_percentage_l2cs:.2f}%")
print(f"m2prop Matching Percentage: {matching_percentage_m2p:.2f}%")
"""

# """
# visible case for line plot of percentage
match_l2cs_vis = []
match_m2p_vis = []
match_base = []
for f in df_group_vis['file'].unique():
    df_group_vis_single = df_group_vis[df_group_vis['file'] == f]
    match_l2cs_vis.append(
        round((df_group_vis_single["class"] == df_group_vis_single["l2cs_pred_class"]).sum() / len(df_group_vis_single["class"]) * 100, 2))
    match_base.append(round((df_group_vis_single["class"] == df_group_vis_single["base_class"]).sum() / len(df_group_vis_single["class"]) * 100, 2))
    match_m2p_vis.append(round((df_group_vis_single["class"] == df_group_vis_single["m2prop_class"]).sum() / len(df_group_vis_single["class"]) * 100, 2))

# Melt the DataFrame to create a suitable format for the Seaborn plot
# melted_df = df_group_invis.melt(id_vars=["file"], value_vars=["matching_l2cs", "matching_m2prop"],
#                     var_name="method", value_name="Percentage")
df_result_vis = pd.DataFrame()
df_result_vis['file'] = df_group_vis['file'].unique().tolist()
# print(df_result_vis['file'])
# df_result_vis['l2cs'] = match_l2cs_vis
df_result_vis['l2cs_m2p'] = match_m2p_vis
df_result_vis['baseline'] = match_base
df_melt_vis = df_result_vis.melt(id_vars=["file"], value_vars=["l2cs_m2p", "baseline"],
                                 var_name="method", value_name="Percentage")
# line plot for visible case
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(x="file", y="Percentage", hue="method", data=df_melt_vis)
plt.xlabel("File")
plt.ylabel("Percentage")
plt.title("Matching Percentages in each visible video file")
plt.xticks(rotation=75, ha='right')
plt.ylim(0, 100)  # Set the y-axis limits
plt.tight_layout()
plt.show()

# visible case for count the percentages
# match_l2cs_vis_s = sorted(match_l2cs_vis)
# match_m2p_vis_s = sorted(match_m2p_vis)
# match_base_s = sorted(match_base)
# df_result_vis['l2cs_sorted'] = match_l2cs_vis_s
# df_result_vis['m2prop_sorted'] = match_m2p_vis_s
# df_result_vis['baseline_sorted'] = match_base_s
# Calculate the counts for each list of percentages
counts_l2cs_vis = calculate_counts(match_l2cs_vis)
counts_m2p_vis = calculate_counts(match_m2p_vis)
counts_base = calculate_counts(match_base)
print(counts_l2cs_vis)
# # Define the x-axis labels for the histogram
x_labels = [i * 5 + 2.5 for i in range(20)]

# df_result_vis_count = pd.DataFrame()
# # df_result_vis_count['l2cs_m2p'] = counts_m2p_vis
# # df_result_vis_count['baseline'] = counts_base
# # df_melt_vis_count = df_result_vis_count.melt(id_vars=x_labels, value_vars=["l2cs_m2p", "baseline"],
# #                                  var_name="method", value_name="Count")
# # plot the histogram for visible case percentage
# # plt.plot(x_labels, counts_m2p_vis, label="L2CS_m2p", marker='o')
# # plt.plot(x_labels, counts_base, label="Baseline", marker='o')
# plt.hist([match_m2p_vis, match_base], bins=50, histtype='step', label=['l2cs_m2p', 'baseline'])
# # plt.hist([match_m2p_vis, match_base], bins=range(0, 105, 5), alpha=0.7, histtype='step', label=['l2cs_m2p', 'baseline'])
# # plt.hist([counts_l2cs_vis, counts_m2p_vis, counts_base], bins=range(0, 105, 5), alpha=0.7, histtype='step', label=['l2cs', 'm2prop', 'baseline'])
# # plt.hist(np.array(counts_m2p_vis), bins=range(0, 105, 5), edgecolor='orange', histtype='step', label='m2prop')
# # plt.hist(np.array(counts_base), bins=range(0, 105, 5), edgecolor='blue', histtype='step', label='baseline')
# plt.xlabel("Percentage (%)")
# plt.ylabel("Count")
# plt.title("Count of visible videos on matching percentage")
# plt.legend()
# plt.tight_layout()
# plt.show()
# """
# """
# invisible case for line plot of percentage
match_l2cs_invis = []
match_m2p_invis = []
for f in df_group_invis['file'].unique():
    df_group_invis_single = df_group_invis[df_group_invis['file'] == f]
    match_l2cs_invis.append(round((df_group_invis_single["class"] == df_group_invis_single["l2cs_pred_class"]).sum() / len(df_group_invis_single["class"]) * 100, 2))
    match_m2p_invis.append(round((df_group_invis_single["class"] == df_group_invis_single["m2prop_class"]).sum() / len(df_group_invis_single["class"]) * 100, 2))

# Melt the DataFrame to create a suitable format for the Seaborn plot
# melted_df = df_group_invis.melt(id_vars=["file"], value_vars=["matching_l2cs", "matching_m2prop"],
#                     var_name="method", value_name="Percentage")
df_result_invis = pd.DataFrame()
df_result_invis['file'] = df_group_invis['file'].unique().tolist()
df_result_invis['l2cs'] = match_l2cs_invis
df_result_invis['m2prop'] = match_m2p_invis
df_melt_invis = df_result_invis.melt(id_vars=["file"], value_vars=["l2cs", "m2prop"],
                                 var_name="method", value_name="Percentage")

# sns.set(style="whitegrid")
# plt.figure(figsize=(10, 6))
# sns.lineplot(x="file", y="Percentage", hue="method", data=df_melt_invis)
# plt.xlabel("File")
# plt.ylabel("Percentage")
# plt.title("Matching Percentages in each invisible video file")
# plt.xticks(rotation=75, ha='right')
# plt.ylim(0, 100)  # Set the y-axis limits
# plt.tight_layout()
# plt.show()
# """
# """
# plot the histogram for invisible case percentage
counts_l2cs_invis = calculate_counts(match_l2cs_invis)
counts_m2p_invis = calculate_counts(match_m2p_invis)
print(counts_l2cs_invis)
# # # Define the x-axis labels for the histogram
# x_labels = [i * 5 + 2.5 for i in range(21)]
# plt.plot(x_labels, counts_l2cs_invis, label="L2CS", marker='o')
# plt.plot(x_labels, counts_m2p_invis, label="m2prop", marker='o')
# # plt.hist([match_l2cs_invis, match_m2p_invis], bins=range(0, 105, 5), alpha=0.7, histtype='step', label=['l2cs', 'm2prop'])
# plt.xlabel("Percentage (%)")
# plt.ylabel("Count")
# plt.title("Count of invisible videos on matching percentage")
# plt.legend()
# plt.tight_layout()
# plt.show()
