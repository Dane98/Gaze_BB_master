
# Inspired by https://towardsdatascience.com/inter-rater-agreement-kappas-69cd8b91ff75
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import numpy as np
import os
import csv

TABLE = 0
ROBOT = 1
TABLET = 2
ELSEWHERE = 3
UNKNOWN = 4

df_manual = pd.read_csv('./annotations_frame.csv')
df_l2cs = pd.read_csv('./l2cs.csv')

"""
# # # calculate ckappa without 4
dfkappa = pd.DataFrame(columns=['file', 'case', 'ckappa'])
for f in df_manual['file'].unique():
    fcase = list(df_manual[df_manual['file'] == f]['case'])[0]
    rater1 = list(df_manual[df_manual['file'] == f]['class'])
    rater2 = list(df_l2cs[df_l2cs['filename'] == f]['class'])

    rater1 = np.array(rater1).astype(int)
    rater1[rater1==4] = 3
    rater2 = np.array(rater2).astype(int)
    rater2[rater2==4] = 3

    cohenk= cohen_kappa_score(rater1, rater2, labels=[TABLE, ROBOT, TABLET, ELSEWHERE, UNKNOWN])
    dfkappa.loc[len(dfkappa)] = [f, fcase, cohenk]

print(dfkappa['ckappa'].mean())
print(dfkappa['ckappa'].max())
dfkappa.to_csv('ckappa_ml_no4.csv')"""

# # # this block is for testing with two files
# filelist = ['51007_sessie1_taskrobotEngagement', '51007_sessie2_taskrobotEngagement']
# for ff in filelist:
#     df_manual_test = df_manual[df_manual['file'] == ff]
#     df_l2cs_test = df_l2cs[df_l2cs['filename'] == ff]
#     common_parts = pd.merge(df_manual_test, df_l2cs_test, on=['framenumber', 'class'], how='inner')
#     common_percentage = "%.4f" % (len(common_parts) / len(df_manual_test))
#     print(common_percentage)
#     with open('inner_ml_with4_test.csv', 'a') as file:
#         writer = csv.writer(file)
#         writer.writerow([ff, common_percentage])
"""
# # # this block is for calculating the inner join manual and l2cs for common part percentage (with class 4)
inner_with4 = pd.DataFrame(columns=['file', 'case', 'percentage'])
for f in df_manual['file'].unique():
    df_manual_single = df_manual[df_manual['file'] == f]
    df_l2cs_single = df_l2cs[df_l2cs['file'] == f]
    case = df_manual_single['case'].iloc[0]
    # print(case)
    common_parts = pd.merge(df_manual_single, df_l2cs_single, on=['framenumber', 'class'], how='inner')
    common_parts.to_csv(os.path.join('./common_manual_l2cs/with4/', 'inner_ml_' + f + '.csv'))
    # print(len(common_parts))
    common_percentage = "%.4f" % (len(common_parts) / len(df_manual_single))
    # print(common_percentage)
    print(f"The percentage of overlapping classes: {float(common_percentage) * 100}%")
    with open('./common_manual_l2cs/inner_ml_with4.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow([f, case, common_percentage])

# # # this block is for calculating the inner join manual and l2cs for common part percentage (without class 4)
inner_without4 = pd.DataFrame(columns=['file', 'case', 'percentage'])
for f in df_manual['file'].unique():
    df_manual_single = df_manual[df_manual['file'] == f]
    df_l2cs_single = df_l2cs[df_l2cs['file'] == f]
    case = df_manual_single['case'].iloc[0]
    # print(case)
    df_m = df_manual_single.copy()
    df_l = df_l2cs_single.copy()
    rater1 = list(df_m['class'])
    rater2 = list(df_l['class'])
    rater1 = np.array(rater1).astype(int)
    rater1[rater1 == 4] = 3
    rater2 = np.array(rater2).astype(int)
    rater2[rater2 == 4] = 3
    df_m.drop(['class'], axis=1)
    df_l.drop(['class'], axis=1)
    df_m['class'] = rater1.tolist()
    df_l['class'] = rater2.tolist()

    common_parts = pd.merge(df_m, df_l, on=['framenumber', 'class'], how='inner')
    common_parts.to_csv(os.path.join('./common_manual_l2cs/without4/', 'inner_ml_' + f + '_no4.csv'))
    # print(len(common_parts))
    common_percentage = "%.4f" % (len(common_parts) / len(df_manual_single))
    # print(common_percentage)
    print(f"The percentage of overlapping classes: {float(common_percentage) * 100}%")
    with open('./common_manual_l2cs/inner_ml_no4.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow([f, case, common_percentage])

inner_with4 = pd.read_csv('./common_manual_l2cs/inner_ml_with4.csv')
print(inner_with4.iloc[1])"""

# Calculate the overlapping percentage
# df_manual_51007 = df_manual[df_manual['file'] == file].reset_index()
# df_l2cs_51007 = df_l2cs[df_l2cs['filename'] == file].reset_index()
# # df_manual_51007.to_csv('manual_51007.csv')
# # df_l2cs_51007.to_csv('l2cs_51007.csv')
# df_manual_class = df_manual_51007[['framenumber', 'class']]
# df_l2cs_class = df_l2cs_51007[['framenumber', 'class']]
# df_manual_class.to_csv('manual_class.csv')
# df_l2cs_class.to_csv('l2cs_class.csv')

# overlap = df_manual_51007[df_manual_51007['class'].isin(df_l2cs_51007['class'])]
# overlap = df_manual_class[df_manual_class['class'].astype(int).isin(df_l2cs_class['class'].astype(int))]
# print(len(overlap))
# overlap_percentage = (len(overlap) / len(df_manual_51007)) * 100
# # overlap.to_csv('inner_manual_l2cs.csv')
# print(f"The percentage of overlapping classes: {overlap_percentage}%")

# dfkappa = pd.DataFrame(columns=['file', 'case', 'ckappa'])
# for f in df_manual['file'].unique():
#     fcase = list(df_manual[df_manual['file'] == f]['case'])[0]
#     manual = list(df_manual[df_manual['file'] == f]['class'])
#     l2cs = list(df_l2cs[df_l2cs['filename'] == f]['class'])
#
#     cohenk = cohen_kappa_score(manual, l2cs, labels=[TABLE, ROBOT, TABLET, ELSEWHERE, UNKNOWN])
#     dfkappa.loc[len(dfkappa)] = [f, fcase, cohenk]
#
# print(dfkappa['ckappa'].mean())
# print(dfkappa['ckappa'].max())
# print(dfkappa)
# dfkappa.to_csv('ckappa_ml.csv')


# dfkappa = pd.DataFrame(columns=['file', 'case', 'ckappa'])
# for f in annotations_double['file'].unique():
#     fcase = list(annotations[annotations['file'] == f]['case'])[0]
#     rater1 = list(annotations[annotations['file'] == f]['class'])
#     rater2 = list(annotations_double[annotations_double['file'] == f]['class'])
#
#     # rater1 = np.array(rater1).astype(int)
#     # rater1[rater1==4] = 3
#     # rater2 = np.array(rater2).astype(int)
#     # rater2[rater2==4] = 3
#
#     cohenk= cohen_kappa_score(rater1, rater2, labels=[TABLE, ROBOT, TABLET, ELSEWHERE, UNKNOWN])
#     dfkappa.loc[len(dfkappa)] = [f, fcase, cohenk]
#
#
# print(dfkappa['ckappa'].mean())
# dfkappa.to_csv('ckappa_scores.csv')


# df_manual_class = df_manual_51007['class']
# df_l2cs_class = df_l2cs_51007['class']
# # df_manual_class.to_csv('manual_class_only.csv')
# # df_l2cs_class.to_csv('l2cs_class_only.csv')
#
# # overlap = df_manual_51007[df_manual_51007['class'].isin(df_l2cs_51007['class'])]
# overlap = df_manual_class[df_manual_class['class'].isin(df_l2cs_class['class'])]
# print(len(overlap))

df4 = pd.read_csv('./common_manual_l2cs/inner_ml_with4.csv', names=['file', 'case', 'p'])
df = pd.read_csv('./common_manual_l2cs/inner_ml_no4.csv', names=['file', 'case', 'p'])

mean4 = "%.4f" % float(df4['p'].mean())
max4 = "%.4f" % float(df4['p'].max())
min4 = "%.4f" % float(df4['p'].min())
max4_file = df4[df4['p']==df4['p'].max()]['file'].iloc[0]
min4_file = df4[df4['p']==df4['p'].min()]['file'].iloc[0]

me = "%.4f" % float(df['p'].mean())
ma = "%.4f" % float(df['p'].max())
mi = "%.4f" % float(df['p'].min())
ma_file = df[df['p']==df['p'].max()]['file'].iloc[0]
mi_file = df[df['p']==df['p'].min()]['file'].iloc[0]

# print('With class 4 unknow:')
# print(f'The average overlap is about {float(mean4)*100}%')
# print(f'The most matching video is {max4_file}.mp4, it is up to {float(max4)*100}%')
# print(f'The less matching video is {min4_file}.mp4, it is only {float(max4)*100}%')
# print()
# print('Without class 4 unknow:')
# print(f'The average overlap is about {float(me)*100}%')
# print(f'The most matching video is {ma_file}.mp4, it is up to {float(ma)*100}%')
# print(f'The less matching video is {mi_file}.mp4, it is only {float(mi)*100}%')

# for i in len(df4['p'].tolist()):
df_kappa4 = pd.read_csv('ckappa_ml.csv', names=['file', 'case', 'kappa'])
df_kappa = pd.read_csv('ckappa_ml_no4.csv', names=['file', 'case', 'kappa'])

df_kappa4_k = df_kappa4['kappa'].tolist()
# print(type(df_kappa4_k))
print(len(df_kappa4_k))
count = 0
for i in range(len(df_kappa4_k)):
    if float(df_kappa4_k[i]) >= 0.4:
        print(f"{df_kappa4_k[i]}")
        count+=1
print(f"There are {count} videos got a cohen's kappa score above 0.4.")

df_kappa_k = df_kappa['k'].tolist()
# print(type(df_kappa4_k))
c = 0
for i in range(len(df_kappa_k)):
    if float(df_kappa_k[i]) >= 0.4:
        print(df_kappa_k[i])
        c+=1
print(c)