import os
from os.path import join, isfile, splitext
import pandas as pd


# df_manual = pd.read_csv('./results/annotations_frame.csv', header=0, names=['file', 'case', 'framenumber', 'class'])
# df_l2cs = pd.read_csv('./results/l2cs.csv', header=0,
#                       names=['yaw', 'pitch', 'pred_class', 'filtered_class', 'base_face', 'l2cs_face', 'framenumber',
#                              'case', 'file'])
# # df_base = pd.read_csv('./results/base_vis.csv', header=0, names=['yaw', 'pitch', 'face', 'class', 'framenumber',
# #                              'case', 'file'])
# df_base_vis = pd.read_csv('./results/base_vis.csv', header=0, names=['yaw', 'pitch', 'face', 'class', 'framenumber',
#                                                                      'case', 'file'])
# df_base_invis = pd.read_csv('./results/base_object.csv', header=0,
#                             names=['yaw', 'pitch', 'face', 'class', 'framenumber', 'case', 'file'])
#
# df_manual_vis = df_manual[df_manual['case'] == 'visible']
# df_l2cs_vis = df_l2cs[df_l2cs['case'] == 'visible']
# df_manual_invis = df_manual[df_manual['case'] == 'invisible']
# df_l2cs_invis = df_l2cs[df_l2cs['case'] == 'invisible']

# print(df_l2cs_vis['base_face'].equals(df_base_vis['face']))
# print(df_l2cs_vis['l2cs_face'].equals(df_base_vis['face']))
# print(df_l2cs_invis['base_face'].equals(df_base_invis['face']))
# print(df_l2cs_invis['l2cs_face'].equals(df_base_invis['face']))
#
# print(df_l2cs_vis['base_face'].iloc[0])
# print(df_base_vis['face'].iloc[0])
# print(type(df_l2cs_vis['base_face'].iloc[0]))
# print(type(df_base_vis['face'].iloc[0]))

# print(df_manual_vis.head())
# print(len(df_base_vis)) # 137853
# print(len(df_l2cs_vis)) # 137853
# print(len(df_manual_vis)) # 137853
# print(len(df_manual_invis)) # 139409
"""
# visible case for m2
df_ml_event_vis = pd.read_csv('./results/ml_vis_filtered_face_th2_blur_event.csv', header=0)
df_m2_vis = pd.read_csv('./results/l2cs_m2_vis.csv', header=0)
print(df_ml_event_vis.head())

df_ml_event_vis = df_ml_event_vis.copy()
m2_class = []
m2prop_class = []
flt_m2_class = []
flt_prop_class = []
for f in df_ml_event_vis['file'].unique():
    manual_f_vis = df_ml_event_vis[df_ml_event_vis['file'] == f]
    m2_f_vis = df_m2_vis[df_m2_vis['file'] == f]
    manual_size = manual_f_vis['class'].size
    m2_size = m2_f_vis['m2_class'].size
    # m2_prop_size = df_m2_vis['m2prop_class'].size
    if manual_size == m2_size:
        m2_class.extend(m2_f_vis['m2_class'].tolist())
        m2prop_class.extend(m2_f_vis['m2prop_class'].tolist())
        flt_m2_class.extend(m2_f_vis['flt_m2_class'].tolist())
        flt_prop_class.extend(m2_f_vis['flt_prop_class'].tolist())
        print(f'l2cs m2 predicted classes for visible file {f} are #{len(m2_class)}')
    else:
        print(f'manual size for class is {manual_size}, l2cs m2_class is {m2_size}')
        m2_class.extend([9] * manual_f_vis.size)
        m2prop_class.extend([9] * manual_f_vis.size)
        flt_m2_class.extend([9] * manual_f_vis.size)
        flt_prop_class.extend([9] * manual_f_vis.size)

df_ml_event_vis['m2_class'] = m2_class
df_ml_event_vis['m2prop_class'] = m2prop_class
df_ml_event_vis['flt_m2_class'] = flt_m2_class
df_ml_event_vis['flt_prop_class'] = flt_prop_class
df_ml_event_vis.to_csv('./results/mlm2_event_vis_merge.csv', index=False)
"""

# this block is to merge the invisible case for manual and l2cs m2.
df_ml_event_invis = pd.read_csv('./results/ml_invis_filtered_face_th2_blur_event.csv', header=0)
df_m2_vis = pd.read_csv('./results/l2cs_m2.csv', header=0)
df_m2_invis = df_m2_vis[df_m2_vis['case'] == 'invisible']

df_m2_invis = df_m2_invis.copy()
print(f'df_m2_invis has {len(df_m2_invis)} rows.')
m2_list_invis = []
m2prop_list_invis = []
flt_m2_list_invis = []
flt_prop_list_invis = []
for f in df_ml_event_invis['file'].unique():
    manual_f_invis = df_ml_event_invis[df_ml_event_invis['file'] == f]
    m2_f_invis = df_m2_invis[df_m2_invis['file'] == f]
    manual_size = manual_f_invis['class'].size
    m2_size = m2_f_invis['m2_class'].size
    if manual_size == m2_size:
        m2_list_invis.extend(m2_f_invis['m2_class'].tolist())
        m2prop_list_invis.extend(m2_f_invis['m2prop_class'].tolist())
        flt_m2_list_invis.extend(m2_f_invis['flt_m2_class'].tolist())
        flt_prop_list_invis.extend(m2_f_invis['flt_prop_class'].tolist())
        print(f'l2cs m2 predicted classes for invisible file {f} are #{len(m2_list_invis)}')
    else:
        print(
            f'manual size for class is {manual_size}, l2cs pred_class is {m2_size}')
        m2_list_invis.extend([9] * manual_f_invis.size)
        m2prop_list_invis.extend([9] * manual_f_invis.size)
        flt_m2_list_invis.extend([9] * manual_f_invis.size)
        flt_prop_list_invis.extend([9] * manual_f_invis.size)

df_ml_event_invis['m2_class'] = m2_list_invis
df_ml_event_invis['m2prop_class'] = m2prop_list_invis
df_ml_event_invis['flt_m2_class'] = flt_m2_list_invis
df_ml_event_invis['flt_prop_class'] = flt_prop_list_invis
print(f'after merge, df_manual_invis has {df_ml_event_invis.shape[0]} rows.')
df_ml_event_invis.to_csv('./results/ml_invis_merge_all.csv', index=False)


"""
# ############################ visible case ##############################
##############################
# these two blocks are for all classes (with class 4 unknown)
# this block is to merge the visible case for manual and l2cs.
df_manual_vis = df_manual_vis.copy()
print(f'df_manual_vis has {len(df_manual_vis)} rows.')
l2cs_class_list_vis = []
l2cs_class_list_vis_filtered = []
for f in df_manual_vis['file'].unique():
    manual_f_vis = df_manual_vis[df_manual_vis['file'] == f]
    l2cs_f_vis = df_l2cs_vis[df_l2cs_vis['file'] == f]
    manual_size = manual_f_vis['class'].size
    pred_l2cs_size = l2cs_f_vis['pred_class'].size
    filtered_l2cs_size = l2cs_f_vis['filtered_class'].size
    if manual_size == pred_l2cs_size:
        l2cs_class_list_vis.extend(l2cs_f_vis['pred_class'].tolist())
        l2cs_class_list_vis_filtered.extend(l2cs_f_vis['filtered_class'].tolist())
        print(f'l2cs predicted classes for visible file {f} are #{len(l2cs_class_list_vis)}')
        print(f'filtered classes for visible file {f} in l2cs are # {len(l2cs_class_list_vis_filtered)}')
    else:
        print(
            f'manual size for class is {manual_size}, l2cs pred_class is {pred_l2cs_size}, filtered class is {filtered_l2cs_size}')
        l2cs_class_list_vis.extend([9] * manual_f_vis.size)
        l2cs_class_list_vis_filtered.extend([9] * manual_f_vis.size)

df_manual_vis['l2cs_pred_class'] = l2cs_class_list_vis
df_manual_vis['l2cs_filtered_class'] = l2cs_class_list_vis_filtered
print(f'after merge, df_manual_vis has {df_manual_vis.shape[0]} rows.')
df_manual_vis.to_csv('./results/ml_vis_merge.csv', index=False)

# this block is to merge the visible case for baseline and (manual_l2cs).
df_ml_vis = pd.read_csv('./results/ml_vis_merge.csv', header=0)
print(f'df_ml_vis has {len(df_ml_vis)} rows.')
base_class_list_vis = []
for f in df_ml_vis['file'].unique():
    ml_f_vis = df_ml_vis[df_ml_vis['file'] == f]
    base_f_vis = df_base_vis[df_base_vis['file'] == f]
    ml_size = ml_f_vis['class'].size
    base_size = base_f_vis['class'].size
    if ml_size == base_size:
        base_class_list_vis.extend(base_f_vis['class'].tolist())
        print(f'classes for visible file {f} in baseline is # {len(base_class_list_vis)}')
    else:
        base_class_list_vis.extend([9] * ml_f_vis.size)
        print(f'file {f} in manual annotation has {ml_size} labels, but has {base_size} labels in baseline.')

df_ml_vis['base_class'] = base_class_list_vis
print(f'after merge, df_ml_vis has {df_ml_vis.shape[0]} rows.')
df_ml_vis.to_csv('./results/mlb_vis_merge.csv', index=False)
##############################

##############################
# this block is for merging face (visible)
df_mlb_vis = pd.read_csv('./results/mlb_vis_merge.csv', header=0)
print(f'df_manual_vis has {len(df_manual_vis)} rows.')
base_face_list_vis = []
l2cs_face_list_vis = []
for f in df_mlb_vis['file'].unique():
    base_f_vis_face = df_l2cs_vis[df_l2cs_vis['file'] == f]['base_face'].tolist()
    l2cs_f_vis_face = df_l2cs_vis[df_l2cs_vis['file'] == f]['l2cs_face'].tolist()
    base_face_list_vis.extend(base_f_vis_face)
    l2cs_face_list_vis.extend(l2cs_f_vis_face)
    print(f'faces for visible file {f} in l2cs is # {len(base_f_vis_face)}')

df_mlb_vis['base_face'] = base_face_list_vis
df_mlb_vis['l2cs_face'] = l2cs_face_list_vis
print(f'after merge, df_mlb_vis has {df_mlb_vis.shape[0]} rows.')
df_mlb_vis.to_csv('./results/mlb_face_vis_merge.csv', index=False)
##############################
"""
"""
# ############################ invisible case ##############################
##############################
# this block is to merge the invisible case for manual and l2cs.
df_manual_invis = df_manual_invis.copy()
print(f'df_manual_invis has {len(df_manual_invis)} rows.')
l2cs_class_list_invis = []
l2cs_class_list_invis_filtered = []
for f in df_manual_invis['file'].unique():
    manual_f_invis = df_manual_invis[df_manual_invis['file'] == f]
    l2cs_f_invis = df_l2cs_invis[df_l2cs_invis['file'] == f]
    manual_size = manual_f_invis['class'].size
    pred_l2cs_size = l2cs_f_invis['pred_class'].size
    filtered_l2cs_size = l2cs_f_invis['filtered_class'].size
    if manual_size == pred_l2cs_size:
        l2cs_class_list_invis.extend(l2cs_f_invis['pred_class'].tolist())
        l2cs_class_list_invis_filtered.extend(l2cs_f_invis['filtered_class'].tolist())
        print(f'l2cs predicted classes for invisible file {f} are #{len(l2cs_class_list_invis)}')
        print(f'filtered classes for invisible file {f} in l2cs are # {len(l2cs_class_list_invis_filtered)}')
    else:
        print(
            f'manual size for class is {manual_size}, l2cs pred_class is {pred_l2cs_size}, filtered class is {filtered_l2cs_size}')
        l2cs_class_list_invis.extend([9] * manual_f_invis.size)
        l2cs_class_list_invis_filtered.extend([9] * manual_f_invis.size)

df_manual_invis['l2cs_pred_class'] = l2cs_class_list_invis
df_manual_invis['l2cs_filtered_class'] = l2cs_class_list_invis_filtered
print(f'after merge, df_manual_invis has {df_manual_invis.shape[0]} rows.')
df_manual_invis.to_csv('./results/ml_invis_merge.csv', index=False)

# this block is for merging face (invisible)
df_ml_invis = pd.read_csv('./results/ml_invis_merge.csv', header=0)
print(f'df_manual_invis has {len(df_manual_invis)} rows.')
base_face_list_invis = []
l2cs_face_list_invis = []
for f in df_ml_invis['file'].unique():
    base_f_invis_face = df_l2cs_invis[df_l2cs_invis['file'] == f]['base_face'].tolist()
    l2cs_f_invis_face = df_l2cs_invis[df_l2cs_invis['file'] == f]['l2cs_face'].tolist()
    base_face_list_invis.extend(base_f_invis_face)
    l2cs_face_list_invis.extend(l2cs_f_invis_face)
    print(f'faces for invisible file {f} in l2cs is # {len(base_f_invis_face)}')

df_ml_invis['base_face'] = base_face_list_invis
df_ml_invis['l2cs_face'] = l2cs_face_list_invis
print(f'after merge, df_ml_invis has {df_ml_invis.shape[0]} rows.')
df_ml_invis.to_csv('./results/ml_face_invis_merge.csv', index=False)
##############################
"""


# df1_s = df1[df1['file'] == '33013_sessie3_robotEngagement3']
# print(len(df1['class']))
# print(len(df1_s['class']))
# 33013_sessie3_robotEngagement3
# df1 = pd.read_csv('./l2cs/manual_test.csv', header=0)
# df2 = pd.read_csv('./l2cs/l2cs_test.csv', header=0)
# print(df1.head())
# print(df2.head())
# df1.insert(4, 'l2cs_class', df2['class'])
# df1.to_csv('./l2cs/add.csv', index=False)

# common_parts_12 = pd.merge(df1, df2, on=['framenumber', 'class'], how='inner')
# common_parts_23 = pd.merge(df2, df3, on=['framenumber', 'class'], how='inner')
# common_parts_23_face = pd.merge(df2, df3, on=['yaw', 'class', 'face'], how='inner')
# common_parts_123 = pd.merge(common_parts_12, df3, on=['yaw', 'class'], how='inner')

# all_parts_12 = pd.merge(df1, df2, how='outer')
# all_parts_23 = pd.merge(df2, df3, how='outer')
# common_parts_23_face = pd.merge(df2, df3, on=['yaw', 'class', 'face'], how='outer')
# all_parts_123 = pd.merge(all_parts_12, df3, how='outer')

# common_parts_12.to_csv('./l2cs/inner_ml_test.csv', index=False)
# common_parts_23.to_csv('./l2cs/inner23.csv', index=False)
# common_parts_23_face.to_csv('./l2cs/inner23_face.csv', index=False)
# common_parts_123.to_csv('./l2cs/inner123.csv', index=False)
#
# all_parts_12.to_csv('./l2cs/outer_ml_test.csv', index=False)
# all_parts_23.to_csv('./l2cs/outer23.csv', index=False)
# all_parts_123.to_csv('./l2cs/outer123.csv', index=False)

"""
annotators = ['Daen', 'Lin', 'Ronald']
files = [n+'/elan_annotations.csv' for n in annotators]
cols = ['tier', 'beginmm', 'begin','endmm', 'end', 'diffmm', 'diff', 'class', 'file']
df = pd.DataFrame()
for f in files:
    data = pd.read_csv(f, names=cols)
    df = pd.concat([df, data], axis=0)

# df.columns = cols
df.to_csv('annotations.csv', index=False)


#double
double_files = ['double/'+n+'/elan_annotations_double.csv' for n in annotators]
df2 = pd.DataFrame()
for f in double_files:
    data = pd.read_csv(f, names=cols)
    df2 = pd.concat([df2, data], axis=0)

# df2.columns = cols
df2.to_csv('annotations_double.csv', index=False)"""
"""
visible = './l2cs/visible/'
invisible = './l2cs/invisible/'
cols = ['yaw', 'pitch', 'class']
df = pd.DataFrame()

filename = []
filecase = []
frame_number = []

for file in os.listdir(visible):
    file_path = join(visible + file)
    if isfile(file_path):
        f = splitext(file)
        data = pd.read_csv(file_path, header=0)
        df = pd.concat([df, data], axis=0)
        df = df.reset_index(drop=True)
        filename.extend([f'{f[0]}'] * len(data))
        filecase.extend(['visible'] * len(data))
        frame_number.extend(range(len(data)))
        # print(df)
        # print(len(data))
        # print(len(filename))
        # print(filename)
        # print(frame_number)
        # print()
df.to_csv('l2cs_vis.csv', index=False)

df = pd.read_csv('l2cs_vis.csv')
for file in os.listdir(invisible):
    file_path = join(invisible + file)
    if isfile(file_path):
        f = splitext(file)
        data = pd.read_csv(file_path, header=0)
        df = pd.concat([df, data], axis=0)
        df = df.reset_index(drop=True)
        filename.extend([f'{f[0]}'] * len(data))
        filecase.extend(['invisible'] * len(data))
        frame_number.extend(range(len(data)))

df['framenumber'] = frame_number
df['case'] = filecase
df['file'] = filename

df.to_csv('l2cs.csv', index=False)"""
