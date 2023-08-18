import os
from os.path import join, isfile, splitext
import pandas as pd
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

# visible = './l2cs/visible/'
# invisible = './l2cs/invisible/'
# visible = os.path.join('./baseline/vis_new/')
# invisible = os.path.join('./baseline/object_new/')
# object_inv = os.path.join('./baseline/object_new/')
cols = ['yaw', 'pitch', 'm2_class', 'm2prop_class', 'flt_m2_class', 'flt_prop_class', 'face']
visible = os.path.join('../bb/pitchjaw/visible_filtered_m2prop/')
invisible = '../bb/pitchjaw/invisible_filtered_m2prop/'
# cols = ['yaw', 'pitch', 'pred_class', 'filtered_class', 'base_face', 'l2cs_face']
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

df['framenumber'] = frame_number
df['case'] = filecase
df['file'] = filename
df.to_csv('./results/l2cs_m2_vis.csv', index=False)
# df.to_csv('./results/base_vis.csv', index=False)

# df = pd.read_csv('./results/l2cs_m2_vis.csv')
# for file in os.listdir(invisible):
#     file_path = join(invisible + file)
#     if isfile(file_path):
#         f = splitext(file)
#         data = pd.read_csv(file_path, header=0)
#         df = pd.concat([df, data], axis=0)
#         df = df.reset_index(drop=True)
#         filename.extend([f'{f[0]}'] * len(data))
#         filecase.extend(['invisible'] * len(data))
#         frame_number.extend(range(len(data)))
#
# df['framenumber'] = frame_number
# df['case'] = filecase
# df['file'] = filename
#
# df.to_csv('./results/l2cs_m2.csv', index=False)
# # df.to_csv('./results/base_object.csv', index=False)
