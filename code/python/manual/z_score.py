import os
import pandas as pd
import numpy as np
from scipy import stats


# df_manual = pd.read_csv('./manual_51007.csv', header=0, names=['file', 'case', 'framenumber', 'class'])
df_l2cs = pd.read_csv('./l2cs_51007.csv', header=0, names=['yaw', 'pitch', 'class', 'framenumber', 'case', 'file'])

data_l_array = np.array(df_l2cs[['yaw', 'pitch']])
print(data_l_array)
#
# mean = np.mean(data)
# std_dev = np.std(data)
#
# z_threshold = 2
# z_scores = (data - mean) / std_dev
#
# outliers = data[np.abs(z_scores) > z_threshold]
# filtered_data = data[np.abs(z_scores) <= z_threshold]
