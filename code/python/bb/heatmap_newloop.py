# import numpy as np
# import scipy.stats as stats
# from matplotlib import pyplot as plt
# import pandas as pd
# import time
# from IPython import display
#
# root = 'E:/Master/MasterThesis/L2CS-Net-main/output/'  # csv file root
# fn = 'simple_mytest6_30.csv'  # csv file name
# vn = 'simple_mytest6_30.avi'  # video file name
# plot_out_path = 'E:/Master/MasterThesis/L2CS-Net-main/heatmap/plots/simple_mytest6_30/20s/'  # path to save plots
#
# f = root + fn
# df1 = pd.read_csv(f)[['yaw', 'pitch']]
# df = df1.copy()
# df.rename(columns={'yaw': 'pitch', 'pitch': 'yaw'}, inplace=True)
# df = df.dropna()
# # print(df.head())
#
#
# df1 = pd.read_csv(f)[['pitch', 'yaw']] / 3.14 * 180
#
# df1 = df1.dropna()
# pause_time = 0.05
# for xi in range(0, 800):
#     dfx = df1[(0 + xi) * 10:(1 + xi) * 10]
#     samples = dfx.to_numpy()
#     kde = stats.kde.gaussian_kde(samples.T)
#     # Regular grid to evaluate kde upon
#     x_flat = np.r_[-100:100]
#     y_flat = np.r_[-100:100]
#     x, y = np.meshgrid(x_flat, y_flat)
#     grid_coords = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)
#     z = kde(grid_coords.T)
#     z = z.reshape(200, 200)
#
#     plt.imshow(z, aspect=x_flat.ptp() / y_flat.ptp(), extent=[-100, 100, -100, 100])
#     plt.title(xi)
#     display.display(plt.gcf())
#     display.clear_output(wait=True)
# #     time.sleep(pause_time)

