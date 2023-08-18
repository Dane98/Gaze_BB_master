import cv2
import numpy as np
import pandas as pd
from os import path
import scipy.stats as stats
import seaborn as sns
from matplotlib import pyplot as plt

# root = '/home/linlincheng/Documents/Projects/L2CS-Net-main/output/'  # csv file root
# fn = 'simple_mytest6.csv'  # csv file name
# vn = 'simple_mytest6.avi'  # video file name
# plot_out_path = '/home/linlincheng/Documents/Projects/L2CS-Net-main/heatmap/plots/simple_mytest6/5s' # path to save plots

# f = 'E:/Master/MasterThesis/L2CS-Net-main/output/simple_mytest6.csv'
root = 'E:/Master/MasterThesis/L2CS-Net-main/output/'  # csv file root
fn = 'simple_mytest6_30.csv'  # csv file name
vn = 'simple_mytest6_30.avi'  # video file name
plot_out_path = 'E:/Master/MasterThesis/L2CS-Net-main/heatmap/plots/simple_mytest6_30/20s/'  # path to save plots

f = root + fn
df1 = pd.read_csv(f)[['yaw', 'pitch']]
df = df1.copy()
df.rename(columns={'yaw': 'pitch', 'pitch': 'yaw'}, inplace=True)
df = df.dropna()
# print(df.head())

# x_max = df[0].max()
# print(x_max)
# y_max = df[1].max()
# print(y_max)

# plt.plot(df['yaw'],df['pitch'],'g.')
# plt.show()
# x=df['yaw']
# y=df['pitch']
# plt.scatter(x, y, s=70, alpha=0.3)
# plt.show()

time_interval = 5  # to plot the heatmap per time interval (seconds)
# num_row = df.shape[0]  # the number of rows in the csv file

# get FPS
v = root + vn
cap = cv2.VideoCapture(v)
fps = cap.get(cv2.CAP_PROP_FPS)
# print(f'num of fps is: {fps}')


# num_records = int(time_interval) * int(fps)  # chunk row size
# print(f'num of records is: {num_records}')
num_records = 150
# list_df = [df[i:i+num_records] for i in range(0,df.shape[0],num_records)]
# list_df = df.iloc[1::num_records, :]  # data frame slices
# samples = df.to_numpy()

# data = np.array(df)                 # convert complete dataset into numpy-array
# print(len(data))
"""
# the first figure
list_df = df[0: num_records]
samples = list_df.to_numpy()
# print(list_df)

kde = stats.gaussian_kde(samples.T)

x_flat = np.r_[samples[:, 0].min():samples[:, 0].max():150j]
y_flat = np.r_[samples[:, 1].min():samples[:, 1].max():150j]
x, y = np.meshgrid(x_flat, y_flat)
grid_coords = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)

z = kde(grid_coords.T)
z = z.reshape(150, 150)

# glue = sns.load_dataset(z).pivot("Yaw", "Pitch", "Score")

# plt.imshow(z, aspect=x_flat.ptp() / y_flat.ptp(), extent=[x_flat.min(), x_flat.max(),y_flat.min(),y_flat.max()])
# plt.show()

fig, ax = plt.subplots()        # generate figure with axes
# ax.imshow(z, aspect='auto', extent=[x_flat.min(), x_flat.max(),y_flat.min(),y_flat.max()])
# ax.xlabel('yaw')
# ax.ylabel('pitch')

# Create a 2D histogram of the gaze data
# heatmap, xedges, yedges = np.histogram2d(x_flat, y_flat, bins=(50, 50))

# im = ax.imshow(z, cmap='hot', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
# ax.imshow(z, extent=[x_flat.min(), x_flat.max(), y_flat.min(), y_flat.max()])

# Set the x- and y-axis labels
ax.set_xlabel('Yaw (degrees)')
ax.set_ylabel('Pitch (degrees)')

# Set the x- and y-axis limits to the same range
ax.set_xlim([min(y_flat), max(y_flat)])
ax.set_ylim([min(x_flat), max(x_flat)])
# sns.heatmap(z)
# Set the colorbar
# cbar = ax.figure.colorbar(im, ax=ax)
# cbar.ax.set_ylabel('Counts', rotation=-90, va="bottom")

plt.draw()
fig.savefig(path.join(plot_out_path, '00000.jpg'))
"""
# multiple figures
x_min = df.loc[df['pitch'].idxmin()][0]
# print(x_min)
y_min = df.loc[df['yaw'].idxmin()][1]
x_max = df.loc[df['pitch'].idxmax()][0]
y_max = df.loc[df['yaw'].idxmax()][1]

for i in range(0, df.shape[0], num_records):
    list_df = df[i: i + num_records]
    samples = list_df.to_numpy()
    # print(list_df)

    fig, ax = plt.subplots()

    kde = stats.gaussian_kde(samples.T)

    x_flat = np.r_[samples[:, 0].min():samples[:, 0].max():150j]
    y_flat = np.r_[samples[:, 1].min():samples[:, 1].max():150j]
    x, y = np.meshgrid(x_flat, y_flat)
    grid_coords = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)

    z = kde(grid_coords.T)
    z = z.reshape(150, 150)

    plt.imshow(z, aspect='auto', extent=[x_min, x_max, y_min, y_max])
    plt.draw()
    # plt.show()
    fig.savefig(path.join(plot_out_path, "%05d.jpg" % i))
    plt.close()
#####
"""
    fig, ax = plt.subplots()  # generate figure with axes
    image, = ax.plot(x, z[0])  # initialize plot
    ax.xlabel('Yaw')
    ax.ylabel('Pitch')
    plt.draw()
    fig.savefig(path.join(plot_out_path, '00000.jpg'))
    plt.close()

    for i in range(1, len(z)):
        image.set_data(x, z[i])
        plt.draw()
        fig.savefig(path.join(plot_out_path, "%05d.jpg" % i))
        plt.close()
#####

    # fig, ax = plt.subplots()  # generate figure with axes
    # image, = ax.plot(x, z[0])  # initialize plot

    # plt.imshow(z, aspect='auto', extent=[x_flat.min(), x_flat.max(),y_flat.min(),y_flat.max()])
    # plt.imshow(z, aspect=x_flat.ptp() / y_flat.ptp(), extent=[-15, 15, -15, 15])

    # ax.set_xlabel('yaw')
    # ax.set_ylabel('pitch')

    # Create a 2D histogram of the gaze data
    heatmap, xedges, yedges = np.histogram2d(x_flat, y_flat, bins=(50, 50))

    im = ax.imshow(heatmap, cmap='hot', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    # Set the x- and y-axis labels
    ax.set_xlabel('Yaw (degrees)')
    ax.set_ylabel('Pitch (degrees)')

    # Set the x- and y-axis limits to the same range
    ax.set_xlim([min(y_flat), max(y_flat)])
    ax.set_ylim([min(x_flat), max(x_flat)])

    # Set the colorbar
    # cbar = ax.figure.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel('Counts', rotation=-90, va="bottom")

    # ax.imshow(z, aspect='auto', extent=[x_flat.min(), x_flat.max(), y_flat.min(), y_flat.max()])
    plt.draw()
    # plt.show()
    fig.savefig(path.join(plot_out_path, "%05d.jpg" % i))
    plt.close()"""

"""
for i in range(0,df.shape[0],num_records):
    plt.figure()
    list_df = df[i:i + num_records]
    samples = list_df.to_numpy()

    kde = stats.gaussian_kde(samples.T)

    x_flat = np.r_[samples[:, 0].min():samples[:, 0].max():5]
    y_flat = np.r_[samples[:, 1].min():samples[:, 1].max():5]
    x, y = np.meshgrid(x_flat, y_flat)
    grid_coords = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)

    z = kde(grid_coords.T)
    z = z.reshape(5, 5)

    plt.imshow(z, aspect=x_flat.ptp() / y_flat.ptp(), extent=[-5, 5, -5, 5])
    plt.savefig("%05d.jpg" % i)
    plt.close()"""

"""
df1=df.dropna()
samples = df1.to_numpy()

kde = stats.gaussian_kde(samples.T)

# Regular grid to evaluate kde upon
x_flat = np.r_[samples[:,0].min():samples[:,0].max():200j]
y_flat = np.r_[samples[:,1].min():samples[:,1].max():200j]
x,y = np.meshgrid(x_flat,y_flat)
grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)

z = kde(grid_coords.T)
z = z.reshape(200,200)

plt.imshow(z,aspect=x_flat.ptp()/y_flat.ptp(),extent=[-50,50,-50,50])
plt.show()"""

"""
from os import path
data = numpy.array(Data_EMG)                 # convert complete dataset into numpy-array
x = pylab.linspace(EMG_start, EMG_stop, Amount_samples) # doesn´t change in loop anyway

outpath = "path/of/your/folder/"

fig, ax = plt.subplots()        # generate figure with axes
image, = ax.plot(x,data[0])     # initialize plot
ax.xlabel('Time(ms)')
ax.ylabel('EMG voltage(microV)')
plt.draw()
fig.savefig(path.join(outpath,"dataname_0.png")

for i in range(1, len(data)):
    image.set_data(x,data[i])
    plt.draw()
    fig.savefig(path.join(outpath,"dataname_{0}.png".format(i))"""
