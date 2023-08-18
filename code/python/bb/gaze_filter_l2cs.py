import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

from math import *
from os import listdir
from os.path import isfile, join
from filterpy.kalman import KalmanFilter


# gaussian function
def f(mu, sigma2, x):
    ''' f takes in a mean and squared variance, and an input x
       and returns the gaussian value.'''
    coefficient = 1.0 / sqrt(2.0 * pi * sigma2)
    exponential = exp(-0.5 * (x - mu) ** 2 / sigma2)
    return coefficient * exponential


# the update function
def update(mean1, var1, mean2, var2):
    ''' This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters.'''
    # Calculate the new parameters
    new_mean = (var2 * mean1 + var1 * mean2) / (var2 + var1)
    new_var = 1 / (1 / var2 + 1 / var1)

    return [new_mean, new_var]


# update(20,9,30,3)


# the motion update/predict function
def predict(mean1, var1, mean2, var2):
    ''' This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters, after motion.'''
    # Calculate the new parameters
    new_mean = mean1 + mean2
    new_var = var1 + var2

    return [new_mean, new_var]


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
        return 'blue'
    elif id == 1:
        return 'green'
    elif id == 2:
        return 'red'
    elif id == 3:
        return 'orange'
    elif id == 4:
        return 'violet'


def klm(data):
    # Create a simple Kalman filter
    kf = KalmanFilter(dim_x=1, dim_z=1)  # 1-dimensional state and measurement

    # Define the state transition matrix
    kf.F = np.array([[1.]])  # Identity matrix since we're dealing with a single value

    # Define the measurement matrix
    kf.H = np.array([[1.]])  # Identity matrix since we're directly measuring the state

    # Define the process noise covariance
    kf.Q = np.array([[0.01]])  # Adjust the value based on the noise level in the system

    # Define the measurement noise covariance
    kf.R = np.array([[1.]])  # Adjust the value based on the noise level in the measurements

    # Initialize the state and covariance matrix
    kf.x = np.array([[0.]])  # Initial state estimate
    kf.P = np.array([[1.]])  # Initial state covariance

    # Create an array of measurements
    measurements = np.array(data)  # Replace with your own array

    # Perform the filtering
    filtered_values = []
    for measurement in measurements:
        kf.predict()  # Predict the next state
        kf.update(measurement)  # Update the state based on the measurement
        filtered_values.append(kf.x[0, 0])  # Store the filtered value

    # print(filtered_values)
    return filtered_values


def filter(filename):
    df = pd.read_csv(filename)

    df['class'] = df['class'].astype(int)
    classes = df['class'].astype(int).unique()

    df_vid = pd.read_csv('vidstat.csv')
    # nr_frames = df_vid.loc[df_vid['file'] == filename, 'frames'].iloc[0].astype(int)
    # fps = df_vid.loc[df_vid['file'] == filename, 'rate'].iloc[0].astype(int)
    # fps = df_vid.query('file == Proefpersoon22016_Sessie1')['rate'].astype(int)

    # bin = fps * 3
    nr_frames = len(df.index)
    x = range(0, nr_frames)

    # print(f'bin: {bin}')
    # print(type(bin))

    aggregated_class = df['class'].copy()
    df_class = aggregated_class.groupby(aggregated_class.index // 300).transform(lambda a: a.value_counts().idxmax())
    # most = df['class'][0:30].mode()[0]

    bars = []
    bottoms = []
    for c in classes:
        arr = 1 * np.array(df_class == c)
        bars.append(arr)
        bottom = np.ones(nr_frames).astype(int)
        bottoms.append(bottom)

    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    fig, ax = plt.subplots()

    labels = [get_classname(i) for i in classes]
    colors = [get_color(i) for i in classes]

    for b, bot, col in zip(bars, bottoms, colors):
        ax.bar(x, height=b, width=1, bottom=bot, color=col)

    ax.set_ylim((1, 3))
    ax.set_xlim((0, nr_frames))
    ax.set_yticks([1.5])
    ax.set_yticklabels(["classes"])
    ax.set_ylabel('gazed upon object', fontsize=12)
    ax.set_xlabel('frame', fontsize=12)
    fig.tight_layout()
    plt.legend(labels)
    # plt.savefig(join(plot_path, filename[:-4] + '_gaze.png'), dpi=300)
    plt.savefig(join('./plot/visible/Proefpersoon22016_Sessie1_gaze_aggr300.png'), dpi=300)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    file = './pitchjaw/visible/Proefpersoon22016_Sessie1.csv'
    df = pd.read_csv(file)

    # klm(df)
    # yaw = klm(df['yaw'])
    # pitch = klm(df['pitch'])
    # # writing the data into the file
    # with open('test_py.csv', 'w+', newline='') as f:
    #     write = csv.writer(f)
    #     write.writerows(zip(map(lambda x: x, yaw), map(lambda y: y, pitch)))

    # file2 = 'test_yaw01.csv'
    # df2 = pd.read_csv(file2)
    #
    # plt.rcParams["figure.figsize"] = [7.00, 3.50]
    # plt.rcParams["figure.autolayout"] = True
    #
    # ax = df['yaw'].plot(y='yaw')
    # df2.plot(ax=ax, y='f01')
    #
    # plt.show()

    file2 = 'test_py.csv'
    df2 = pd.read_csv(file2)

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True

    # ax = df[['yaw', 'pitch']].plot('r.', x='pitch', y='yaw')
    # df2.plot('b.', ax=ax, x='f_pitch', y='f_yaw')

    plt.plot(df[['yaw', 'pitch']], 'r.')
    plt.plot(df2, 'b.')
    plt.show()
