import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join


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


def plot(filename):
    df = pd.read_csv(filename)

    df['class'] = df['class'].astype(int)
    classes = df['class'].astype(int).unique()

    # df_vid = pd.read_csv('vidstat.csv')
    # nr_frames = df_vid.loc[df_vid['file'] == filename, 'frames'].iloc[0].astype(int)
    # fps = df_vid.loc[df_vid['file'] == filename, 'rate'].iloc[0].astype(int)
    # fps = df_vid.query('file == Proefpersoon22016_Sessie1')['rate'].astype(int)

    # bin = fps * 3
    nr_frames = len(df.index)
    x = range(0, nr_frames)

    # print(f'bin: {bin}')
    # print(type(bin))

    aggregated_class = df['class'].copy()
    df_class = aggregated_class.groupby(aggregated_class.index // 1).transform(lambda a: a.value_counts().idxmax())

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
    plt.savefig(join(plot_path, f[:-4] + '_gaze.png'), dpi=300)
    # plt.savefig(join('./annotation_plot/plot/Proefpersoon22016_Sessie1_gaze_aggr.png'), dpi=300)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    plot('./annotation_plot/33008_sessie1_taskrobotEngagement.csv')
    # root = './annotation_plot/'
    # files = [f for f in listdir(root) if isfile(join(root, f))]
    # plot_path = './annotation_plot/plot/org/'
    #
    # for f in files:
    #     plot(root + f)

    # # #plot l2cs in visible case and invisible case
    # file_path = './pitchjaw/visible/'
    # plot_path = './plot/visible/'
    # vis_files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    # for f in vis_files:
    #     plot(file_path + f)
    #
    # file_path = './pitchjaw/invisible/'
    # plot_path = './plot/invisible/'
    # invis_files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    # for f in invis_files:
    #     plot(file_path + f)
