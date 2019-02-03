import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import pickle
from baselines.common import plot_util

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
Y_REWARD = 'reward'
Y_TIMESTEPS = 'timesteps'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'darkred', 'darkblue']

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def window_func(x, y, window, func):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x[window-1:], yw_func

def ts2xy(ts, xaxis, yaxis):
    if xaxis == X_TIMESTEPS:
        x = np.cumsum(ts.l.values)
    elif xaxis == X_EPISODES:
        x = np.arange(len(ts))
    elif xaxis == X_WALLTIME:
        x = ts.t.values / 3600.
    else:
        raise NotImplementedError
    if yaxis == Y_REWARD:
        y = ts.r.values
    elif yaxis == Y_TIMESTEPS:
        y = ts.l.values
    else:
        raise NotImplementedError
    return x, y

def plot_curves(xy_list, xaxis, yaxis, title):
    fig = plt.figure(figsize=(8,2))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    for (i, (x, y)) in enumerate(xy_list):
        color = COLORS[i % len(COLORS)]
        plt.scatter(x, y, s=2)
        x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
        plt.plot(x, y_mean, color=color)
    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.tight_layout()
    fig.canvas.mpl_connect('resize_event', lambda event: plt.tight_layout())
    plt.grid(True)


def split_by_task(taskpath):
    return taskpath.dirname.split('/')[-1].split('-')[0]
def default_split_fn(r):
    import re
    # match name between slash and -<digits> at the end of the string
    # (slash in the beginning or -<digits> in the end or either may be missing)
    return r.dirname.split('-')[-3]
def plot_results(dirs, num_timesteps=10e6, xaxis=X_TIMESTEPS, yaxis=Y_REWARD, title='', split_fn=split_by_task):
    results = plot_util.load_results(dirs)
    # plot_curves(results,'x','y','t')
    plot_util.plot_results(results, xy_fn=lambda r: ts2xy(r.monitor, xaxis, yaxis), split_fn=None, group_fn=default_split_fn, average_group=True, resample=100,smooth_step=1,shaded_std=False)



# Example usage in jupyter-notebook
# from baselines.results_plotter import plot_results
# %matplotlib inline
# plot_results("./log")
# Here ./log is a directory containing the monitor.csv files

def main():
    sns.set()
    import argparse
    import os
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dirs', help='List of log directories', nargs = '*', default=['./log'])
    parser.add_argument('--num_timesteps', type=int, default=int(10e6))
    parser.add_argument('--xaxis', help = 'Varible on X-axis', default = X_TIMESTEPS)
    parser.add_argument('--yaxis', help = 'Varible on Y-axis', default = Y_REWARD)
    parser.add_argument('--task_name', help = 'Title of plot', default = 'Breakout')
    args = parser.parse_args()
    args.dirs = [os.path.abspath(dir) for dir in args.dirs]
    plot_results(args.dirs, args.num_timesteps, args.xaxis, args.yaxis, args.task_name)
    # plt.ylim([-1000,2000])
    font = {'weight': 'normal',
             'size': 25,
             }
    plt.subplots_adjust(top=0.93, bottom=0.13, left=0.14, right=0.95)
    plt.xlabel('Timesteps',font)
    plt.ylabel('Cumulative Return', font)
    # plt.xlabel('')

    plt.savefig('C:/Users/zzhuang1/Desktop/tmp_last/3')
    plt.show()

if __name__ == '__main__':
    main()