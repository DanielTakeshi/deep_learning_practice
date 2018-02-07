"""
Plots results. Adjust names from `plot_names.py`. Assumes ordering:

epoch | l2_loss (v) | ce_loss (v) | valid_err (s) | valid_err (m) | test_err (s) | test_err (m)

for each line
"""

import argparse, matplotlib, os, pickle, sys
matplotlib.use('Agg')
matplotlib.rcParams['axes.color_cycle'] = ['red', 'blue', 'yellow', 'black', 'purple']
import matplotlib.pyplot as plt
import plot_names as pn
import numpy as np
np.set_printoptions(edgeitems=100, linewidth=100, suppress=True)
from collections import defaultdict

# Some matplotlib settings.
plt.style.use('seaborn-darkgrid')
error_region_alpha = 0.20
LOGDIR = 'experiments/logs/'
FIGDIR = 'experiments/figures/'
title_size = 22
tick_size = 18
legend_size = 18
ysize = 20
xsize = 20
lw, ms = 3, 8


def parse(file_head, headname, dirs):
    """
    Parse line based on the pattern we know.  Makes key assumption that we can
    assume that `seed-x` is placed AT THE END. And that `len(x)==1`.

    Very research-code like. Don't judge.
    """
    files = sorted([x for x in dirs if file_head in x])
    info = defaultdict(list)

    for ff in files:
        path = headname + ff
        with open(path, 'r') as f:
            all_lines = [x.strip('\n') for x in f.readlines()]
        idx = 0
        while True:
            if 'epoch | l2_loss (v)' in all_lines[idx]:
                all_lines = all_lines[idx+1:]
                print("found info for {}, at line {}, w/{} data points".format(
                        ff, idx, len(all_lines)))
                break
            idx += 1
            if idx > 50:
                raise Exception()
        all_stuff = [x.split() for x in all_lines]
        for idx in range(len(all_stuff)):
            all_stuff[idx] = [float(x) for x in all_stuff[idx]]
        results = np.array(all_stuff)
        assert len(results.shape) == 2 and results.shape[0] == len(all_lines)

        # Yeah it's ugly we have to know these ...
        info['l2_loss_v'].append(results[:,1])
        info['ce_loss_v'].append(results[:,2])
        info['valid_err_s'].append(results[:,3])
        info['valid_err_m'].append(results[:,4])
        info['test_err_s'].append(results[:,5])
        info['test_err_m'].append(results[:,6])

    # Turn into numpy arrays and collect mean/std information.
    keys = info.keys()
    for key in list(keys):
        info[key] = np.array(info[key])
        assert len(info[key].shape) == 2 and info[key].shape[0] == len(files)
        info[key+'_mean'] = np.mean(info[key], axis=0)
        info[key+'_std']  = np.std(info[key], axis=0)
    info['x'] = np.arange(len(all_lines))
    return info


def plot_one_type_5x5(headname, lrates, figname):
    """
    First column, validation, second test.  Assumes we did 5x5 evaluation of
    learning rates and weight decays. The 5x5 isn't the number of subplots.
    """
    dirs = sorted([e for e in os.listdir(headname) if 'seed' in e])
    unique_dirs = sorted( list( set([x[:-1].replace('seed-','') for x in dirs]) ) )
    print("\nPlotting one figure with {} files".format(len(dirs)))
    print("and {} unique stems".format(len(unique_dirs)))
    nrows, ncols = len(lrates), 2
    fig,ax = plt.subplots(nrows, ncols, figsize=(5*nrows,20*ncols))

    def axarr_plot(axarr, row, col, xcoords, mean, std, name):
        axarr[row,col].plot(xcoords, mean, lw=lw, label=name)
        axarr[row,col].fill_between(xcoords, mean-std, mean+std,
                alpha=error_region_alpha)

    int1,int2 = 50,100

    for head in unique_dirs:
        print("\nCurrently on head {}".format(head))
        info = parse(head, headname, dirs)
        row = 0
        for lr in lrates:
            if 'lrate-{}-'.format(lr) in head:
                row = lrates[lr]
        print("for {} row is {}".format(head, row))

        # Only do the 100th point and the avg from 50 to 100 due to prior work
        valid_info = "one-{:.3f}-avg-{:.3f}".format(
                info['valid_err_s_mean'][int2-1],
                np.mean(info['valid_err_s_mean'][int1:int2]))
        test_info  = "one-{:.3f}-avg-{:.3f}".format(
                info['test_err_s_mean'][int2-1],
                np.mean(info['test_err_s_mean'][int1:int2]))

        axarr_plot(ax, row, 0, info['x'],
                   info['valid_err_s_mean'],
                   info['valid_err_s_std'],
                   name=head+valid_info)
        axarr_plot(ax, row, 1, info['x'],
                   info['test_err_s_mean'],
                   info['test_err_s_std'],
                   name=head+test_info)

    # Bells and whistles
    for row in range(nrows):
        for col in range(ncols):
            ax[row,col].tick_params(axis='x', labelsize=tick_size)
            ax[row,col].tick_params(axis='y', labelsize=tick_size)
            ax[row,col].legend(loc="best", prop={'size':legend_size})
            ax[row,col].set_ylim([1.00, 4.00])
            ax[row,col].set_xlabel("Epochs (55k digits each)", fontsize=xsize)
            if col == 0:
                ax[row,col].set_ylabel("Valid Error % (5k digits)", fontsize=ysize)
                ax[row,col].set_title('Valid Error', fontsize=title_size)
            elif col == 1:
                ax[row,col].set_ylabel("Test Error % (10k digits)", fontsize=ysize)
                ax[row,col].set_title('Test Error', fontsize=title_size)
            ax[row,col].axvline(x=int1, ls='--', color='darkblue')
            ax[row,col].axvline(x=int2, ls='--', color='darkblue')
    plt.tight_layout()
    plt.savefig(figname)


if __name__ == "__main__":
    lrates = {'0.01':0, '0.05':1, '0.1':2, '0.3':3, '0.5':4}
    plot_one_type_5x5('logs/sgd-tune/',    lrates, "figures/tune_sgd_coarse.png")
    plot_one_type_5x5('logs/momsgd-tune/', lrates, "figures/tune_momsgd_coarse.png")
