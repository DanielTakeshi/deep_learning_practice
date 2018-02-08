"""
Plots results. Adjust names from `plot_names.py`. Assumes ordering:

epoch | l2_loss (v) | ce_loss (v) | valid_err (s) | valid_err (m) | test_err (s) | test_err (m)

for each line. This is for plotting 'coarse' tuning.
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
title_size = 22
tick_size = 18
legend_size = 18
ysize = 20
xsize = 20
lw, ms = 3, 8

# ADJUST
HPARAMS = {
    'lrate': ['0.01', '0.05', '0.1', '0.3', '0.5'],
    'wd':    ['0.0', '0.000001', '0.00001', '0.0001', '0.001'],
}


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
                #print("found info for {}, at line {}, w/{} data points".format(
                #        ff, idx, len(all_lines)))
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


def get_row_index(head):
    num_lr = len(HPARAMS['lrate'])
    num_wd = len(HPARAMS['wd'])
    for idx,lr in enumerate(HPARAMS['lrate']):
        if 'lrate-{}-'.format(lr) in head:
            lr_idx = idx
    for idx,wd in enumerate(HPARAMS['wd']):
        if 'wd-{}-'.format(wd) in head:
            wd_idx = idx
    row_idx = lr_idx + num_lr*wd_idx
    return row_idx


def get_rank(row_tuples, row):
    for idx,element in enumerate(row_tuples):
        if row == element[0]:
            return idx
    print("something bad happened")
    sys.exit()


def axarr_plot(axarr, row, col, xcoords, mean, std, name):
    axarr[row,col].plot(xcoords, mean, lw=lw, label=name)
    axarr[row,col].fill_between(xcoords, mean-std, mean+std,
            alpha=error_region_alpha)


def plot_one_type(headname, figname):
    """
    First column, validation, second test. There is a lot of information to
    process. We'll have to form a ranking and add to the plot titles.
    """
    dirs = sorted([e for e in os.listdir(headname) if 'seed' in e])
    unique_dirs = sorted( list( set([x[:-1].replace('seed-','') for x in dirs]) ) )
    print("\nPlotting one figure with {} files".format(len(dirs)))
    print("and {} unique stems".format(len(unique_dirs)))
    nrows = len(HPARAMS['lrate']) * len(HPARAMS['wd'])
    ncols = 2
    fig,ax = plt.subplots(nrows, ncols, figsize=(10*ncols,10*nrows))

    # Let's first get *validation rankings* for these (shouldn't be using test).
    LIM1 = 100-1
    LIM2 = 200-1
    ranks = defaultdict(list)

    for head in unique_dirs:
        info = parse(head, headname, dirs)
        row = get_row_index(head)
        print("Currently on head {} w/row idx {}".format(head, row))

        # Validation, 100.
        ranks['row_to_s_100'].append(   (row, info['valid_err_s_mean'][LIM1-1]) )
        ranks['row_to_m_100'].append(   (row, info['valid_err_m_mean'][LIM1-1]) )
        # Validation, 200.
        ranks['row_to_s_200'].append(   (row, info['valid_err_s_mean'][LIM2-1]) )
        ranks['row_to_m_200'].append(   (row, info['valid_err_m_mean'][LIM2-1]) )
        # Test, 100.
        ranks['row_to_s_100_t'].append( (row, info['test_err_s_mean'][LIM1-1]) )
        ranks['row_to_m_100_t'].append( (row, info['test_err_m_mean'][LIM1-1]) )
        # Test, 200.
        ranks['row_to_s_200_t'].append( (row, info['test_err_s_mean'][LIM2-1]) )
        ranks['row_to_m_200_t'].append( (row, info['test_err_m_mean'][LIM2-1]) )

    # Order the rankings.
    for key in ranks:
        ranks[key] = sorted(ranks[key], key=lambda x: x[1])

    # Now loop again to plot, this time using the rankings we've stored.
    for head in unique_dirs:
        info = parse(head, headname, dirs)
        row = get_row_index(head)

        # Validation, single and model.
        valid_s_info = "single-ep100-{:.3f}-ep200-{:.3f}".format(
            info['valid_err_s_mean'][LIM1], info['valid_err_s_mean'][LIM2],
        )
        valid_m_info = "model-ep100-{:.3f}-ep200-{:.3f}".format(
            info['valid_err_m_mean'][LIM1], info['valid_err_m_mean'][LIM2],
        )

        # Test, single and model.
        test_s_info = "single-ep100-{:.3f}-ep200-{:.3f}".format(
            info['test_err_s_mean'][LIM1], info['test_err_s_mean'][LIM2],
        )
        test_m_info = "model-ep100-{:.3f}-ep200-{:.3f}".format(
            info['test_err_m_mean'][LIM1], info['test_err_m_mean'][LIM2],
        )

        # Add validation to plot, column 0.
        axarr_plot(ax, row, 0, info['x'],
                   info['valid_err_s_mean'],
                   info['valid_err_s_std'],
                   name=head+valid_s_info)
        axarr_plot(ax, row, 0, info['x'],
                   info['valid_err_m_mean'],
                   info['valid_err_m_std'],
                   name=head+valid_m_info)

        # Add test to plot, colum 1.
        axarr_plot(ax, row, 1, info['x'],
                   info['test_err_s_mean'],
                   info['test_err_s_std'],
                   name=head+test_s_info)
        axarr_plot(ax, row, 1, info['x'],
                   info['test_err_m_mean'],
                   info['test_err_m_std'],
                   name=head+test_m_info)

        # Titles
        r1 = get_rank(ranks['row_to_s_100'], row)
        r2 = get_rank(ranks['row_to_m_100'], row)
        r3 = get_rank(ranks['row_to_s_200'], row)
        r4 = get_rank(ranks['row_to_m_200'], row)
        r5 = get_rank(ranks['row_to_s_100_t'], row)
        r6 = get_rank(ranks['row_to_m_100_t'], row)
        r7 = get_rank(ranks['row_to_s_200_t'], row)
        r8 = get_rank(ranks['row_to_m_200_t'], row)
        title1 = 's100-m100-s200-m200r-{}-{}-{}-{}'.format(r1,r2,r3,r4)
        title2 = 's100-m100-s200-m200r-{}-{}-{}-{}'.format(r5,r6,r7,r8)
        ax[row,0].set_title('Valid-'+title1, fontsize=title_size)
        ax[row,1].set_title('Test-'+title2,  fontsize=title_size)

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
            elif col == 1:
                ax[row,col].set_ylabel("Test Error % (10k digits)", fontsize=ysize)
            ax[row,col].axvline(x=LIM1, ls='--', color='darkblue')
            ax[row,col].axvline(x=LIM2, ls='--', color='darkblue')
    plt.tight_layout()
    plt.savefig(figname)


if __name__ == "__main__":
    plot_one_type('logs/sgd-tune/', "figures/tune_sgd_coarse.png")
