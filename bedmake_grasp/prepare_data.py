"""After `python prepare_data.py`, I get:

Just loaded: /nfs/diskstation/seita/bed-make/cache_combo_v03/grasp_list_of_dicts_nodaug_cv_0_len_210.pkl (len: 210)
Just loaded: /nfs/diskstation/seita/bed-make/cache_combo_v03/grasp_list_of_dicts_nodaug_cv_1_len_210.pkl (len: 210)
Just loaded: /nfs/diskstation/seita/bed-make/cache_combo_v03/grasp_list_of_dicts_nodaug_cv_2_len_210.pkl (len: 210)
Just loaded: /nfs/diskstation/seita/bed-make/cache_combo_v03/grasp_list_of_dicts_nodaug_cv_3_len_210.pkl (len: 210) 
Just loaded: /nfs/diskstation/seita/bed-make/cache_combo_v03/grasp_list_of_dicts_nodaug_cv_4_len_210.pkl (len: 210)
Just loaded: /nfs/diskstation/seita/bed-make/cache_combo_v03/grasp_list_of_dicts_nodaug_cv_5_len_210.pkl (len: 210)
Just loaded: /nfs/diskstation/seita/bed-make/cache_combo_v03/grasp_list_of_dicts_nodaug_cv_6_len_210.pkl (len: 210)
Just loaded: /nfs/diskstation/seita/bed-make/cache_combo_v03/grasp_list_of_dicts_nodaug_cv_7_len_209.pkl (len: 209)
Just loaded: /nfs/diskstation/seita/bed-make/cache_combo_v03/grasp_list_of_dicts_nodaug_cv_8_len_209.pkl (len: 209)
Just loaded: /nfs/diskstation/seita/bed-make/cache_combo_v03/grasp_list_of_dicts_nodaug_cv_9_len_209.pkl (len: 209)
done loading data, train 1677 & valid 420 (total 2097)
len(numbers):  644198400  (has the single-channel mean/std info)
mean(numbers): 95.5442559777
std(numbers):  84.812095941

But, use this for actual mean/std because we want them in [0,256) ...
mean(scaled): 0.374683356775
std(scaled):  0.33259645467
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
from torchvision import datasets, transforms
import argparse, copy, cv2, os, sys, pickle, time
import numpy as np
from os.path import join

# ------------------------------------------------------------------------------
# Target is where we re-format the data for PyTorch convenience methods.
HEAD   = '/nfs/diskstation/seita/bed-make/cache_combo_v03'

# Move locally after data creation! Can get up to 10x speed-up!!
#TARGET = '/nfs/diskstation/seita/bed-make/cache_combo_v03_pytorch'
TARGET = 'cache_combo_v03_pytorch'
# ------------------------------------------------------------------------------


def prepare_raw_data():
    """Create appropriate data for PyTorch. Delete target directory if needed.
    
    Unlike in classification, we don't need to create a complicated directory
    system, but I think it's helpful to have the images saved with the targets
    in a separate file, following the data loading tutorial.

    We need a way to provide an index into the correct file name and target.
    """
    assert not os.path.exists(TARGET), "target directory exists:\n\t{}".format(TARGET)
    os.makedirs(TARGET)
    path_train = join(TARGET,'train')
    path_valid = join(TARGET,'valid')
    os.makedirs(path_train)
    os.makedirs(path_valid)
    total_train = 0
    total_valid = 0

    # For data loader, need to go from index to target, for BOTH train and valid.
    loader_train_path = join(TARGET,'train','data_train_loader.pkl')
    loader_valid_path = join(TARGET,'valid','data_valid_loader.pkl')
    loader_train_dict = []
    loader_valid_dict = []

    # Load the pickle files that I used for the bed-making paper.
    pickle_files = sorted([join(HEAD,x) for x in os.listdir(HEAD) if x[-4:] == '.pkl'])
    total_pickles = len(pickle_files)

    # Put all numbers here. For PyTorch we can use one scalar for each of the
    # mean and std, because we have one scalar here (for our depth images).
    numbers = []

    for p_idx,p_ff in enumerate(pickle_files):
        with open(p_ff, 'r') as fh:
            data = pickle.load(fh)
            N = len(data)
            print("Just loaded: {}  (len: {})".format(p_ff, N))

            # Pick validation indices.
            indx_random = np.random.permutation(N)
            indx_train  = indx_random[ : int(N*0.8)]
            indx_valid  = indx_random[int(N*0.8) : ]

            # Each `item` here has a 'd_img' key, and a target key, 'pose'.
            for idx,item in enumerate(data):
                if idx in indx_train:
                    pname = path_train
                else:
                    pname = path_valid

                target_str = "{}-{}".format(item['pose'][0], item['pose'][1])
                target_tuple = (item['pose'][0], item['pose'][1])

                suffix = 'd_{}_{}_{}.png'.format(str(p_idx).zfill(2), str(idx).zfill(4), target_str)
                png_name = join(pname, suffix)

                # Accumulate statistics for mean and std computation across our
                # lone channel. We made values same across all three channels.
                d_img = item['d_img']
                assert d_img.shape == (480,640,3)
                assert np.sum(d_img[:,:,0]) == np.sum(d_img[:,:,1]) == np.sum(d_img[:,:,2])
                numbers.extend( d_img[:,:,0].flatten() )
                cv2.imwrite(png_name, d_img)

                # Don't forget! Add info to our data loaders!!
                if idx in indx_train:
                    loader_train_dict.append( (png_name, target_tuple) )
                    total_train += 1
                else:
                    loader_valid_dict.append( (png_name, target_tuple) )
                    total_valid += 1

    assert len(loader_train_dict) == total_train
    assert len(loader_valid_dict) == total_valid

    with open(loader_train_path, 'w') as fh:
        pickle.dump(loader_train_dict, fh)
    with open(loader_valid_path, 'w') as fh:
        pickle.dump(loader_valid_dict, fh)

    print("done loading data, train {} & valid {} (total {})".format(
            total_train, total_valid, total_train+total_valid))
    numbers = np.array(numbers)
    print("len(numbers):  {}  (has the single-channel mean/std info)".format(len(numbers)))
    print("mean(numbers): {}".format(np.mean(numbers)))
    print("std(numbers):  {}".format(np.std(numbers)))
    print("\nBut, use this for actual mean/std because we want them in [0,256) ...")
    print("mean(scaled): {}".format(np.mean(numbers/255.0)))
    print("std(scaled):  {}".format(np.std(numbers/255.0)))


if __name__ == "__main__":
    prepare_raw_data()
