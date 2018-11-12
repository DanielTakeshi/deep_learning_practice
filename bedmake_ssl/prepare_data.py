import argparse, copy, cv2, os, sys, pickle, time
import numpy as np
from os.path import join


def prepare_ryan_data():
    """
    Create appropriate data for PyTorch, from Ryan's tentative data collection.
    As usual, need a way to provide an index into the correct file name/target.
    BTW, for training and validation splits, we should split based on episodes,
    not on images, as the data is actually (s_t, a_t, s_{t+1}). Think of it as
    iterating through actions, not images, but we have to _skip_ indices which
    correspond to any `None` actions; those indicate episode transitions.
    """
    raw_data = 'ssldata/rollout.pkl'
    dir_pyt = 'ssldata_pytorch'
    assert not os.path.exists(dir_pyt), "target exists:\n\t{}".format(dir_pyt)
    os.makedirs(dir_pyt)
    path_train = join(dir_pyt,'train')
    path_valid = join(dir_pyt,'valid')
    os.makedirs(path_train)
    os.makedirs(path_valid)
    total_train = 0
    total_valid = 0

    # For data loader, need to go from index to target, for BOTH train and valid.
    loader_train_path = join(dir_pyt,'train','data_train_loader.pkl')
    loader_valid_path = join(dir_pyt,'valid','data_valid_loader.pkl')
    loader_train_dict = []
    loader_valid_dict = []

    # Put numbers here for computing the normalization statistics.
    numbers_0 = []
    numbers_1 = []
    numbers_2 = []

    # FOR NOW, temporary. We know the last episode is 'held out'.
    idx_to_skip = [0, 20, 40, 60]
    train_cutoff = 60

    # For Ryan's data it's a simple pickle file.
    with open(raw_data, 'r') as fh:
        data = pickle.load(fh)
        N = len(data)
        print("Just loaded: {}  (len: {})".format(raw_data, N))

        # Each `item` has 'image' and 'action' keys.
        # Due to the (s_t, a_t, s_{t+1}) nature of data, we should think of
        # this as iterating through actions. The actions are dicts like this:
        # {'y': 302, 'x': 475, 'length': 20, 'angle': 0}
        # The `t` here is synced with `c_img_t.png`, fyi. We skip 0.
        for t in range(N-1):
            if t in idx_to_skip:
                print("skipping {}".format(t))
                continue
            s_t   = data[t]['image']
            a_t   = data[t]['action']
            s_tp1 = data[t+1]['image']

            # Accumulate statistics for mean and std computation across our
            # lone channel. We made values same across all three channels.
            assert s_t.shape == s_tp1.shape == (480,640,3)
            numbers_0.extend( s_t[:,:,0].flatten() )
            numbers_1.extend( s_t[:,:,1].flatten() )
            numbers_2.extend( s_t[:,:,2].flatten() )

            # Don't forget! Add info to our data loaders!! We need enough info
            # to determine a full data point, which is a pair: `(input,target)`.
            png_t   = 'd_img_{}'.format(str(t).zfill(3))
            png_tp1 = 'd_img_{}'.format(str(t).zfill(3))

            # TODO? Fix action representation?

            if t <= train_cutoff:
                loader_train_dict.append( (png_t, png_tp1, a_t) )
                total_train += 1
            else:
                loader_valid_dict.append( (png_t, png_tp1, a_t) )
                total_valid += 1

    assert len(loader_train_dict) == total_train
    assert len(loader_valid_dict) == total_valid

    with open(loader_train_path, 'w') as fh:
        pickle.dump(loader_train_dict, fh)
    with open(loader_valid_path, 'w') as fh:
        pickle.dump(loader_valid_dict, fh)

    print("done loading data, train {} & valid {} (total {})".format(
            total_train, total_valid, total_train+total_valid))
    numbers = np.array([numbers_0,numbers_1,numbers_2]) # Will be shape (3,D)
    print("numbers.shape: {}  (for channel mean/std)".format(numbers.shape))
    print("mean(numbers): {}".format(np.mean(numbers, axis=1)))
    print("std(numbers):  {}".format(np.std(numbers, axis=1)))
    print("\nBut, use this for actual mean/std because we want them in [0,256) ...")
    print("mean(scaled): {}".format(np.mean(numbers/255.0, axis=1)))
    print("std(scaled):  {}".format(np.std(numbers/255.0, axis=1)))


if __name__ == "__main__":
    prepare_ryan_data()
