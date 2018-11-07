import copy, cv2, os, sys, pickle, time
import numpy as np
from os.path import join

TARGET = 'tmp/'
RAW_PICKLE_FILE = 'data_raw_115_items.pkl'

def prepare_data():
    """Create the appropriate data for PyTorch using `ImageFolder`. From:

    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    https://pytorch.org/docs/stable/torchvision/datasets.html?highlight=imagefolder#torchvision.datasets.ImageFolder
    https://discuss.pytorch.org/t/questions-about-imagefolder/774

    If you need to call this again, delete the TARGET directory.
    """
    assert not os.path.exists(TARGET), "target directory exists:\n\t{}".format(TARGET)
    os.makedirs(TARGET)
    paths = ['train', 'valid']
    path_train = join(TARGET,paths[0])
    path_valid = join(TARGET,paths[1])
    for p in paths:
        os.makedirs(join(TARGET,p))
        os.makedirs(join(TARGET,p,'success'))
        os.makedirs(join(TARGET,p,'failure'))
    t_success = 0
    t_failure = 0

    # Put all numbers here. For PyTorch we can use one scalar for each of the
    # mean and std, because we have one scalar here (for our depth images),
    # whose values are 'triplicated' across all three channels.
    numbers = []

    with open(RAW_PICKLE_FILE, 'r') as fh:
        data = pickle.load(fh)

        # Pick validation indices.
        N = len(data)
        indx_random = np.random.permutation(N)
        indx_train  = indx_random[ : int(N*0.8)]
        indx_valid  = indx_random[int(N*0.8) : ]

        # Each `item` here has a 'd_img' key, and a class label 'class' key.
        for idx,item in enumerate(data):
            if idx in indx_train:
                pname = path_train
            else:
                pname = path_valid

            if item['class'] == 0:
                png_name = join(pname, 'success', 'd_{}.png'.format(str(idx).zfill(4)))
                t_success += 1
            elif item['class'] == 1:
                png_name = join(pname, 'failure', 'd_{}.png'.format(str(idx).zfill(4)))
                t_failure += 1
            else:
                raise ValueError(item['class'])
            cv2.imwrite(png_name, item['d_img'])

            # Accumulate statistics for mean and std computation across our
            # lone channel. We made values same across all three channels.
            d_img = item['d_img']
            assert d_img.shape == (480,640,3)
            assert np.sum(d_img[:,:,0]) == np.sum(d_img[:,:,1]) == np.sum(d_img[:,:,2])
            numbers.extend( d_img[:,:,0].flatten() )

    print("done loading data, success {} vs failure {} (total {})".format(
            t_success, t_failure, N))
    numbers = np.array(numbers)
    print("len(numbers):  {}  (has the single-channel mean/std info)".format(len(numbers)))
    print("mean(numbers): {}".format(np.mean(numbers)))
    print("std(numbers):  {}".format(np.std(numbers)))

    # I doubt it matters too much but, I divide by 255 to try and match what
    # ToTensor() does, since they put values in [0,1], so 1 is a possibility
    # (hence, we need 255/255, and not 255/256).
    print("\nBut, use this for actual mean/std because we want them in [0,256) ...")
    print("mean(scaled): {}".format(np.mean(numbers/255.0)))
    print("std(scaled):  {}".format(np.std(numbers/255.0)))


if __name__ == "__main__":
    prepare_data()
