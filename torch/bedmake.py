"""Let's see if I can train using the bed-making data I have.

After running `prepare_raw_data`, I get:

Just loaded: /nfs/diskstation/seita/bed-make/cache_combo_v01_success/success_list_of_dicts_nodaug_cv_0_len_66.pkl (len: 66)
so far: success 32 vs failure 34
...
Just loaded: /nfs/diskstation/seita/bed-make/cache_combo_v01_success/success_list_of_dicts_nodaug_cv_9_len_65.pkl (len: 65)
so far: success 327 vs failure 327
done loading data, success 327 vs failure 327 (total 654)
len(numbers):  200908800  (has the single-channel mean/std info)
mean(numbers): 96.8104350432
std(numbers):  84.6227108358

I think this makes sense. The depth values were in [0,255] and there's high
standard deviation due to white vs black. But, maybe we should try and scale
into [0,1]? The code for ImageNet folks appear to have done that as values for
the mean are within [0,1].
"""
import torch
import torchvision.models as models
from torchvision import datasets, transforms
import cv2, os, sys, pickle, time
import numpy as np
from os.path import join

# Target is where we re-format the data for PyTorch convenience methods.
# In the `cache` files, I already processed the depth images.
HEAD = '/nfs/diskstation/seita/bed-make/cache_combo_v01_success'
TARGET = '/nfs/diskstation/seita/bed-make/cache_combo_v03_success_pytorch'

# Torch stuff
resnet18 = models.resnet18(pretrained=True)
resnet34 = models.resnet34(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
# ------------------------------------------------------------------------------


def prepare_raw_data():
    """Create the appropriate data for PyTorch.

    In particular, for classification, it's easiest to put them in their own
    separate folders and then to use `ImageFolder`.

    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

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

    pickle_files = sorted([join(HEAD,x) for x in os.listdir(HEAD) if x[-4:] == '.pkl'])
    total_pickles = len(pickle_files)

    # Put all numbers here. For PyTorch we can use one scalar for each of the
    # mean and std, because we have one scalar here (for our depth images).
    numbers = []

    for p_idx,p_ff in enumerate(pickle_files):
        with open(p_ff, 'r') as fh:
            data = pickle.load(fh)
            print("Just loaded: {} (len: {})".format(p_ff, len(data)))

            for item in data:
                pname = path_train if p_idx != total_pickles-1 else path_valid
                if item['class'] == 0:
                    png_name = join(pname, 'success', 'd_{}.png'.format(str(t_success).zfill(5)))
                    t_success += 1
                elif item['class'] == 1:
                    png_name = join(pname, 'failure', 'd_{}.png'.format(str(t_failure).zfill(5)))
                    t_failure += 1
                else:
                    raise ValueError(item['class'])
                cv2.imwrite(png_name, item['d_img'])

                # Accumulate statistics for mean and std computation across our
                # lone channel. We actually 'triplcated' depth so take one axis.
                numbers.extend( item['d_img'][:,:,0].flatten() )

        print("so far: success {} vs failure {}".format(t_success, t_failure))

    print("done loading data, success {} vs failure {} (total {})".format(
            t_success, t_failure, t_success+t_failure))
    print("len(numbers):  {}  (has the single-channel mean/std info)".format(len(numbers)))
    print("mean(numbers): {}".format(np.mean(numbers)))
    print("std(numbers):  {}".format(np.std(numbers)))


def pytorch_data():
    """
    Straight from the tutorial ... but let's see what happens when I play around
    with different transformations.
    """
    mean = [96.8104350432]
    std  = [84.6227108358]

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    
    image_datasets = {x: datasets.ImageFolder(join(TARGET,x), data_transforms[x]) 
                        for x in ['train', 'valid']}
    dataloaders    = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                     batch_size=32,
                                                     shuffle=True,
                                                     num_workers=4)
                        for x in ['train', 'valid']}
    dataset_sizes  = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names    = image_datasets['train'].classes
    info = {'data_transforms' : data_transforms,
            'image_datasets' : image_datasets,
            'dataloaders' : dataloaders,
            'dataset_sizes' : dataset_sizes,
            'class_names' : class_names}
    return info


def train(info, model):
    data_transforms = info['data_transforms']
    image_datasets  = info['image_datasets']
    dataloaders     = info['dataloaders']
    dataset_sizes   = info['dataset_sizes']
    class_names     = info['class_names']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\nNow training!! On device: {}".format(device))
    print("class_names: {}".format(class_names))
    print("dataset_sizes: {}".format(dataset_sizes))



if __name__ == "__main__":
    # We only need to call this ONCE, then we can comment out since it creates
    # the data in the format we need for `ImageLoader`.
    #prepare_raw_data()

    # Prepare the ImageLoader.
    info = pytorch_data()

    # Train the ResNet.
    train(info, resnet18)
