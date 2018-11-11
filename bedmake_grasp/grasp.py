"""Let's see if I can train using the bed-making data I have.
After running `prepare_raw_data()` only, I get:

    done loading data, success 559 vs failure 582 (total 1141)
    len(numbers):  350515200  (has the single-channel mean/std info)
    mean(numbers): 93.709165437
    std(numbers):  85.0125809655
    
    But, use this for actual mean/std because we want them in [0,256) ...
    mean(scaled): 0.366051427488
    std(scaled):  0.332080394397
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

# Target is where we re-format the data for PyTorch convenience methods.
# In the `cache` files, I already processed the depth images.
HEAD   = '/nfs/diskstation/seita/bed-make/cache_combo_v03_success'

# Move locally after data creation! With 10 epochs, I get an 8x speed-up: 4min -> 30sec.
#TARGET = '/nfs/diskstation/seita/bed-make/cache_combo_v03_success_pytorch'
TARGET = 'cache_combo_v03_success_pytorch'

TMPDIR = 'tmp/'

# From `prepare_raw_data`. Remember, we really have three channels.
MEAN = [0.36605, 0.36605, 0.36605]
STD  = [0.33208, 0.33208, 0.33208]

# Pre-trained models
resnet18 = models.resnet18(pretrained=True)
resnet34 = models.resnet34(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
# ------------------------------------------------------------------------------


def _save_images(inputs, labels, phase):
    """Debugging the data transformations, labels, etc.

    OpenCV can't save if you use floats. You need: `img = img.astype(int)`.
    But, we also need the axes channel to be last, i.e. (height,width,channel).
    But PyTorch puts the channels earlier ... (channel,height,width).
    The raw depth images in the pickle files were of shape (480,640,3).

    Given `img` from a PyTorch Tensor, it will be of shape (3,224,224) with
    ResNet pre-processing. For this, DON'T use `img = img.swapaxes(0,2)` as that
    will not correctly change the axes; it rotates the image. The docs say:

    https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToPILImage
        Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
        H x W x C to a PIL Image while preserving the value range.

    Thus, we need `img = img.transpose((1,2,0))`. Then channel goes at the end
    BUT the height and width are also both 'translated' over to the first (i.e.,
    0-th index) and second channels, respectively.
    """
    assert not os.path.exists(TMPDIR), "directory exists:\n\t{}".format(TMPDIR)
    os.makedirs(TMPDIR)

    # Extract numpy tensor on the CPU.
    inputs = inputs.cpu().numpy()
    B = inputs.shape[0]

    # Iterate through all (data-augmented) images in minibatch and save.
    for b in range(B):
        img = inputs[b,:,:,:]

        # A good sanity check, all channels of _processed_ image have same sum.
        # Alsom, transpose to get 3-channel at the _end_, so shape (224,224,3).
        assert np.sum(img[0,:,:]) == np.sum(img[1,:,:]) == np.sum(img[2,:,:])
        assert img.shape == (3,224,224)
        img = img.transpose((1,2,0))

        # Undo the normalization, multiply by 255, then turn to integers.
        img = img*STD + MEAN
        img = img*255.0
        img = img.astype(int)

        # 0 means it is 'success' (i.e., blanket covered), 1 is failure.
        label = labels[b]
        if int(label) == 0:
            label = 'success'
        else:
            label = 'failure'

        fname = '{}/{}_{}_{}.png'.format(TMPDIR, phase, str(b).zfill(4), label)
        cv2.imwrite(fname, img)


def train(model, args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]),
    }
    
    # So, ImageFolder (within the `train/` and `valid/`) requires images to be
    # stored within sub-directories based on their labels. Also, I wonder, maybe
    # better to drop the last batch for the DataLoader?
    image_datasets = {x: datasets.ImageFolder(join(TARGET,x), data_transforms[x]) 
                        for x in ['train', 'valid']}
    dataloaders    = {x: torch.utils.data.DataLoader(image_datasets[x],
                        batch_size=32, shuffle=True, num_workers=4)
                          for x in ['train', 'valid']}
    dataset_sizes  = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names    = image_datasets['train'].classes

    # ADJUST CUDA DEVICE! Be careful about multi-GPU machines like the Tritons!!
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\nNow training!! On device: {}".format(device))
    print("class_names: {}".format(class_names))
    print("dataset_sizes: {}\n".format(dataset_sizes))

    # Get things setup. Since ResNet has 1000 outputs, we need to adjust the
    # last layer to only give two outputs (since I'm doing classification).
    # And as usual, don't forget to add it to your correct device!!
    num_penultimate_layer = model.fc.in_features
    model.fc = nn.Linear(num_penultimate_layer, 2)
    model = model.to(device)

    # Loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # not doing now, but could decay LR by factor of 0.1 every 7 epochs
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    else:
        raise ValueError(args.optim)

    # --------------------------------------------------------------------------
    # FINALLY TRAINING!!
    # --------------------------------------------------------------------------
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    all_train = []
    all_valid = []

    for epoch in range(args.num_epochs):
        print('\nEpoch {}/{}'.format(epoch, args.num_epochs-1))
        print('-' * 20)

        # Each epoch has a training and validation phase
        # Epochs automatically tracked via the for loop over `dataloaders`.
        for phase in ['train', 'valid']:
            if phase == 'train':
                #scheduler.step() # not doing this for now
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data and labels (minibatches), by default, for one
            # epoch. Data augmentation happens here on the fly. :-)
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                #_save_images(inputs, labels, phase)
                #sys.exit()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward: track (gradient?) history _only_ if training
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)             # forward pass
                    _, preds = torch.max(outputs, 1)    # returns (max vals, indices)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # We summed (not averaged) the losses earlier, so divide by full size.
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('({})  Loss: {:.4f}, Acc: {:.4f} (num: {})'.format(
                    phase, epoch_loss, epoch_acc, running_corrects))
            if phase == 'train':
                all_train.append(round(epoch_acc.item(),3))
            else:
                all_valid.append(round(epoch_acc.item(),3))

            # deep copy the model, use `state_dict()`.
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('\nTrained in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('train:  {}'.format(all_train))
    print('valid:  {}'.format(all_valid))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    # We only need to call this ONCE, then we can comment out since it creates
    # the data in the format we need for `ImageLoader`.
    #prepare_raw_data()

    pp = argparse.ArgumentParser()
    pp.add_argument('--optim', type=str)
    pp.add_argument('--model', type=str)
    pp.add_argument('--num_epochs', type=int, default=20)
    args = pp.parse_args() 

    # Train the ResNet. Then I can do stuff with it ...  I get similar best
    # validation set performance with ResNet-{18,34,50}, fyi.
    if args.model == 'resnet18':
        model = train(resnet18, args)
    elif args.model == 'resnet34':
        model = train(resnet34, args)
    elif args.model == 'resnet50':
        model = train(resnet50, args)
    else:
        raise ValueError(args.model)
