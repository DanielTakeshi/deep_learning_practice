"""See README for results."""
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import argparse, copy, cv2, os, sys, pickle, time
import numpy as np
from os.path import join

# ------------------------------------------------------------------------------
# Local data directory, from `prepare_data.py`.
TARGET = 'cache_combo_v03_success_pytorch'

# For the custom dataset we use.
DATA_TRAIN_INFO = 'cache_combo_v03_pytorch/train/data_train_loader.pkl'
DATA_VALID_INFO = 'cache_combo_v03_pytorch/valid/data_valid_loader.pkl'

# For saving images+targets from minibatches, to inspect data augmentation.
TMPDIR = 'tmp/'

# See `prepare_data.py`. Remember, we really have three channels.
MEAN = [0.37468, 0.37468, 0.37468]
STD  = [0.33259, 0.33259, 0.33259]

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


class GraspDataset(Dataset):
    """Custom Grasp dataset, inspired by Face Landmarks dataset."""

    def __init__(self, infodir, transform=None):
        self.infodir = infodir
        with open(self.infodir, 'r') as fh:
            self.data = pickle.load(fh)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """As in the face landmarks, samples are dicts with images and labels."""
        png_path, target = self.data[idx]
        image = cv2.imread(png_path)
        target = ( float(target[0]), float(target[1]) )
        sample = {'image': image, 'target': target}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'target': torch.from_numpy(target)}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        # Daniel: wait, where is `transform` coming from?
        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]
        return {'image': img, 'landmarks': landmarks}


class HorizontalFlip(object):
    """AKA, a flip _about_ the *VERICAL* axis."""

    def __init__(self, flipping_ratio=0.5):
        self.flipping_ratio = 0.5

    def __call__(self, sample):
        """
        If we want a 'vertical' flip (the 'intuitive' meaning) then second arg
        in `cv2.flip()` is 0, not 1.  If we flip, need to adjust label, using
        CURRENT image size, since it might have been resized earlier.
        """
        image, target = sample['image'], sample['target']
        targetx, targety = target
        if np.random.rand() < self.flipping_ratio:
            h, w, c = image.shape
            image = cv2.flip(image, 1)
            targetx = w - target[0]
        target = (targetx, targety)
        return {'image': image, 'target': target}


def _save_viz(sample, idx):
    img, target = sample['image'], sample['target']
    pose_int = int(target[0]),int(target[1])
    cv2.circle(img, center=pose_int, radius=2, color=(0,0,255), thickness=-1)
    cv2.circle(img, center=pose_int, radius=3, color=(0,0,0), thickness=1)
    fname = join(TMPDIR, 'example_{}.png'.format(str(idx).zfill(4)))
    cv2.imwrite(fname, img)


def train(model, args):
    transforms_train = transforms.Compose([
        #Rescale(224),
        HorizontalFlip(),
        #ToTensor()
    ])
    transforms_valid = transforms.Compose([
        Rescale(224),
        ToTensor()
    ])
    gdata_t = GraspDataset(infodir=DATA_TRAIN_INFO, transform=transforms_train)
    gdata_v = GraspDataset(infodir=DATA_VALID_INFO, transform=transforms_valid)

    for i in range(10):
        _save_viz(gdata_t[i], idx=i)

    sys.exit()
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
    pp = argparse.ArgumentParser()
    pp.add_argument('--model', type=str, default='resnet18')
    pp.add_argument('--optim', type=str, default='adam')
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
