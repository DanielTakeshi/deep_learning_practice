import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
import custom_transforms as CT

import argparse, copy, cv2, os, sys, pickle, time
import numpy as np
from os.path import join

# ------------------------------------------------------------------------------
# Local data directories, from `prepare_data.py`.
TARGET1    = 'ssldata/'
TARGET2    = 'ssldata_pytorch/'
TRAIN_INFO = 'ssldata_pytorch/train/data_train_loader.pkl'
VALID_INFO = 'ssldata_pytorch/valid/data_valid_loader.pkl'

# For saving images+targets from minibatches, to inspect data augmentation.
TMPDIR1 = 'tmp_augm/'
if not os.path.exists(TMPDIR1):
    os.makedirs(TMPDIR1)

# For saving images+targets for model predictions, to inspect accuracy.
TMPDIR2 = 'tmp_model/'
if not os.path.exists(TMPDIR2):
    os.makedirs(TMPDIR2)

# See output of `prepare_data.py`.
MEAN = [0.41979732, 0.40260704, 0.4141044 ]
STD  = [0.43067302, 0.44038301, 0.44804261]

# Pre-trained models
resnet18 = models.resnet18(pretrained=True)
resnet34 = models.resnet34(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
# ------------------------------------------------------------------------------


def _save_images(inputs, labels, outputs, loss, phase):
    """Debugging the data transformations, labels, etc.

    OpenCV can't save if you use floats. You need: `img = img.astype(int)`.
    But, we also need the axes channel to be last, i.e. (height,width,channel).
    But PyTorch puts the channels earlier ... (channel,height,width).
    The raw depth images in the pickle files were of shape (480,640,3).

    Right now, the un-normalized images and predictions are for the RESIZED AND
    CROPPED images. Getting the 'true' un-normalized ones for the validation set
    can be done, but the training ones will require some knowledge of what we
    used for random cropping.
    """
    # Extract numpy tensor on the CPU.
    inputs = inputs.cpu().numpy()
    labels = labels.cpu().numpy()
    preds  = outputs.cpu().numpy()

    # Iterate through all (data-augmented) images in minibatch and save.
    for b in range(inputs.shape[0]):
        img  = inputs[b,:,:,:]
        targ = labels[b,:]
        pred = preds[b,:]

        # A good sanity check, all channels of _processed_ image have same sum.
        # Alsom, transpose to get 3-channel at the _end_, so shape (224,224,3).
        assert np.sum(img[0,:,:]) == np.sum(img[1,:,:]) == np.sum(img[2,:,:])
        assert img.shape == (3,224,224)
        img = img.transpose((1,2,0))

        # Undo the normalization, multiply by 255, then turn to integers.
        img = img*STD + MEAN
        img = img*255.0
        img = img.astype(int)

        # And similarly, for predictions.
        targ = targ*255.0
        pred = pred*255.0
        targ_int = int(targ[0]), int(targ[1])
        pred_int = int(pred[0]), int(pred[1])

        # Computing 'raw' L2, well for the (224,224) input image ...
        L2_pix = np.linalg.norm(targ - pred)
        # Later, I can do additional 'un-processing' to get truly original L2s.

        # Overlay prediction vs target.
        # Using `img` gets a weird OpenCV error, I had to add 'contiguous' here.
        img = np.ascontiguousarray(img, dtype=np.uint8)
        cv2.circle(img, center=targ_int, radius=2, color=(0,0,255), thickness=-1)
        cv2.circle(img, center=targ_int, radius=3, color=(0,0,0),   thickness=1)
        cv2.circle(img, center=pred_int, radius=2, color=(255,0,0), thickness=-1)
        cv2.circle(img, center=pred_int, radius=3, color=(0,255,0), thickness=1)

        # Inspect!
        fname = '{}/{}_{}_{:.0f}.png'.format(TMPDIR2, phase, str(b).zfill(4), L2_pix)
        cv2.imwrite(fname, img)


def train(model, args):
    # To debug transformation(s), pick any one to run, get images, and save.
    transforms_train = transforms.Compose([
        CT.Rescale((256,256)),
        CT.RandomCrop((224,224)),
        CT.RandomHorizontalFlip(),
        CT.ToTensor(),
        CT.Normalize(MEAN, STD),
    ])
    transforms_valid = transforms.Compose([
        CT.Rescale((256,256)),
        CT.CenterCrop((224,224)),
        CT.ToTensor(),
        CT.Normalize(MEAN, STD),
    ])

    gdata_t = CT.BedGraspDataset(infodir=TRAIN_INFO, transform=transforms_train)
    gdata_v = CT.BedGraspDataset(infodir=VALID_INFO, transform=transforms_valid)

    dataloaders = {
        'train': DataLoader(gdata_t, batch_size=32, shuffle=True, num_workers=8),
        'valid': DataLoader(gdata_v, batch_size=32, shuffle=False, num_workers=8),
    }
    dataset_sizes = {'train': len(gdata_t), 'valid': len(gdata_v)}

    # ADJUST CUDA DEVICE! Be careful about multi-GPU machines like the Tritons!!
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\nNow training!! On device: {}".format(device))
    print("dataset_sizes: {}\n".format(dataset_sizes))

    # Get things setup. Since ResNet has 1000 outputs, we need to adjust the
    # last layer to only give two outputs (since I'm doing classification).
    # And as usual, don't forget to add it to your correct device!!
    num_penultimate_layer = model.fc.in_features
    model.fc = nn.Linear(num_penultimate_layer, 2)
    model = model.to(device)

    # TODO now fix and include the dual architecture? Siamese?
    sys.exit()

    # Loss function & optimizer
    criterion = nn.MSELoss()
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    else:
        raise ValueError(args.optim)

    # --------------------------------------------------------------------------
    # FINALLY TRAINING!! Here, track loss and the 'original' loss in raw pixels.
    # --------------------------------------------------------------------------
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss     = np.float('inf')
    best_loss_pix = np.float('inf')
    all_train = []
    all_valid = []

    for epoch in range(args.num_epochs):
        print('\nEpoch {}/{}'.format(epoch, args.num_epochs-1))
        print('-' * 20)

        # Each epoch has a training and validation phase.
        # Epochs automatically tracked via the for loop over `dataloaders`.
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss     = 0.0
            running_loss_pix = 0.0

            # Iterate over data and labels (minibatches), by default, one epoch.
            for minibatch in dataloaders[phase]:
                inputs = (minibatch['image']).to(device)    # (B,3,224,224)
                labels = (minibatch['target']).to(device)   # (B,2)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward: track (gradient?) history _only_ if training
                # Confused, I need `labels.float()` even though `labels` should be a float!
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # TODO:  later double check this, I think this works if we want
                # the L2 for the (224,224) images that the network actually sees.
                # Need to also know cpu() and detach().

                targ = labels.cpu().numpy() * 255.0
                pred = outputs.detach().cpu().numpy() * 255.0
                delta = targ - pred  # shape (B,2)
                L2_pix = np.mean( np.linalg.norm(delta,axis=1) )

                running_loss += loss.item() * inputs.size(0)
                running_loss_pix += L2_pix * inputs.size(0)

            # We summed (not averaged) the losses earlier, so divide by full size.
            epoch_loss = running_loss / float(dataset_sizes[phase])
            epoch_loss_pix = running_loss_pix / float(dataset_sizes[phase])

            print('({})  Loss: {:.4f}, LossPix: {:.4f}'.format(
                    phase, epoch_loss, epoch_loss_pix))
            if phase == 'train':
                all_train.append(round(epoch_loss,4))
            else:
                all_valid.append(round(epoch_loss,4))

            # deep copy the model, use `state_dict()`.
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_loss_pix = epoch_loss_pix
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('\nTrained in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch losses: {:4f}  (pix: {:.4f})'.format(best_loss, best_loss_pix))
    print('train:  {}'.format(all_train))
    print('valid:  {}'.format(all_valid))

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Can make predictions on one minibatch just to confirm.
    print("\nChecking performance on one validation set minibatch:")
    model.eval()
    for minibatch in dataloaders['valid']:
        inputs = (minibatch['image']).to(device)
        labels = (minibatch['target']).to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
        _save_images(inputs, labels, outputs, loss, phase='valid')
        break

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
