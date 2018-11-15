import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
import custom_transforms as CT
from net import PolicyNet

import argparse, copy, cv2, os, sys, pickle, time
import numpy as np
from os.path import join
from collections import defaultdict

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


def _save_images(imgs_t, imgs_tp1, labels_pos, labels_ang, out_pos, 
                 out_ang, ang_predict, loss, phase='valid'):
     """Debugging the data transforms, labels, net predictions, etc.
 
     OpenCV can't save if you use floats. You need: `img = img.astype(int)`.
     But, we also need the axes channel to be last, i.e. (height,width,channel).
     But PyTorch puts the channels earlier ... (channel,height,width).
     The raw depth images in the pickle files were of shape (480,640,3).
 
     Right now, the un-normalized images and predictions are for the RESIZED AND
     CROPPED images. Getting the 'true' un-normalized ones for the validation set
     can be done, but the training ones will require some knowledge of what we
     used for random cropping.
     """
     B = imgs_t.shape[0]
     imgs_t   = imgs_t.cpu().numpy()
     imgs_tp1 = imgs_tp1.cpu().numpy()

     # Iterate through all (data-augmented) images in minibatch and save.
     for b in range(B):
         img_t    = imgs_t[b,:,:,:]
         img_tp1  = imgs_tp1[b,:,:,:]
         targ_pos = labels_pos[b,:]
         targ_ang = labels_ang[b]
         pred_pos = out_pos[b,:]
         pred_ang = out_ang[b,:]

         assert img_t.shape == img_tp1.shape == (3,224,224)
         img_t   = img_t.transpose((1,2,0))
         img_tp1 = img_tp1.transpose((1,2,0))
         h,w,c = img_t.shape
 
         # Undo the normalization, multiply by 255, then turn to integers.
         img_t   = img_t*STD + MEAN
         img_t   = img_t*255.0
         img_t   = img_t.astype(int)
         img_tp1 = img_tp1*STD + MEAN
         img_tp1 = img_tp1*255.0
         img_tp1 = img_tp1.astype(int)

         # And similarly, for predictions.
         targ_pos_int = int(targ_pos[0]*w), int(targ_pos[1]*h)
         pred_pos_int = int(pred_pos[0]*w), int(pred_pos[1]*h)
 
         # Computing 'raw' L2, well for the (224,224) input image ...
         #L2_pix = np.linalg.norm(targ_pos_ing - pred_pos_int)
         L2_pix = 0.0 # will do later
         # Later, I can do additional 'un-processing' to get truly original L2s.
 
         # Overlay prediction vs target.
         # Using `img` gets a weird OpenCV error, I had to add 'contiguous' here.
         img = np.ascontiguousarray(img_t, dtype=np.uint8)
         cv2.circle(img, center=targ_pos_int, radius=2, color=(0,0,255), thickness=-1)
         cv2.circle(img, center=targ_pos_int, radius=3, color=(0,0,0),   thickness=1)
         cv2.circle(img, center=pred_pos_int, radius=2, color=(255,0,0), thickness=-1)
         cv2.circle(img, center=pred_pos_int, radius=3, color=(0,255,0), thickness=1)
 
         # Inspect!
         fname = '{}/{}_{}_{:.0f}.png'.format(TMPDIR2, phase, str(b).zfill(4), L2_pix)
         cv2.imwrite(fname, img)

    # TODO: use the below code and merge it with the stuff above.
    ## targ = (int(target_xy[0]), int(target_xy[1]))
    ## cv2.circle(img_t, center=targ, radius=2, color=RED, thickness=-1)
    ## cv2.circle(img_t, center=targ, radius=3, color=BLACK, thickness=1)

    ## # This is the direction. Don't worry about the length, we can't easily get
    ## # it in pixel space and we keep length in world-space roughly fixed anyway.
    ## if target_ang[0] == 1:
    ##     offset = [50, 0]
    ## elif target_ang[1] == 1:
    ##     offset = [0, -50]
    ## elif target_ang[2] == 1:
    ##     offset = [-50, 0]
    ## elif target_ang[3] == 1:
    ##     offset = [0, 50]
    ## else:
    ##     raise ValueError(target_ang)

    ## goal = (targ[0] + offset[0], targ[1] + offset[1])
    ## int(target_xy[0]),int(target_xy[1])
    ## cv2.arrowedLine(img_t, targ, goal, color=BLUE, thickness=2)
    ## cv2.putText(img=img_t, 
    ##             text="raw ang: {}".format(raw_ang),
    ##             org=(50,50),
    ##             fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
    ##             fontScale=0.75, 
    ##             color=GREEN,
    ##             thickness=2)

    ## # Combine images (t,tp1) together.
    ## hstack = np.concatenate((img_t, img_tp1), axis=1)
    ## fname = join(TMPDIR1, 'example_{}.png'.format(str(idx).zfill(4)))
    ## cv2.imwrite(fname, hstack)


def _log(phase, ep_loss, ep_loss_pos, ep_loss_ang, ep_correct_ang):
    """For logging."""
    print("  ({})".format(phase))
    print("loss total:  {:.4f}".format(ep_loss))
    print("loss_pos:    {:.4f}".format(ep_loss_pos))
    print("loss_ang:    {:.4f}".format(ep_loss_ang))
    print("correct_ang: {:.4f}".format(ep_correct_ang))


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
    data_sizes = {'train': float(len(gdata_t)), 'valid': float(len(gdata_v))}

    # ADJUST CUDA DEVICE! Be careful about multi-GPU machines like the Tritons!!
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\nNow training!! On device: {}".format(device))
    print("data_sizes: {}\n".format(data_sizes))

    # Build policy w/pre-trained stem. Can print it to debug.
    policy = PolicyNet(model, args)
    policy = policy.to(device)

    # Optimizer and loss functions.
    if args.optim == 'sgd':
        optimizer = optim.SGD(policy.parameters(), lr=0.01, momentum=0.9)
    elif args.optim == 'adam':
        optimizer = optim.Adam(policy.parameters(), lr=0.0001)
    else:
        raise ValueError(args.optim)
    criterion_mse  = nn.MSELoss()
    criterion_cent = nn.CrossEntropyLoss()

    # --------------------------------------------------------------------------
    # FINALLY TRAINING!! Here, track loss and the 'original' loss in raw pixels.
    # --------------------------------------------------------------------------
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.float('inf')
    all_train = defaultdict(list)
    all_valid = defaultdict(list)
    lambda1 = 1.0
    lambda2 = 1.0

    for epoch in range(args.num_epochs):
        print('')
        print('-' * 30)
        print('Epoch {}/{}'.format(epoch, args.num_epochs-1))
        print('-' * 30)

        # Each epoch has a training and validation phase.
        # Epochs automatically tracked via the for loop over `dataloaders`.
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # Track statistics over _this_ coming epoch (only).
            running = defaultdict(list)

            # Iterate over data and labels (minibatches), by default, one epoch.
            for mb in dataloaders[phase]:
                imgs_t     = (mb['img_t']).to(device)           # (B,3,224,224)
                imgs_tp1   = (mb['img_tp1']).to(device)         # (B,3,224,224)
                labels     = (mb['label']).to(device)           # (B,3)
                labels_pos = labels[:,:2].float()               # (B,2)
                labels_ang = torch.squeeze(labels[:,2:].long()) # (B,1)

                # Zero the parameter gradients!
                optimizer.zero_grad()

                # Forward: track gradient history _only_ if training
                with torch.set_grad_enabled(phase == 'train'):
                    out_pos, out_ang = policy(imgs_t, imgs_tp1)

                    # Get classification accuracy from the predicted angle probs
                    _, ang_predict = torch.max(out_ang, dim=1)
                    correct_ang = (ang_predict == labels_ang).sum().item()

                    if args.model_type == 1:
                        # First loss needs (B,2). Second (B,) for class _index_.
                        loss_pos = criterion_mse(out_pos, labels_pos)
                        loss_ang = criterion_cent(out_ang, labels_ang)
                        loss = (lambda1 * loss_pos) + (lambda2 * loss_ang)
                    elif args.model_type == 2:
                        raise NotImplementedError()
                    elif args.model_type == 3:
                        raise NotImplementedError()
                    else:
                        raise ValueError()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Keep track of stats, mult by batch size since we average earlier
                running['loss'].append(loss.item() * imgs_t.size(0))
                running['loss_pos'].append(loss_pos.item() * imgs_t.size(0))
                running['loss_ang'].append(loss_ang.item() * imgs_t.size(0))
                running['correct_ang'].append(correct_ang)

            # We summed (not averaged) the losses earlier, so divide by full size.
            ep_loss        = np.sum(running['loss']) / data_sizes[phase]
            ep_loss_pos    = np.sum(running['loss_pos']) / data_sizes[phase]
            ep_loss_ang    = np.sum(running['loss_ang']) / data_sizes[phase]
            ep_correct_ang = np.sum(running['correct_ang']) / data_sizes[phase]
            _log(phase, ep_loss, ep_loss_pos, ep_loss_ang, ep_correct_ang)

            if phase == 'train':
                all_train['loss'].append(round(ep_loss,5))
                all_train['loss_pos'].append(round(ep_loss_pos,5))
                all_train['loss_ang'].append(round(ep_loss_ang,5))
            else:
                # Can print outputs and labels here for the last minibatch
                # evaluated from the validation set during this epoch.
                #print(out_pos, out_ang, labels)
                all_valid['loss'].append(round(ep_loss,5))
                all_valid['loss_pos'].append(round(ep_loss_pos,5))
                all_valid['loss_ang'].append(round(ep_loss_ang,5))

            # deep copy the model, use `state_dict()`.
            if phase == 'valid' and ep_loss < best_loss:
                best_loss = ep_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        print('-' * 30)

    time_elapsed = time.time() - since
    print('\nTrained in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation epoch total loss:  {:4f}'.format(best_loss))
    print('  train:\n{}'.format(all_train['loss']))
    print('  valid:\n{}'.format(all_valid['loss']))

    # Load best model weights, make predictions on validatoin to confirm
    model.load_state_dict(best_model_wts)
    model.eval()
    print("\nVisualizing performance of best model on validation set:")

    for minibatch in dataloaders['valid']:
        imgs_t     = (mb['img_t']).to(device)
        imgs_tp1   = (mb['img_tp1']).to(device)
        labels     = (mb['label']).to(device)
        labels_pos = labels[:,:2].float()
        labels_ang = torch.squeeze(labels[:,2:].long())

        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            out_pos, out_ang = policy(imgs_t, imgs_tp1)
            _, ang_predict = torch.max(out_ang, dim=1)
            correct_ang = (ang_predict == labels_ang).sum().item()

            loss_pos = criterion_mse(out_pos, labels_pos)
            loss_ang = criterion_cent(out_ang, labels_ang)
            loss = (lambda1 * loss_pos) + (lambda2 * loss_ang)
            print("  {} / {} angle accuracy".format(correct_ang, imgs_t.size(0)))

            _save_images(imgs_t, imgs_tp1, labels_pos, labels_ang, out_pos, 
                         out_ang, ang_predict, loss, phase='valid')

    return model, all_train, all_valid


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument('--model', type=str, default='resnet18')
    pp.add_argument('--optim', type=str, default='adam')
    pp.add_argument('--num_epochs', type=int, default=30)
    # Rely on several options for the loss type. See README for details.
    pp.add_argument('--model_type', type=int, default=1)
    args = pp.parse_args() 

    # Train the ResNet. Then I can do stuff with it ...  I get similar best
    # validation set performance with ResNet-{18,34,50}, fyi.
    if args.model == 'resnet18':
        model, stats_train, stats_valid = train(resnet18, args)
    elif args.model == 'resnet34':
        model, stats_train, stats_valid = train(resnet34, args)
    elif args.model == 'resnet50':
        model, stats_trai, stats_valid = train(resnet50, args)
    else:
        raise ValueError(args.model)

    # Save model in appropriate directory for deployment later.
    # TODO e.g., pickle.dump(...)
    # Also save the stats_train and stats_valid for plotting.

