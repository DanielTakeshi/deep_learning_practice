import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
from torchvision import datasets, transforms
import copy, cv2, os, sys, pickle, time
import numpy as np
from os.path import join

# ------------------------------------------------------------------------------
TARGET = 'tmp/'
TEST_IMG = 'debug_img/'

# When dealing with mean and std, need to use a three-item list.
# Also, since we call ToTensor(), we want mean/std for *scaled* data.
#MEAN = [93.8304761096, 93.8304761096, 93.8304761096] # wrong
#STD  = [84.9985507432, 84.9985507432, 84.9985507432] # wrong
#MEAN = [93.8304761096] # wrong
#STD  = [84.9985507432] # wrong
MEAN = [0.36796265141, 0.36796265141, 0.36796265141]
STD  = [0.33332764997, 0.33332764997, 0.33332764997]
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
    assert not os.path.exists(TEST_IMG), "directory exists:\n\t{}".format(TEST_IMG)
    os.makedirs(TEST_IMG)
    import torchvision.transforms.functional as F
    B = inputs.shape[0]

    # Extract numpy tensor.
    inputs = inputs.cpu().numpy()

    # Iterate through all (data-augmented) images in minibatch and save.
    for b in range(B):
        img = inputs[b,:,:,:]

        # A good sanity check, all channels of _processed_ image have same sum.
        assert np.sum(img[0,:,:]) == np.sum(img[1,:,:]) == np.sum(img[2,:,:])

        # Transpose to get 3-channel at the _end_, so shape (224,224,3).
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
        fname = 'debug_img/{}_{}_{}.png'.format(phase, str(b).zfill(4), label)

        # Save using standard CV2 stuff.
        cv2.imwrite(fname, img)


def train(model):
    """From this tutorial with minor edits:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    BTW: you need the `ToTensor()` BEFORE `Normalize()`. And because of this, we
    want the MEAN and STD to reflect the *scaled* images, NOT the raw ones.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
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
    image_datasets = {x: datasets.ImageFolder(join(TARGET,x), data_transforms[x]) 
                        for x in ['train', 'valid']}
    dataloaders    = {x: torch.utils.data.DataLoader(image_datasets[x],
                        batch_size=32, shuffle=True, num_workers=4)
                          for x in ['train', 'valid']}
    dataset_sizes  = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names    = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\nNow training!! On device: {}".format(device))
    print("class_names: {}".format(class_names))
    print("dataset_sizes: {}\n".format(dataset_sizes))

    # Since ResNet has 1000 outputs, we need to adjust the last layer for two outputs.
    num_penultimate_layer = model.fc.in_features
    model.fc = nn.Linear(num_penultimate_layer, 2)
    model = model.to(device)

    # Loss function & optimizer; decay LR by factor of 0.1 every 7 epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --------------------------------------------------------------------------
    # FINALLY TRAINING!!
    # --------------------------------------------------------------------------
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    num_epochs = 50

    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 20)

        # Each epoch has a training and validation phase
        # Epochs automatically tracked via the for loop over `dataloaders`.
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data and labels (minibatches), by default, for one epoch.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Important!! I want to see the augmented data!
                _save_images(inputs, labels, phase)
                sys.exit()

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
            print('({})  Loss: {:.4f}, Acc: {:.4f} (num right: {})'.format(
                    phase, epoch_loss, epoch_acc, running_corrects))
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc

    time_elapsed = time.time() - since
    print('\nTrained in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


if __name__ == "__main__":
    resnet18 = models.resnet18(pretrained=True)
    train(resnet18)
