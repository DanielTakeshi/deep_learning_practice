"""
My custom transforms that I use, mainly for detection and non-classification
stuff. (If doing classification just borrow the ones from torchvision.)
Also the dataset class.
"""
import cv2, os, sys
import numpy as np
import torch
from torchvision.transforms import functional as F


class GraspDataset(Dataset):
    """Custom dataset, inspired by Face Landmarks dataset."""

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


#TODO BELOW

class Normalize(object):
    """Normalize.
    
    https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py#L129
    https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py#L157

    Actually we can just call the functional ...
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        assert image.shape[0] == 3, image.shape
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'target': target}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors.
    
    Like in normal PyTorch's ToTensor() we scale pixel values to be in [0,1].
    BUT we should also scale the labels to better condition the optimization.

    https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py#L70
    https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py#L38

    The first link, the transform, calls the second one, the functional. There
    are cases for ndarrays and PIL images, but we should only deal w/the former.
    """
    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        assert isinstance(image, np.ndarray), image
        assert isinstance(target, tuple), target

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        # Convert to numpy, but we need to then divide by 255.
        image  = torch.from_numpy( image )
        target = torch.from_numpy( np.array(target) )
        assert isinstance(image, torch.ByteTensor), image
        image  = image.float().div(255)
        target = target.div(255)         # these are already floats
        return {'image': image, 'target': target}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same. NOTE: I am likely to
            use this only for making square images, but it might be useful to
            keep the aspect ratio for later... or we can rescale and THEN do the
            random crop after that to make it square.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        # In cv2, image.shape represents (height, width, channels).
        h, w, channels = image.shape
        h, w = float(h), float(w)
        assert channels == 3, channels

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        # Daniel: tutorial said h,w but cv2.resize uses w,h ... I tested it.
        # Despite order of w,h here, for `img.shape` it's h,w,(channels). Confusing.
        img = cv2.resize(image, (new_w, new_h))

        target = ( target[0] * (new_w / w), target[1] * (new_h / h) )
        return {'image': img, 'target': target}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        h, w, channels = image.shape
        assert channels == 3, channels
        h, w = float(h), float(w)
        new_h, new_w = self.output_size

        # Intuition: we want to "remove" `w-new_w` pixels. So, starting from the
        # left, randomly pick starting point, and only go `new_w` to the right.
        top  = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        target_x = target[0] - left
        target_y = target[1] - top

        # IMPORTANT! In detection, our targets could have been cropped out. To
        # avoid this, only crop a little bit. That way we can 'approximate' it
        # by thresholding the value to be within the image.
        target_x = min( max(target_x, 0.0), new_w )
        target_y = min( max(target_y, 0.0), new_h )

        return {'image': image, 'target': (target_x, target_y)}


class CenterCrop(object):
    """Crop the image, in a *centered* manner!! Similar to RandomCrop except the
    `top` and `left` parts are determined analytically.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        h, w, channels = image.shape
        assert channels == 3, channels
        h, w = float(h), float(w)
        new_h, new_w = self.output_size

        # Intuition: we want to "remove" `w-new_w` pixels. So, starting from the
        # left, randomly pick starting point, and only go `new_w` to the right.
        # Since this is CENTERED we want to take the midpoint of these.
        top  = int((h - new_h) / 2.0)
        left = int((w - new_w) / 2.0)

        image = image[top: top + new_h,
                      left: left + new_w]
        target_x = target[0] - left
        target_y = target[1] - top

        # Similar considerations as in `RandomCrop`.
        target_x = min( max(target_x, 0.0), new_w )
        target_y = min( max(target_y, 0.0), new_h )

        return {'image': image, 'target': (target_x, target_y)}


class RandomHorizontalFlip(object):
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

