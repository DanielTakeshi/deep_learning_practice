"""Testing ResNet pre-trained models.

The goal is to see how well each of these does on the ImageNet validation data.
That's it. So, no training. Just load the pre-trained models and see results.
"""
import torch
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)
resnet34 = models.resnet34(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("On device: {}\n".format(device))


def debug_parameters(model):
    """Access names via `model.named_parameters()`.

    (ResNet 18) total parameters: 11689512
    (ResNet 34) total parameters: 21797672
    (ResNet 50) total parameters: 25557032

    "Robot Learning in Homes": used ResNet 18
    "Zero Shot Visual Imitation": used ResNet 50
    """
    print("")
    total = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name, p.numel())
            total += p.numel()
    print("\ntotal parameters: {}".format(total))


def investigate(name, model):
    print("\n\n{}:".format(name))
    print(model)
    debug_parameters(model)


def validation():
    """Load ImageNet validation set and evaluate.

    https://github.com/pytorch/examples/tree/master/imagenet

    Note: for loading data, you need to deal with `transforms.Normalize()`, but
    fortunately, for ImageNet and other datasets, they have already pre-computed
    per-channel means and standard deviations, so we don't need to compute
    those. This lets us avoid rescaling the input image. (But, should we still
    do this for transfer learning?)
    """
    pass


if __name__ == "__main__":
    # For checking parameters, etc. Even ResNet50 has "only" 25M params. :-)
    investigate('ResNet18', resnet18)
    investigate('ResNet34', resnet34)
    investigate('ResNet50', resnet50)

    # Check performance on ImageNet validation set.
    validation()
