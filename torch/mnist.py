""" MNIST classification with PyTorch, version 0.4.0.

From: https://github.com/pytorch/examples/blob/master/mnist/main.py
With additional comments by myself.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys


class Net(nn.Module):
    """
    Use `nn` to build the neural network. Also use `nn.functional`. These share
    some computations so it might be up to me how to use it, but I'll just
    borrow from existing code conventions, e.g., using `F.relu` instead of
    `nn.relu`. Looks like it's easier to use the functional stuff for stuff
    without trainable parameters, such as ReLUs.
    """

    def __init__(self):
        """ Looks like it's standard to have separate network construction and
        forward pass methods.

        Actually, it's required if using the `nn.Module` class. See the docs:
            http://pytorch.org/docs/master/nn.html#torch.nn.Module
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """ Forward pass to produce the output, specifies the _connections_.

        The docs say that this method *must* be overriden by all sub-classes.
        Also, `x` must be of type `Variable`. So is the output. And BTW we
        always assume a minibatch of samples. In TF we had some flexibility (I
        think) but it's better to always leave an extra dimension for MBs.
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320) # Flatten but w/o allocating new memory.
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train(epoch, train_loader, model, optimizer):
    """ Training.

    Set `model.train()`, even though it's not necessary to get good performance
    on this particular MNIST. It seems to be "good PyTorch practice" because it
    makes our intent to train clearer. By default models are set to training
    mode. The mode matters for modules such as BatchNorm and Dropout which
    behave differently during them.

    We repeatedly call from the train_loader which provides us batches. However,
    those are not on the CPU, hence why we keep transferring. It's also
    interesting to note how we repeatedly compute the loss here, rather than
    it being computed ahead of time in TensorFlow.

    Regarding types, before:
      data:   [torch.FloatTensor of size 64x1x28x28]
      target: [torch.LongTensor of size 64]
    after:
      data:   [torch.cuda.FloatTensor of size 64x1x28x28 (GPU 0)]
      target: [torch.cuda.LongTensor of size 64 (GPU 0)]

    In general, I will always be calling `x.cuda()`.  Wrap them with `Variable`
    so that we can compute gradients.  As usual, set `optimizer.zero_grad()`
    because otherwise the gradients will be ACCUMULATED with the previous ones!

    We call `backward()` on a SCALAR variable, hence no further arguments,
    meaning that "o" is just some output where we can write `do/dx_1`,
    `do/dx_2`, etc., for all inputs x_i which produce "o".

    Again, as usual, take `optimizer.step()` to update the parameters. To get
    the actual gradients, we could do `x.grad` where here `x` is the model
    parameters, though I believe we keep that encapsulated in the `model`?

    For loss functions, use the ones provided, e.g., here we use the negative
    log likelihood class: http://pytorch.org/docs/master/nn.html#loss-functions.
    Though we are using the functional ones here, with slightly different names.
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        # Get used to this; it's a very common patten with PyTorch code.
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test(test_loader, model):
    """ Similar to training. A few things:

    We use `model.eval()` to make it clearer that we are in "evaluation mode."
    It matters for modules such as BatchNorm and Dropout which behave
    differently during those modes.

    We use `F.nll_loss(...)` in a slightly different way, without taking an
    average but just summing it. We save averaging for the end.

    And note the use of `[...].data` to extract the contents of a Variable.
    """
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Hmm ... must offer some parallelism benefit to loaders?
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Load the MNIST data if necessary. There's also CIFAR, etc.
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data',
                       train=True,
                       download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data',
                       train=False,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Form network and optimizer. And yes we need to keep doing `model.cuda()`.
    # Looks like in PyTorch, you form the model and then pass in the parameters.
    # In TensorFlow, we had `trainable_weights` which covers the same ground.
    model = Net()
    if args.cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Finally, training and testing.
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loader, model, optimizer)
    test(test_loader, model)
