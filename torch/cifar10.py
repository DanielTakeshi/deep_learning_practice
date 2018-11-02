""" From 60 min blitz tutorial. Now for PyTorch 0.4.1.

Btw, note that Conv2d has args (from ipython command line):

nn.Conv2d(in_channels, out_channels, kernel_size, 
          stride=1, padding=0, dilation=1, groups=1, bias=True)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
torch.set_printoptions(linewidth=180) # :-)
import sys


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Using a different pooling vs earlier part
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # If we set net to this device, we need data on the device as well
    # It will give a warning if `cuda:k` does not exist on the machine.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("our device: {}\n".format(device))

    # Loading data, looks like have `set` and `loader` for both train/test.
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    net = Net()
    net.to(device)
    print("\nour net:\n{}\n".format(net))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # loop over the dataset multiple times (use trainloader for convenience)
    for epoch in range(4):  
        running_loss = 0.0

        for i, data in enumerate(trainloader):
            # assuming we kept batch_size=4 obviously
            # inputs.size():  torch.Size([4, 3, 32, 32])
            # labels.size():  torch.Size([4])
            # outputs.size(): torch.Size([4, 10])
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero grads, then forward + backward + optimize
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')

    # Check on test data, note the `torch.no_grad()` ...
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('\nAccuracy of the network on the 10000 test images: %d %%\n' % (
        100 * correct / total))
    # analyze per class performance
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100*class_correct[i] / class_total[i]))
