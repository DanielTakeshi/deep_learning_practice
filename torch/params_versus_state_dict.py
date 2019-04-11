""" A bit puzzling why they are different.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# All of these return False for requires_grad
for k,v in net.state_dict().items():
    print(v.requires_grad, v.size())
print()

# All of these return True for requires_grad
for p in net.parameters():
    print(p.requires_grad, p.size())
print()

# All of these return True for requires_grad --- AH!
# https://discuss.pytorch.org/t/grad-is-none-in-net-state-dict-but-not-in-net-named-parameters-why/20794
for k,v in net.state_dict(keep_vars=True).items():
    print(v.requires_grad, v.size())
print()

