""" From the 60 min blitz tutorial, need layers, then a forward function """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_printoptions(linewidth=180) # :-)
import sys


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution, etc
        self.conv1 = nn.Conv2d(1,6, kernel_size=5)
        self.conv2 = nn.Conv2d(6,16, kernel_size=5)
        # Now affine operations
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120,    out_features=84)
        self.fc3 = nn.Linear(in_features=84,     out_features=10)

    def forward(self, x):
        # Max pooling
        x = F.max_pool2d( F.relu(self.conv1(x)), kernel_size=(2,2) )
        x = F.max_pool2d( F.relu(self.conv2(x)), kernel_size=2 )
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # ignore batch size dimension
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# Create network, get nice overview
net = Net()
print(net)

# Learnable parameters
params = list(net.parameters())
print("\nlen(params): {}".format(len(params)))
print(params[0].size())  # conv1's weights
print("\nAll parameters:")
for p in params:
    print(p.size())

print("\nNow let's try network, using leading batch size of 1")
print("ALL inputs MUST have a batch size dim in first (0-th) axis")
print("and conv2d stuff takes in the channel count as second (1-th) axis")
input = torch.randn(1, 1, 32, 32)
out = net(input)
print("\nhere's the network's output tensor:\n{}".format(out))

# Now make gradients, from random input, don't forget to _zero_ out!
net.zero_grad()
out.backward(torch.randn(1, 10))

# Loss function, with dummy target
output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(output, target)
print("\nloss:  {}".format(loss))
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# investigate d(loss) / d(weight)
net.zero_grad()
print('\nconv1.bias.grad before backward (should be zero)')
print(net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# optimizer, MUST ZERO gradient buffers!! (this belongs in a training loop)
# Also, it appears to be 'best practice' to `zero_grad()` on the _optimizer_.
optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # Does the update
