""" From the 60 min blitz tutorial, need layers, then a forward function """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_printoptions(linewidth=180) # :-)


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution, etc
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        # Now affine operations
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

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
print("len(params): {}".format(len(params)))
print(params[0].size())  # conv1's weights
print("\nall param sizes:")
all_sizes = [p.size() for p in params]
for s in all_sizes:
    print(s)

print("\nNow let's try network, using leading batch size of 1")
print("ALL inputs MUST have a batch size dim in first (0-th) axis")
print("and conv2d stuff takes in the channel count as second (1-th) axis")
input = torch.randn(1, 1, 32, 32)
out = net(input)
print("{}\n".format(out))

# Now make gradients, from random input, don't forget to _zero_ out!
net.zero_grad()
out.backward(torch.randn(1, 10))

# Loss function, with dummy target
output = net(input)
target = torch.arange(1,11)
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# investigate d(loss) / d(weight)
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
print()

# optimizer, MUST ZERO gradient buffers!! (this belongs in a training loop)
optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # Does the update
