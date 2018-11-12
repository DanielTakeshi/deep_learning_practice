# Benchmarking Grasp Network in PyTorch

Here's how to (roughly) reproduce the results I got from earlier with grasping.
This is detection so one needs to fix the target labels during data
augmentation. Use `custom_transforms.py` for my transformations.

Run `prepare_data.py` to prepare the data. Then run `grasp.py` to train. 

## Example with Bed-Making Data

Run: `python grasp.py --num_epochs 20` and get the following results. HUGE NOTE:
The "LossPix" is supposed to represent the losses on the raw input to the
network, but the way I have it structured, that's the (3,224,224)-sized input
image to the network. It is NOT the (480,640,3) original images, and my L2s from
the bed-making paper were benchmarked on the (480,640,3) images.

```
Now training!! On device: cuda:0
dataset_sizes: {'train': 1677, 'valid': 420}


Epoch 0/19
--------------------
(train)  Loss: 0.0784, LossPix: 78.5019
(valid)  Loss: 0.0723, LossPix: 90.5590

Epoch 1/19
--------------------
(train)  Loss: 0.0148, LossPix: 38.1078
(valid)  Loss: 0.0148, LossPix: 38.3170

Epoch 2/19
--------------------
(train)  Loss: 0.0097, LossPix: 30.7384
(valid)  Loss: 0.0078, LossPix: 27.7207

Epoch 3/19
--------------------
(train)  Loss: 0.0073, LossPix: 26.7446
(valid)  Loss: 0.0115, LossPix: 34.4069

Epoch 4/19
--------------------
(train)  Loss: 0.0063, LossPix: 24.6097
(valid)  Loss: 0.0081, LossPix: 28.9242

Epoch 5/19
--------------------
(train)  Loss: 0.0059, LossPix: 23.8597
(valid)  Loss: 0.0102, LossPix: 33.1818

Epoch 6/19
--------------------
(train)  Loss: 0.0056, LossPix: 23.3459
(valid)  Loss: 0.0079, LossPix: 28.3630

Epoch 7/19
--------------------
(train)  Loss: 0.0047, LossPix: 21.2511
(valid)  Loss: 0.0043, LossPix: 20.1657

Epoch 8/19
--------------------
(train)  Loss: 0.0048, LossPix: 21.4995
(valid)  Loss: 0.0065, LossPix: 25.5460

Epoch 9/19
--------------------
(train)  Loss: 0.0043, LossPix: 20.3216
(valid)  Loss: 0.0040, LossPix: 19.4170

Epoch 10/19
--------------------
(train)  Loss: 0.0039, LossPix: 19.6687
(valid)  Loss: 0.0078, LossPix: 27.9493

Epoch 11/19
--------------------
(train)  Loss: 0.0036, LossPix: 18.7637
(valid)  Loss: 0.0133, LossPix: 38.1339

Epoch 12/19
--------------------
(train)  Loss: 0.0035, LossPix: 18.4141
(valid)  Loss: 0.0043, LossPix: 20.4205

Epoch 13/19
--------------------
(train)  Loss: 0.0036, LossPix: 18.7190
(valid)  Loss: 0.0036, LossPix: 18.6473

Epoch 14/19
--------------------
(train)  Loss: 0.0040, LossPix: 19.8019
(valid)  Loss: 0.0055, LossPix: 23.6382

Epoch 15/19
--------------------
(train)  Loss: 0.0031, LossPix: 17.2271
(valid)  Loss: 0.0047, LossPix: 21.8161

Epoch 16/19
--------------------
(train)  Loss: 0.0028, LossPix: 16.2656
(valid)  Loss: 0.0032, LossPix: 17.4154

Epoch 17/19
--------------------
(train)  Loss: 0.0029, LossPix: 16.9455
(valid)  Loss: 0.0046, LossPix: 21.8354

Epoch 18/19
--------------------
(train)  Loss: 0.0027, LossPix: 16.4027
(valid)  Loss: 0.0030, LossPix: 16.5295

Epoch 19/19
--------------------
(train)  Loss: 0.0027, LossPix: 16.0330
(valid)  Loss: 0.0043, LossPix: 20.8582

Trained in 1m 56s
Best epoch losses: 0.003031  (pix: 16.5295)
train:  [0.0784, 0.0148, 0.0097, 0.0073, 0.0063, 0.0059, 0.0056, 0.0047, 0.0048,
0.0043, 0.0039, 0.0036, 0.0035, 0.0036, 0.004, 0.0031, 0.0028, 0.0029, 0.0027,
0.0027]
valid:  [0.0723, 0.0148, 0.0078, 0.0115, 0.0081, 0.0102, 0.0079, 0.0043, 0.0065,
0.004, 0.0078, 0.0133, 0.0043, 0.0036, 0.0055, 0.0047, 0.0032, 0.0046, 0.003,
0.0043]

Checking performance on one validation set minibatch:
```
