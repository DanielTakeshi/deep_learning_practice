# Tentative Results

(Will change this doc when needed)

I get this with default settings for commit
1d4340ade98d315c6c7655f8aaca90ea1a6c2466 running `python ryan_data.py`.

The validation loss is decreasing. But we will have to investigate this deeply.

```
Now training!! On device: cuda:0
dataset_sizes: {'train': 55, 'valid': 18}


Epoch 0/19
--------------------
(train)  Loss: 0.2729, LossPix: 0.0000
(valid)  Loss: 0.1839, LossPix: 0.0000

Epoch 1/19
--------------------
(train)  Loss: 0.1571, LossPix: 0.0000
(valid)  Loss: 0.1487, LossPix: 0.0000

Epoch 2/19
--------------------
(train)  Loss: 0.1033, LossPix: 0.0000
(valid)  Loss: 0.1746, LossPix: 0.0000

Epoch 3/19
--------------------
(train)  Loss: 0.1127, LossPix: 0.0000
(valid)  Loss: 0.1956, LossPix: 0.0000

Epoch 4/19
--------------------
(train)  Loss: 0.1048, LossPix: 0.0000
(valid)  Loss: 0.1792, LossPix: 0.0000

Epoch 5/19
--------------------
(train)  Loss: 0.0930, LossPix: 0.0000
(valid)  Loss: 0.1371, LossPix: 0.0000

Epoch 6/19
--------------------
(train)  Loss: 0.0655, LossPix: 0.0000
(valid)  Loss: 0.1060, LossPix: 0.0000

Epoch 7/19
--------------------
(train)  Loss: 0.0483, LossPix: 0.0000
(valid)  Loss: 0.0932, LossPix: 0.0000

Epoch 8/19
--------------------
(train)  Loss: 0.0454, LossPix: 0.0000
(valid)  Loss: 0.0890, LossPix: 0.0000

Epoch 9/19
--------------------
(train)  Loss: 0.0393, LossPix: 0.0000
(valid)  Loss: 0.0897, LossPix: 0.0000

Epoch 10/19
--------------------
(train)  Loss: 0.0369, LossPix: 0.0000
(valid)  Loss: 0.0867, LossPix: 0.0000

Epoch 11/19
--------------------
(train)  Loss: 0.0251, LossPix: 0.0000
(valid)  Loss: 0.0799, LossPix: 0.0000

Epoch 12/19
--------------------
(train)  Loss: 0.0272, LossPix: 0.0000
(valid)  Loss: 0.0722, LossPix: 0.0000

Epoch 13/19
--------------------
(train)  Loss: 0.0184, LossPix: 0.0000
(valid)  Loss: 0.0677, LossPix: 0.0000

Epoch 14/19
--------------------
(train)  Loss: 0.0195, LossPix: 0.0000
(valid)  Loss: 0.0652, LossPix: 0.0000

Epoch 15/19
--------------------
(train)  Loss: 0.0218, LossPix: 0.0000
(valid)  Loss: 0.0653, LossPix: 0.0000

Epoch 16/19
--------------------
(train)  Loss: 0.0153, LossPix: 0.0000
(valid)  Loss: 0.0667, LossPix: 0.0000

Epoch 17/19
--------------------
(train)  Loss: 0.0300, LossPix: 0.0000
(valid)  Loss: 0.0636, LossPix: 0.0000

Epoch 18/19
--------------------
(train)  Loss: 0.0151, LossPix: 0.0000
(valid)  Loss: 0.0622, LossPix: 0.0000

Epoch 19/19
--------------------
(train)  Loss: 0.0203, LossPix: 0.0000
(valid)  Loss: 0.0630, LossPix: 0.0000

Trained in 0m 25s
Best epoch losses: 0.062189  (pix: 0.0000)
train:  [0.2729, 0.1571, 0.1033, 0.1127, 0.1048, 0.093, 0.0655, 0.0483, 0.0454,
0.0393, 0.0369, 0.0251, 0.0272, 0.0184, 0.0195, 0.0218, 0.0153, 0.03, 0.0151,
0.0203]
valid:  [0.1839, 0.1487, 0.1746, 0.1956, 0.1792, 0.1371, 0.106, 0.0932, 0.089,
0.0897, 0.0867, 0.0799, 0.0722, 0.0677, 0.0652, 0.0653, 0.0667, 0.0636, 0.0622,
0.063]
```
