# Benchmarking BedMake Results

(Trying to reproduce results from my paper.)

Actually these are already on par with what I am getting. Tricky to compare
since I think we had to adjust the data with Honda's newer stuff later (and that
wasn't reported in the paper). But I think getting 98% accuracy is the same as
what we had earlier.

```
(py2-torch) seita@triton1:~/tf_practice/torch$ time python bedmake.py --optim sgd --model resnet18

Now training!! On device: cuda:0
class_names: ['failure', 'success']
dataset_sizes: {'train': 911, 'valid': 230}


Epoch 0/19
--------------------
(train)  Loss: 0.4145, Acc: 0.8310 (num: 757)
(valid)  Loss: 0.4161, Acc: 0.8261 (num: 190)

Epoch 1/19
--------------------
(train)  Loss: 0.3920, Acc: 0.9001 (num: 820)
(valid)  Loss: 4.9542, Acc: 0.5348 (num: 123)

Epoch 2/19
--------------------
(train)  Loss: 0.1574, Acc: 0.9495 (num: 865)
(valid)  Loss: 0.8814, Acc: 0.7174 (num: 165)

Epoch 3/19
--------------------
(train)  Loss: 0.0668, Acc: 0.9726 (num: 886)
(valid)  Loss: 0.1857, Acc: 0.9130 (num: 210)

Epoch 4/19
--------------------
(train)  Loss: 0.1092, Acc: 0.9649 (num: 879)
(valid)  Loss: 1.3848, Acc: 0.6652 (num: 153)

Epoch 5/19
--------------------
(train)  Loss: 0.1103, Acc: 0.9671 (num: 881)
(valid)  Loss: 1.5274, Acc: 0.6348 (num: 146)

Epoch 6/19
--------------------
(train)  Loss: 0.0711, Acc: 0.9759 (num: 889)
(valid)  Loss: 0.1082, Acc: 0.9478 (num: 218)

Epoch 7/19
--------------------
(train)  Loss: 0.0705, Acc: 0.9769 (num: 890)
(valid)  Loss: 0.5025, Acc: 0.8043 (num: 185)

Epoch 8/19
--------------------
(train)  Loss: 0.0438, Acc: 0.9901 (num: 902)
(valid)  Loss: 0.0598, Acc: 0.9913 (num: 228)

Epoch 9/19
--------------------
(train)  Loss: 0.0492, Acc: 0.9846 (num: 897)
(valid)  Loss: 0.7295, Acc: 0.7478 (num: 172)

Epoch 10/19
--------------------
(train)  Loss: 0.0614, Acc: 0.9802 (num: 893)
(valid)  Loss: 0.1577, Acc: 0.9391 (num: 216)

Epoch 11/19
--------------------
(train)  Loss: 0.0362, Acc: 0.9912 (num: 903)
(valid)  Loss: 0.1936, Acc: 0.9217 (num: 212)

Epoch 12/19
--------------------
(train)  Loss: 0.0096, Acc: 0.9978 (num: 909)
(valid)  Loss: 0.0839, Acc: 0.9739 (num: 224)

Epoch 13/19
--------------------
(train)  Loss: 0.0106, Acc: 0.9956 (num: 907)
(valid)  Loss: 0.0406, Acc: 0.9826 (num: 226)

Epoch 14/19
--------------------
(train)  Loss: 0.0114, Acc: 0.9967 (num: 908)
(valid)  Loss: 0.1256, Acc: 0.9609 (num: 221)

Epoch 15/19
--------------------
(train)  Loss: 0.0204, Acc: 0.9923 (num: 904)
(valid)  Loss: 0.0451, Acc: 0.9870 (num: 227)

Epoch 16/19
--------------------
(train)  Loss: 0.0206, Acc: 0.9934 (num: 905)
(valid)  Loss: 0.0963, Acc: 0.9652 (num: 222)

Epoch 17/19
--------------------
(train)  Loss: 0.0107, Acc: 0.9967 (num: 908)
(valid)  Loss: 0.0500, Acc: 0.9739 (num: 224)

Epoch 18/19
--------------------
(train)  Loss: 0.0158, Acc: 0.9923 (num: 904)
(valid)  Loss: 0.1077, Acc: 0.9391 (num: 216)

Epoch 19/19
--------------------
(train)  Loss: 0.0118, Acc: 0.9967 (num: 908)
(valid)  Loss: 0.0498, Acc: 0.9783 (num: 225)

Trained in 0m 56s
Best val Acc: 0.991304

real	1m0.988s
user	3m5.191s
sys	0m31.154s
(py2-torch) seita@triton1:~/tf_practice/torch$ time python bedmake.py --optim sgd --model resnet34

Now training!! On device: cuda:0
class_names: ['failure', 'success']
dataset_sizes: {'train': 911, 'valid': 230}


Epoch 0/19
--------------------
(train)  Loss: 0.4198, Acc: 0.8332 (num: 759)
(valid)  Loss: 2.8053, Acc: 0.4870 (num: 112)

Epoch 1/19
--------------------
(train)  Loss: 0.2954, Acc: 0.8957 (num: 816)
(valid)  Loss: 0.3450, Acc: 0.8304 (num: 191)

Epoch 2/19
--------------------
(train)  Loss: 0.1695, Acc: 0.9473 (num: 863)
(valid)  Loss: 0.7522, Acc: 0.7261 (num: 167)

Epoch 3/19
--------------------
(train)  Loss: 0.1076, Acc: 0.9649 (num: 879)
(valid)  Loss: 0.5370, Acc: 0.8304 (num: 191)

Epoch 4/19
--------------------
(train)  Loss: 0.1353, Acc: 0.9616 (num: 876)
(valid)  Loss: 1.1982, Acc: 0.6522 (num: 150)

Epoch 5/19
--------------------
(train)  Loss: 0.0628, Acc: 0.9791 (num: 892)
(valid)  Loss: 0.1571, Acc: 0.9348 (num: 215)

Epoch 6/19
--------------------
(train)  Loss: 0.0592, Acc: 0.9802 (num: 893)
(valid)  Loss: 0.2908, Acc: 0.8652 (num: 199)

Epoch 7/19
--------------------
(train)  Loss: 0.0400, Acc: 0.9846 (num: 897)
(valid)  Loss: 0.0866, Acc: 0.9609 (num: 221)

Epoch 8/19
--------------------
(train)  Loss: 0.0423, Acc: 0.9846 (num: 897)
(valid)  Loss: 0.1994, Acc: 0.9130 (num: 210)

Epoch 9/19
--------------------
(train)  Loss: 0.0428, Acc: 0.9791 (num: 892)
(valid)  Loss: 0.1049, Acc: 0.9609 (num: 221)

Epoch 10/19
--------------------
(train)  Loss: 0.0383, Acc: 0.9857 (num: 898)
(valid)  Loss: 0.1187, Acc: 0.9522 (num: 219)

Epoch 11/19
--------------------
(train)  Loss: 0.0424, Acc: 0.9879 (num: 900)
(valid)  Loss: 0.0801, Acc: 0.9609 (num: 221)

Epoch 12/19
--------------------
(train)  Loss: 0.0308, Acc: 0.9890 (num: 901)
(valid)  Loss: 0.2078, Acc: 0.9043 (num: 208)

Epoch 13/19
--------------------
(train)  Loss: 0.0477, Acc: 0.9857 (num: 898)
(valid)  Loss: 0.1328, Acc: 0.9565 (num: 220)

Epoch 14/19
--------------------
(train)  Loss: 0.0162, Acc: 0.9956 (num: 907)
(valid)  Loss: 0.0835, Acc: 0.9696 (num: 223)

Epoch 15/19
--------------------
(train)  Loss: 0.0165, Acc: 0.9945 (num: 906)
(valid)  Loss: 0.0660, Acc: 0.9783 (num: 225)

Epoch 16/19
--------------------
(train)  Loss: 0.0134, Acc: 0.9945 (num: 906)
(valid)  Loss: 0.1231, Acc: 0.9652 (num: 222)

Epoch 17/19
--------------------
(train)  Loss: 0.0222, Acc: 0.9934 (num: 905)
(valid)  Loss: 0.0426, Acc: 0.9826 (num: 226)

Epoch 18/19
--------------------
(train)  Loss: 0.0122, Acc: 0.9956 (num: 907)
(valid)  Loss: 0.0790, Acc: 0.9565 (num: 220)

Epoch 19/19
--------------------
(train)  Loss: 0.0208, Acc: 0.9945 (num: 906)
(valid)  Loss: 0.1405, Acc: 0.9478 (num: 218)

Trained in 1m 25s
Best val Acc: 0.982609

real	1m30.273s
user	3m23.670s
sys	0m40.532s
(py2-torch) seita@triton1:~/tf_practice/torch$ time python bedmake.py --optim sgd --model resnet50

Now training!! On device: cuda:0
class_names: ['failure', 'success']
dataset_sizes: {'train': 911, 'valid': 230}


Epoch 0/19
--------------------
(train)  Loss: 0.5465, Acc: 0.7849 (num: 715)
(valid)  Loss: 11.1771, Acc: 0.5348 (num: 123)

Epoch 1/19
--------------------
(train)  Loss: 0.2460, Acc: 0.9155 (num: 834)
(valid)  Loss: 6.2254, Acc: 0.5348 (num: 123)

Epoch 2/19
--------------------
(train)  Loss: 0.1464, Acc: 0.9528 (num: 868)
(valid)  Loss: 0.1897, Acc: 0.9174 (num: 211)

Epoch 3/19
--------------------
(train)  Loss: 0.0908, Acc: 0.9638 (num: 878)
(valid)  Loss: 0.8140, Acc: 0.7087 (num: 163)

Epoch 4/19
--------------------
(train)  Loss: 0.0931, Acc: 0.9660 (num: 880)
(valid)  Loss: 0.5400, Acc: 0.8783 (num: 202)

Epoch 5/19
--------------------
(train)  Loss: 0.0639, Acc: 0.9769 (num: 890)
(valid)  Loss: 1.1298, Acc: 0.7609 (num: 175)

Epoch 6/19
--------------------
(train)  Loss: 0.0706, Acc: 0.9748 (num: 888)
(valid)  Loss: 0.9547, Acc: 0.6304 (num: 145)

Epoch 7/19
--------------------
(train)  Loss: 0.0616, Acc: 0.9769 (num: 890)
(valid)  Loss: 0.6100, Acc: 0.7870 (num: 181)

Epoch 8/19
--------------------
(train)  Loss: 0.0678, Acc: 0.9737 (num: 887)
(valid)  Loss: 1.2953, Acc: 0.8130 (num: 187)

Epoch 9/19
--------------------
(train)  Loss: 0.0581, Acc: 0.9780 (num: 891)
(valid)  Loss: 0.1592, Acc: 0.9391 (num: 216)

Epoch 10/19
--------------------
(train)  Loss: 0.0329, Acc: 0.9879 (num: 900)
(valid)  Loss: 0.0680, Acc: 0.9783 (num: 225)

Epoch 11/19
--------------------
(train)  Loss: 0.0331, Acc: 0.9879 (num: 900)
(valid)  Loss: 0.0378, Acc: 0.9870 (num: 227)

Epoch 12/19
--------------------
(train)  Loss: 0.0209, Acc: 0.9923 (num: 904)
(valid)  Loss: 0.0589, Acc: 0.9739 (num: 224)

Epoch 13/19
--------------------
(train)  Loss: 0.0125, Acc: 0.9956 (num: 907)
(valid)  Loss: 0.0822, Acc: 0.9783 (num: 225)

Epoch 14/19
--------------------
(train)  Loss: 0.0105, Acc: 0.9956 (num: 907)
(valid)  Loss: 0.1214, Acc: 0.9478 (num: 218)

Epoch 15/19
--------------------
(train)  Loss: 0.0277, Acc: 0.9912 (num: 903)
(valid)  Loss: 0.0938, Acc: 0.9652 (num: 222)

Epoch 16/19
--------------------
(train)  Loss: 0.0186, Acc: 0.9923 (num: 904)
(valid)  Loss: 0.0635, Acc: 0.9739 (num: 224)

Epoch 17/19
--------------------
(train)  Loss: 0.0121, Acc: 0.9967 (num: 908)
(valid)  Loss: 0.0268, Acc: 0.9913 (num: 228)

Epoch 18/19
--------------------
(train)  Loss: 0.0154, Acc: 0.9978 (num: 909)
(valid)  Loss: 0.1104, Acc: 0.9565 (num: 220)

Epoch 19/19
--------------------
(train)  Loss: 0.0104, Acc: 0.9967 (num: 908)
(valid)  Loss: 0.3413, Acc: 0.8913 (num: 205)

Trained in 1m 60s
Best val Acc: 0.991304

real	2m3.684s
user	3m57.151s
sys	    0m44.915s
```
