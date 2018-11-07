# Benchmarking BedMake Results

(Trying to reproduce results from my paper.)

Actually these are already on par with what I am getting. Tricky to compare
since I think we had to adjust the data with Honda's newer stuff later (and that
wasn't reported in the paper). But I think getting 98% accuracy is the same as
what we had earlier.


- [SGD](#sgd)
- [Adam](#adam)

Informal conclusions: we are able to get 98 or 99 percent validation accuracy
over a variety of settings. Training set accuracy goes higher first, as
expected. It is somewhat unclear with regards to ResNet-{18,34,50} which one is
better. Also, SGD (lr 0.01) and Adam (lr 0.0001) both do well after the first
few epochs, except Adam does better initially. Hard to tell, I'd probably go
with Adam, naturally.

## SGD

```
(py2-torch) seita@triton1:~/tf_practice/torch$ time python bedmake.py --optim sgd --model resnet18 --num_epochs 20

Now training!! On device: cuda:0
class_names: ['failure', 'success']
dataset_sizes: {'train': 911, 'valid': 230}


Epoch 0/19
--------------------
(train)  Loss: 0.3793, Acc: 0.8299 (num: 756)
(valid)  Loss: 0.6208, Acc: 0.6870 (num: 158)

Epoch 1/19
--------------------
(train)  Loss: 0.2268, Acc: 0.9243 (num: 842)
(valid)  Loss: 2.9801, Acc: 0.4739 (num: 109)

Epoch 2/19
--------------------
(train)  Loss: 0.2127, Acc: 0.9199 (num: 838)
(valid)  Loss: 0.4201, Acc: 0.7783 (num: 179)

Epoch 3/19
--------------------
(train)  Loss: 0.1210, Acc: 0.9528 (num: 868)
(valid)  Loss: 2.6231, Acc: 0.5870 (num: 135)

Epoch 4/19
--------------------
(train)  Loss: 0.1426, Acc: 0.9561 (num: 871)
(valid)  Loss: 0.8555, Acc: 0.6217 (num: 143)

Epoch 5/19
--------------------
(train)  Loss: 0.1030, Acc: 0.9638 (num: 878)
(valid)  Loss: 0.5536, Acc: 0.7696 (num: 177)

Epoch 6/19
--------------------
(train)  Loss: 0.0949, Acc: 0.9693 (num: 883)
(valid)  Loss: 0.1803, Acc: 0.9522 (num: 219)

Epoch 7/19
--------------------
(train)  Loss: 0.1233, Acc: 0.9605 (num: 875)
(valid)  Loss: 0.5537, Acc: 0.8913 (num: 205)

Epoch 8/19
--------------------
(train)  Loss: 0.0638, Acc: 0.9780 (num: 891)
(valid)  Loss: 0.4921, Acc: 0.8043 (num: 185)

Epoch 9/19
--------------------
(train)  Loss: 0.0394, Acc: 0.9868 (num: 899)
(valid)  Loss: 0.1085, Acc: 0.9609 (num: 221)

Epoch 10/19
--------------------
(train)  Loss: 0.0452, Acc: 0.9846 (num: 897)
(valid)  Loss: 0.1568, Acc: 0.9304 (num: 214)

Epoch 11/19
--------------------
(train)  Loss: 0.0219, Acc: 0.9934 (num: 905)
(valid)  Loss: 0.0614, Acc: 0.9783 (num: 225)

Epoch 12/19
--------------------
(train)  Loss: 0.0270, Acc: 0.9890 (num: 901)
(valid)  Loss: 0.0573, Acc: 0.9739 (num: 224)

Epoch 13/19
--------------------
(train)  Loss: 0.0198, Acc: 0.9912 (num: 903)
(valid)  Loss: 0.0666, Acc: 0.9739 (num: 224)

Epoch 14/19
--------------------
(train)  Loss: 0.0306, Acc: 0.9890 (num: 901)
(valid)  Loss: 0.0718, Acc: 0.9696 (num: 223)

Epoch 15/19
--------------------
(train)  Loss: 0.0190, Acc: 0.9912 (num: 903)
(valid)  Loss: 1.3709, Acc: 0.7391 (num: 170)

Epoch 16/19
--------------------
(train)  Loss: 0.0194, Acc: 0.9934 (num: 905)
(valid)  Loss: 0.1772, Acc: 0.9174 (num: 211)

Epoch 17/19
--------------------
(train)  Loss: 0.0347, Acc: 0.9846 (num: 897)
(valid)  Loss: 0.2843, Acc: 0.8826 (num: 203)

Epoch 18/19
--------------------
(train)  Loss: 0.0156, Acc: 0.9923 (num: 904)
(valid)  Loss: 0.0365, Acc: 0.9826 (num: 226)

Epoch 19/19
--------------------
(train)  Loss: 0.0570, Acc: 0.9846 (num: 897)
(valid)  Loss: 0.2959, Acc: 0.9000 (num: 207)

Trained in 0m 57s
Best val Acc: 0.982609
train:  [0.83, 0.924, 0.92, 0.953, 0.956, 0.964, 0.969, 0.96, 0.978, 0.987, 0.985, 0.993, 0.989, 0.991, 0.989, 0.991, 0.993, 0.985, 0.992, 0.985]
valid:  [0.687, 0.474, 0.778, 0.587, 0.622, 0.77, 0.952, 0.891, 0.804, 0.961, 0.93, 0.978, 0.974, 0.974, 0.97, 0.739, 0.917, 0.883, 0.983, 0.9]

real	1m3.048s
user	3m8.431s
sys	0m33.544s
(py2-torch) seita@triton1:~/tf_practice/torch$ time python bedmake.py --optim sgd --model resnet34 --num_epochs 20

Now training!! On device: cuda:0
class_names: ['failure', 'success']
dataset_sizes: {'train': 911, 'valid': 230}


Epoch 0/19
--------------------
(train)  Loss: 0.4384, Acc: 0.8266 (num: 753)
(valid)  Loss: 0.4217, Acc: 0.8174 (num: 188)

Epoch 1/19
--------------------
(train)  Loss: 0.1879, Acc: 0.9374 (num: 854)
(valid)  Loss: 1.9833, Acc: 0.6478 (num: 149)

Epoch 2/19
--------------------
(train)  Loss: 0.1668, Acc: 0.9429 (num: 859)
(valid)  Loss: 0.3678, Acc: 0.8826 (num: 203)

Epoch 3/19
--------------------
(train)  Loss: 0.1079, Acc: 0.9671 (num: 881)
(valid)  Loss: 0.1444, Acc: 0.9261 (num: 213)

Epoch 4/19
--------------------
(train)  Loss: 0.0877, Acc: 0.9660 (num: 880)
(valid)  Loss: 0.2717, Acc: 0.8696 (num: 200)

Epoch 5/19
--------------------
(train)  Loss: 0.0522, Acc: 0.9769 (num: 890)
(valid)  Loss: 0.5856, Acc: 0.8174 (num: 188)

Epoch 6/19
--------------------
(train)  Loss: 0.0580, Acc: 0.9813 (num: 894)
(valid)  Loss: 0.3217, Acc: 0.8522 (num: 196)

Epoch 7/19
--------------------
(train)  Loss: 0.0840, Acc: 0.9671 (num: 881)
(valid)  Loss: 1.7786, Acc: 0.5217 (num: 120)

Epoch 8/19
--------------------
(train)  Loss: 0.0590, Acc: 0.9802 (num: 893)
(valid)  Loss: 0.1715, Acc: 0.9391 (num: 216)

Epoch 9/19
--------------------
(train)  Loss: 0.0362, Acc: 0.9846 (num: 897)
(valid)  Loss: 0.1021, Acc: 0.9565 (num: 220)

Epoch 10/19
--------------------
(train)  Loss: 0.0286, Acc: 0.9879 (num: 900)
(valid)  Loss: 0.0865, Acc: 0.9739 (num: 224)

Epoch 11/19
--------------------
(train)  Loss: 0.0268, Acc: 0.9901 (num: 902)
(valid)  Loss: 0.1528, Acc: 0.9435 (num: 217)

Epoch 12/19
--------------------
(train)  Loss: 0.0326, Acc: 0.9901 (num: 902)
(valid)  Loss: 0.0539, Acc: 0.9826 (num: 226)

Epoch 13/19
--------------------
(train)  Loss: 0.0141, Acc: 0.9945 (num: 906)
(valid)  Loss: 0.0554, Acc: 0.9696 (num: 223)

Epoch 14/19
--------------------
(train)  Loss: 0.0119, Acc: 0.9967 (num: 908)
(valid)  Loss: 0.0465, Acc: 0.9826 (num: 226)

Epoch 15/19
--------------------
(train)  Loss: 0.0058, Acc: 0.9978 (num: 909)
(valid)  Loss: 0.2549, Acc: 0.9130 (num: 210)

Epoch 16/19
--------------------
(train)  Loss: 0.0168, Acc: 0.9934 (num: 905)
(valid)  Loss: 0.3106, Acc: 0.9217 (num: 212)

Epoch 17/19
--------------------
(train)  Loss: 0.0402, Acc: 0.9901 (num: 902)
(valid)  Loss: 0.3936, Acc: 0.8696 (num: 200)

Epoch 18/19
--------------------
(train)  Loss: 0.0309, Acc: 0.9835 (num: 896)
(valid)  Loss: 0.2415, Acc: 0.9130 (num: 210)

Epoch 19/19
--------------------
(train)  Loss: 0.0285, Acc: 0.9890 (num: 901)
(valid)  Loss: 0.1482, Acc: 0.9348 (num: 215)

Trained in 1m 25s
Best val Acc: 0.982609
train:  [0.827, 0.937, 0.943, 0.967, 0.966, 0.977, 0.981, 0.967, 0.98, 0.985, 0.988, 0.99, 0.99, 0.995, 0.997, 0.998, 0.993, 0.99, 0.984, 0.989]
valid:  [0.817, 0.648, 0.883, 0.926, 0.87, 0.817, 0.852, 0.522, 0.939, 0.957, 0.974, 0.943, 0.983, 0.97, 0.983, 0.913, 0.922, 0.87, 0.913, 0.935]

real	1m28.707s
user	3m24.706s
sys	0m39.662s
(py2-torch) seita@triton1:~/tf_practice/torch$ time python bedmake.py --optim sgd --model resnet50 --num_epochs 20

Now training!! On device: cuda:0
class_names: ['failure', 'success']
dataset_sizes: {'train': 911, 'valid': 230}


Epoch 0/19
--------------------
(train)  Loss: 0.3924, Acc: 0.8364 (num: 762)
(valid)  Loss: 1.4466, Acc: 0.6087 (num: 140)

Epoch 1/19
--------------------
(train)  Loss: 0.3320, Acc: 0.8924 (num: 813)
(valid)  Loss: 2.4966, Acc: 0.6261 (num: 144)

Epoch 2/19
--------------------
(train)  Loss: 0.1134, Acc: 0.9583 (num: 873)
(valid)  Loss: 0.1024, Acc: 0.9478 (num: 218)

Epoch 3/19
--------------------
(train)  Loss: 0.1534, Acc: 0.9451 (num: 861)
(valid)  Loss: 15.4600, Acc: 0.5348 (num: 123)

Epoch 4/19
--------------------
(train)  Loss: 0.1366, Acc: 0.9528 (num: 868)
(valid)  Loss: 0.9074, Acc: 0.7043 (num: 162)

Epoch 5/19
--------------------
(train)  Loss: 0.0854, Acc: 0.9715 (num: 885)
(valid)  Loss: 0.1686, Acc: 0.9391 (num: 216)

Epoch 6/19
--------------------
(train)  Loss: 0.0686, Acc: 0.9748 (num: 888)
(valid)  Loss: 0.3593, Acc: 0.8565 (num: 197)

Epoch 7/19
--------------------
(train)  Loss: 0.0749, Acc: 0.9726 (num: 886)
(valid)  Loss: 7.0674, Acc: 0.5435 (num: 125)

Epoch 8/19
--------------------
(train)  Loss: 0.0690, Acc: 0.9813 (num: 894)
(valid)  Loss: 0.1395, Acc: 0.9478 (num: 218)

Epoch 9/19
--------------------
(train)  Loss: 0.0294, Acc: 0.9857 (num: 898)
(valid)  Loss: 0.1206, Acc: 0.9565 (num: 220)

Epoch 10/19
--------------------
(train)  Loss: 0.0271, Acc: 0.9879 (num: 900)
(valid)  Loss: 0.1641, Acc: 0.9696 (num: 223)

Epoch 11/19
--------------------
(train)  Loss: 0.0391, Acc: 0.9868 (num: 899)
(valid)  Loss: 0.1017, Acc: 0.9609 (num: 221)

Epoch 12/19
--------------------
(train)  Loss: 0.0200, Acc: 0.9923 (num: 904)
(valid)  Loss: 0.1085, Acc: 0.9565 (num: 220)

Epoch 13/19
--------------------
(train)  Loss: 0.0212, Acc: 0.9945 (num: 906)
(valid)  Loss: 0.0767, Acc: 0.9783 (num: 225)

Epoch 14/19
--------------------
(train)  Loss: 0.0125, Acc: 0.9967 (num: 908)
(valid)  Loss: 0.0785, Acc: 0.9870 (num: 227)

Epoch 15/19
--------------------
(train)  Loss: 0.0206, Acc: 0.9956 (num: 907)
(valid)  Loss: 0.0967, Acc: 0.9652 (num: 222)

Epoch 16/19
--------------------
(train)  Loss: 0.0402, Acc: 0.9857 (num: 898)
(valid)  Loss: 0.2254, Acc: 0.9217 (num: 212)

Epoch 17/19
--------------------
(train)  Loss: 0.0381, Acc: 0.9901 (num: 902)
(valid)  Loss: 0.1894, Acc: 0.9217 (num: 212)

Epoch 18/19
--------------------
(train)  Loss: 0.0248, Acc: 0.9890 (num: 901)
(valid)  Loss: 0.1328, Acc: 0.9609 (num: 221)

Epoch 19/19
--------------------
(train)  Loss: 0.0436, Acc: 0.9890 (num: 901)
(valid)  Loss: 0.1252, Acc: 0.9478 (num: 218)

Trained in 2m 0s
Best val Acc: 0.986957
train:  [0.836, 0.892, 0.958, 0.945, 0.953, 0.971, 0.975, 0.973, 0.981, 0.986, 0.988, 0.987, 0.992, 0.995, 0.997, 0.996, 0.986, 0.99, 0.989, 0.989]
valid:  [0.609, 0.626, 0.948, 0.535, 0.704, 0.939, 0.857, 0.543, 0.948, 0.957, 0.97, 0.961, 0.957, 0.978, 0.987, 0.965, 0.922, 0.922, 0.961, 0.948]

real	2m4.381s
user	3m56.326s
sys	0m44.532s
```



## Adam

```
(py2-torch) seita@triton1:~/tf_practice/torch$ time python bedmake.py --optim adam --model resnet18 --num_epochs 20

Now training!! On device: cuda:0
class_names: ['failure', 'success']
dataset_sizes: {'train': 911, 'valid': 230}


Epoch 0/19
--------------------
(train)  Loss: 0.2288, Acc: 0.9012 (num: 821)
(valid)  Loss: 0.2247, Acc: 0.9087 (num: 209)

Epoch 1/19
--------------------
(train)  Loss: 0.0913, Acc: 0.9649 (num: 879)
(valid)  Loss: 0.2539, Acc: 0.9130 (num: 210)

Epoch 2/19
--------------------
(train)  Loss: 0.0713, Acc: 0.9769 (num: 890)
(valid)  Loss: 0.1482, Acc: 0.9348 (num: 215)

Epoch 3/19
--------------------
(train)  Loss: 0.0672, Acc: 0.9769 (num: 890)
(valid)  Loss: 0.1816, Acc: 0.9304 (num: 214)

Epoch 4/19
--------------------
(train)  Loss: 0.0414, Acc: 0.9901 (num: 902)
(valid)  Loss: 0.0736, Acc: 0.9783 (num: 225)

Epoch 5/19
--------------------
(train)  Loss: 0.0353, Acc: 0.9879 (num: 900)
(valid)  Loss: 0.1105, Acc: 0.9478 (num: 218)

Epoch 6/19
--------------------
(train)  Loss: 0.0316, Acc: 0.9901 (num: 902)
(valid)  Loss: 0.0808, Acc: 0.9652 (num: 222)

Epoch 7/19
--------------------
(train)  Loss: 0.0200, Acc: 0.9912 (num: 903)
(valid)  Loss: 0.0484, Acc: 0.9783 (num: 225)

Epoch 8/19
--------------------
(train)  Loss: 0.0138, Acc: 0.9934 (num: 905)
(valid)  Loss: 0.0709, Acc: 0.9652 (num: 222)

Epoch 9/19
--------------------
(train)  Loss: 0.0129, Acc: 0.9956 (num: 907)
(valid)  Loss: 0.0899, Acc: 0.9696 (num: 223)

Epoch 10/19
--------------------
(train)  Loss: 0.0279, Acc: 0.9901 (num: 902)
(valid)  Loss: 0.0458, Acc: 0.9783 (num: 225)

Epoch 11/19
--------------------
(train)  Loss: 0.0115, Acc: 0.9956 (num: 907)
(valid)  Loss: 0.0424, Acc: 0.9826 (num: 226)

Epoch 12/19
--------------------
(train)  Loss: 0.0058, Acc: 0.9989 (num: 910)
(valid)  Loss: 0.0303, Acc: 0.9870 (num: 227)

Epoch 13/19
--------------------
(train)  Loss: 0.0091, Acc: 0.9978 (num: 909)
(valid)  Loss: 0.1357, Acc: 0.9435 (num: 217)

Epoch 14/19
--------------------
(train)  Loss: 0.0126, Acc: 0.9967 (num: 908)
(valid)  Loss: 0.0565, Acc: 0.9739 (num: 224)

Epoch 15/19
--------------------
(train)  Loss: 0.0119, Acc: 0.9945 (num: 906)
(valid)  Loss: 0.1082, Acc: 0.9652 (num: 222)

Epoch 16/19
--------------------
(train)  Loss: 0.0499, Acc: 0.9835 (num: 896)
(valid)  Loss: 0.0621, Acc: 0.9783 (num: 225)

Epoch 17/19
--------------------
(train)  Loss: 0.0164, Acc: 0.9923 (num: 904)
(valid)  Loss: 0.0847, Acc: 0.9609 (num: 221)

Epoch 18/19
--------------------
(train)  Loss: 0.0099, Acc: 0.9989 (num: 910)
(valid)  Loss: 0.0399, Acc: 0.9870 (num: 227)

Epoch 19/19
--------------------
(train)  Loss: 0.0065, Acc: 0.9989 (num: 910)
(valid)  Loss: 0.0393, Acc: 0.9739 (num: 224)

Trained in 0m 58s
Best val Acc: 0.986957
train:  [0.901, 0.965, 0.977, 0.977, 0.99, 0.988, 0.99, 0.991, 0.993, 0.996, 0.99, 0.996, 0.999, 0.998, 0.997, 0.995, 0.984, 0.992, 0.999, 0.999]
valid:  [0.909, 0.913, 0.935, 0.93, 0.978, 0.948, 0.965, 0.978, 0.965, 0.97, 0.978, 0.983, 0.987, 0.943, 0.974, 0.965, 0.978, 0.961, 0.987, 0.974]

real	1m2.265s
user	3m10.310s
sys	0m33.481s
(py2-torch) seita@triton1:~/tf_practice/torch$ time python bedmake.py --optim adam --model resnet34 --num_epochs 20

Now training!! On device: cuda:0
class_names: ['failure', 'success']
dataset_sizes: {'train': 911, 'valid': 230}


Epoch 0/19
--------------------
(train)  Loss: 0.2285, Acc: 0.9001 (num: 820)
(valid)  Loss: 0.4531, Acc: 0.8130 (num: 187)

Epoch 1/19
--------------------
(train)  Loss: 0.1093, Acc: 0.9605 (num: 875)
(valid)  Loss: 0.0986, Acc: 0.9739 (num: 224)

Epoch 2/19
--------------------
(train)  Loss: 0.0618, Acc: 0.9791 (num: 892)
(valid)  Loss: 0.1188, Acc: 0.9696 (num: 223)

Epoch 3/19
--------------------
(train)  Loss: 0.0423, Acc: 0.9901 (num: 902)
(valid)  Loss: 0.1457, Acc: 0.9391 (num: 216)

Epoch 4/19
--------------------
(train)  Loss: 0.0413, Acc: 0.9846 (num: 897)
(valid)  Loss: 0.1337, Acc: 0.9435 (num: 217)

Epoch 5/19
--------------------
(train)  Loss: 0.0372, Acc: 0.9868 (num: 899)
(valid)  Loss: 0.0601, Acc: 0.9783 (num: 225)

Epoch 6/19
--------------------
(train)  Loss: 0.0468, Acc: 0.9835 (num: 896)
(valid)  Loss: 0.0487, Acc: 0.9870 (num: 227)

Epoch 7/19
--------------------
(train)  Loss: 0.0274, Acc: 0.9923 (num: 904)
(valid)  Loss: 0.0655, Acc: 0.9783 (num: 225)

Epoch 8/19
--------------------
(train)  Loss: 0.0183, Acc: 0.9967 (num: 908)
(valid)  Loss: 0.0600, Acc: 0.9739 (num: 224)

Epoch 9/19
--------------------
(train)  Loss: 0.0228, Acc: 0.9901 (num: 902)
(valid)  Loss: 0.1416, Acc: 0.9435 (num: 217)

Epoch 10/19
--------------------
(train)  Loss: 0.0149, Acc: 0.9945 (num: 906)
(valid)  Loss: 0.0488, Acc: 0.9783 (num: 225)

Epoch 11/19
--------------------
(train)  Loss: 0.0046, Acc: 0.9989 (num: 910)
(valid)  Loss: 0.0964, Acc: 0.9739 (num: 224)

Epoch 12/19
--------------------
(train)  Loss: 0.0024, Acc: 1.0000 (num: 911)
(valid)  Loss: 0.0512, Acc: 0.9783 (num: 225)

Epoch 13/19
--------------------
(train)  Loss: 0.0402, Acc: 0.9890 (num: 901)
(valid)  Loss: 0.0290, Acc: 0.9913 (num: 228)

Epoch 14/19
--------------------
(train)  Loss: 0.0417, Acc: 0.9813 (num: 894)
(valid)  Loss: 0.1187, Acc: 0.9609 (num: 221)

Epoch 15/19
--------------------
(train)  Loss: 0.0235, Acc: 0.9934 (num: 905)
(valid)  Loss: 0.0892, Acc: 0.9783 (num: 225)

Epoch 16/19
--------------------
(train)  Loss: 0.0131, Acc: 0.9945 (num: 906)
(valid)  Loss: 0.0536, Acc: 0.9870 (num: 227)

Epoch 17/19
--------------------
(train)  Loss: 0.0130, Acc: 0.9956 (num: 907)
(valid)  Loss: 0.0895, Acc: 0.9565 (num: 220)

Epoch 18/19
--------------------
(train)  Loss: 0.0261, Acc: 0.9923 (num: 904)
(valid)  Loss: 0.2290, Acc: 0.9391 (num: 216)

Epoch 19/19
--------------------
(train)  Loss: 0.0297, Acc: 0.9890 (num: 901)
(valid)  Loss: 0.0897, Acc: 0.9696 (num: 223)

Trained in 1m 27s
Best val Acc: 0.991304
train:  [0.9, 0.96, 0.979, 0.99, 0.985, 0.987, 0.984, 0.992, 0.997, 0.99, 0.995, 0.999, 1.0, 0.989, 0.981, 0.993, 0.995, 0.996, 0.992, 0.989]
valid:  [0.813, 0.974, 0.97, 0.939, 0.943, 0.978, 0.987, 0.978, 0.974, 0.943, 0.978, 0.974, 0.978, 0.991, 0.961, 0.978, 0.987, 0.957, 0.939, 0.97]

real	1m32.851s
user	3m32.343s
sys	0m36.455s
(py2-torch) seita@triton1:~/tf_practice/torch$ time python bedmake.py --optim adam --model resnet50 --num_epochs 20

Now training!! On device: cuda:0
class_names: ['failure', 'success']
dataset_sizes: {'train': 911, 'valid': 230}


Epoch 0/19
--------------------
(train)  Loss: 0.2268, Acc: 0.9177 (num: 836)
(valid)  Loss: 0.0976, Acc: 0.9565 (num: 220)

Epoch 1/19
--------------------
(train)  Loss: 0.1112, Acc: 0.9605 (num: 875)
(valid)  Loss: 0.4457, Acc: 0.8174 (num: 188)

Epoch 2/19
--------------------
(train)  Loss: 0.0601, Acc: 0.9846 (num: 897)
(valid)  Loss: 0.1142, Acc: 0.9478 (num: 218)

Epoch 3/19
--------------------
(train)  Loss: 0.0531, Acc: 0.9835 (num: 896)
(valid)  Loss: 0.1444, Acc: 0.9261 (num: 213)

Epoch 4/19
--------------------
(train)  Loss: 0.0405, Acc: 0.9868 (num: 899)
(valid)  Loss: 0.0578, Acc: 0.9826 (num: 226)

Epoch 5/19
--------------------
(train)  Loss: 0.0247, Acc: 0.9923 (num: 904)
(valid)  Loss: 0.1152, Acc: 0.9696 (num: 223)

Epoch 6/19
--------------------
(train)  Loss: 0.0235, Acc: 0.9934 (num: 905)
(valid)  Loss: 0.0576, Acc: 0.9826 (num: 226)

Epoch 7/19
--------------------
(train)  Loss: 0.0426, Acc: 0.9857 (num: 898)
(valid)  Loss: 0.0724, Acc: 0.9652 (num: 222)

Epoch 8/19
--------------------
(train)  Loss: 0.0409, Acc: 0.9890 (num: 901)
(valid)  Loss: 0.1149, Acc: 0.9435 (num: 217)

Epoch 9/19
--------------------
(train)  Loss: 0.0134, Acc: 0.9956 (num: 907)
(valid)  Loss: 0.0805, Acc: 0.9565 (num: 220)

Epoch 10/19
--------------------
(train)  Loss: 0.0062, Acc: 0.9989 (num: 910)
(valid)  Loss: 0.0690, Acc: 0.9783 (num: 225)

Epoch 11/19
--------------------
(train)  Loss: 0.0060, Acc: 0.9978 (num: 909)
(valid)  Loss: 0.0355, Acc: 0.9913 (num: 228)

Epoch 12/19
--------------------
(train)  Loss: 0.0505, Acc: 0.9857 (num: 898)
(valid)  Loss: 0.2926, Acc: 0.8783 (num: 202)

Epoch 13/19
--------------------
(train)  Loss: 0.0302, Acc: 0.9879 (num: 900)
(valid)  Loss: 0.1308, Acc: 0.9435 (num: 217)

Epoch 14/19
--------------------
(train)  Loss: 0.0223, Acc: 0.9934 (num: 905)
(valid)  Loss: 0.2446, Acc: 0.9174 (num: 211)

Epoch 15/19
--------------------
(train)  Loss: 0.0280, Acc: 0.9901 (num: 902)
(valid)  Loss: 0.0502, Acc: 0.9870 (num: 227)

Epoch 16/19
--------------------
(train)  Loss: 0.0162, Acc: 0.9945 (num: 906)
(valid)  Loss: 0.0396, Acc: 0.9957 (num: 229)

Epoch 17/19
--------------------
(train)  Loss: 0.0041, Acc: 0.9989 (num: 910)
(valid)  Loss: 0.0441, Acc: 0.9783 (num: 225)

Epoch 18/19
--------------------
(train)  Loss: 0.0297, Acc: 0.9934 (num: 905)
(valid)  Loss: 0.1145, Acc: 0.9565 (num: 220)

Epoch 19/19
--------------------
(train)  Loss: 0.0086, Acc: 0.9978 (num: 909)
(valid)  Loss: 0.0490, Acc: 0.9870 (num: 227)

Trained in 2m 1s
Best val Acc: 0.995652
train:  [0.918, 0.96, 0.985, 0.984, 0.987, 0.992, 0.993, 0.986, 0.989, 0.996, 0.999, 0.998, 0.986, 0.988, 0.993, 0.99, 0.995, 0.999, 0.993, 0.998]
valid:  [0.957, 0.817, 0.948, 0.926, 0.983, 0.97, 0.983, 0.965, 0.943, 0.957, 0.978, 0.991, 0.878, 0.943, 0.917, 0.987, 0.996, 0.978, 0.957, 0.987]

real	2m5.796s
user	4m11.029s
sys	0m31.795s
```
