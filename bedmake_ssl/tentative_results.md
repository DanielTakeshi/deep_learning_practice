# Tentative Results

Running this gets the output below. The validation performance improves. This is good. We will investigate more later.

```
$~/deep_learning_practice/bedmake_ssl$ python ryan_data.py --num_epochs 30

Now training!! On device: cuda:0
data_sizes: {'train': 55.0, 'valid': 18.0}


------------------------------
Epoch 0/29
------------------------------
  (train)
loss total:  1.6345
loss_pos:    0.2561
loss_ang:    1.3784
correct_ang: 0.2727
  (valid)
loss total:  1.4795
loss_pos:    0.1083
loss_ang:    1.3712
correct_ang: 0.3889
------------------------------

------------------------------
Epoch 1/29
------------------------------
  (train)
loss total:  1.4180
loss_pos:    0.0806
loss_ang:    1.3375
correct_ang: 0.2909
  (valid)
loss total:  1.3630
loss_pos:    0.0319
loss_ang:    1.3312
correct_ang: 0.3333
------------------------------

------------------------------
Epoch 2/29
------------------------------
  (train)
loss total:  1.2808
loss_pos:    0.0202
loss_ang:    1.2606
correct_ang: 0.5455
  (valid)
loss total:  1.3331
loss_pos:    0.0405
loss_ang:    1.2926
correct_ang: 0.3333
------------------------------

------------------------------
Epoch 3/29
------------------------------
  (train)
loss total:  1.2599
loss_pos:    0.0517
loss_ang:    1.2082
correct_ang: 0.4909
  (valid)
loss total:  1.3248
loss_pos:    0.0475
loss_ang:    1.2773
correct_ang: 0.3333
------------------------------

------------------------------
Epoch 4/29
------------------------------
  (train)
loss total:  1.1909
loss_pos:    0.0475
loss_ang:    1.1434
correct_ang: 0.5818
  (valid)
loss total:  1.3162
loss_pos:    0.0390
loss_ang:    1.2773
correct_ang: 0.3333
------------------------------

------------------------------
Epoch 5/29
------------------------------
  (train)
loss total:  1.1816
loss_pos:    0.0434
loss_ang:    1.1382
correct_ang: 0.4182
  (valid)
loss total:  1.2975
loss_pos:    0.0309
loss_ang:    1.2666
correct_ang: 0.3333
------------------------------

------------------------------
Epoch 6/29
------------------------------
  (train)
loss total:  1.0528
loss_pos:    0.0153
loss_ang:    1.0376
correct_ang: 0.5455
  (valid)
loss total:  1.2734
loss_pos:    0.0260
loss_ang:    1.2474
correct_ang: 0.3333
------------------------------

------------------------------
Epoch 7/29
------------------------------
  (train)
loss total:  1.0245
loss_pos:    0.0111
loss_ang:    1.0134
correct_ang: 0.5273
  (valid)
loss total:  1.2365
loss_pos:    0.0244
loss_ang:    1.2120
correct_ang: 0.3889
------------------------------

------------------------------
Epoch 8/29
------------------------------
  (train)
loss total:  0.9471
loss_pos:    0.0151
loss_ang:    0.9321
correct_ang: 0.6364
  (valid)
loss total:  1.1903
loss_pos:    0.0260
loss_ang:    1.1643
correct_ang: 0.5000
------------------------------

------------------------------
Epoch 9/29
------------------------------
  (train)
loss total:  0.9241
loss_pos:    0.0190
loss_ang:    0.9051
correct_ang: 0.6545
  (valid)
loss total:  1.1407
loss_pos:    0.0270
loss_ang:    1.1137
correct_ang: 0.7222
------------------------------

------------------------------
Epoch 10/29
------------------------------
  (train)
loss total:  0.8000
loss_pos:    0.0216
loss_ang:    0.7783
correct_ang: 0.8182
  (valid)
loss total:  1.0997
loss_pos:    0.0254
loss_ang:    1.0744
correct_ang: 0.7222
------------------------------

------------------------------
Epoch 11/29
------------------------------
  (train)
loss total:  0.7325
loss_pos:    0.0145
loss_ang:    0.7180
correct_ang: 0.8727
  (valid)
loss total:  1.0563
loss_pos:    0.0209
loss_ang:    1.0354
correct_ang: 0.7222
------------------------------

------------------------------
Epoch 12/29
------------------------------
  (train)
loss total:  0.6308
loss_pos:    0.0148
loss_ang:    0.6161
correct_ang: 0.9273
  (valid)
loss total:  1.0079
loss_pos:    0.0181
loss_ang:    0.9898
correct_ang: 0.7222
------------------------------

------------------------------
Epoch 13/29
------------------------------
  (train)
loss total:  0.5673
loss_pos:    0.0125
loss_ang:    0.5549
correct_ang: 0.9091
  (valid)
loss total:  0.9536
loss_pos:    0.0175
loss_ang:    0.9361
correct_ang: 0.7222
------------------------------

------------------------------
Epoch 14/29
------------------------------
  (train)
loss total:  0.4696
loss_pos:    0.0128
loss_ang:    0.4569
correct_ang: 0.9636
  (valid)
loss total:  0.9028
loss_pos:    0.0169
loss_ang:    0.8858
correct_ang: 0.7222
------------------------------

------------------------------
Epoch 15/29
------------------------------
  (train)
loss total:  0.4507
loss_pos:    0.0119
loss_ang:    0.4389
correct_ang: 0.9273
  (valid)
loss total:  0.8529
loss_pos:    0.0169
loss_ang:    0.8360
correct_ang: 0.7222
------------------------------

------------------------------
Epoch 16/29
------------------------------
  (train)
loss total:  0.3495
loss_pos:    0.0115
loss_ang:    0.3380
correct_ang: 0.9636
  (valid)
loss total:  0.8067
loss_pos:    0.0159
loss_ang:    0.7908
correct_ang: 0.7778
------------------------------

------------------------------
Epoch 17/29
------------------------------
  (train)
loss total:  0.2755
loss_pos:    0.0103
loss_ang:    0.2652
correct_ang: 1.0000
  (valid)
loss total:  0.7660
loss_pos:    0.0152
loss_ang:    0.7507
correct_ang: 0.8333
------------------------------

------------------------------
Epoch 18/29
------------------------------
  (train)
loss total:  0.2633
loss_pos:    0.0080
loss_ang:    0.2553
correct_ang: 1.0000
  (valid)
loss total:  0.7332
loss_pos:    0.0158
loss_ang:    0.7174
correct_ang: 0.8333
------------------------------

------------------------------
Epoch 19/29
------------------------------
  (train)
loss total:  0.1873
loss_pos:    0.0065
loss_ang:    0.1808
correct_ang: 1.0000
  (valid)
loss total:  0.7080
loss_pos:    0.0175
loss_ang:    0.6905
correct_ang: 0.8333
------------------------------

------------------------------
Epoch 20/29
------------------------------
  (train)
loss total:  0.1604
loss_pos:    0.0090
loss_ang:    0.1513
correct_ang: 1.0000
  (valid)
loss total:  0.6938
loss_pos:    0.0200
loss_ang:    0.6739
correct_ang: 0.8889
------------------------------

------------------------------
Epoch 21/29
------------------------------
  (train)
loss total:  0.1823
loss_pos:    0.0069
loss_ang:    0.1754
correct_ang: 0.9818
  (valid)
loss total:  0.6849
loss_pos:    0.0201
loss_ang:    0.6648
correct_ang: 0.8889
------------------------------

------------------------------
Epoch 22/29
------------------------------
  (train)
loss total:  0.0981
loss_pos:    0.0074
loss_ang:    0.0907
correct_ang: 1.0000
  (valid)
loss total:  0.6703
loss_pos:    0.0178
loss_ang:    0.6525
correct_ang: 0.8889
------------------------------

------------------------------
Epoch 23/29
------------------------------
  (train)
loss total:  0.1233
loss_pos:    0.0075
loss_ang:    0.1158
correct_ang: 0.9818
  (valid)
loss total:  0.6702
loss_pos:    0.0179
loss_ang:    0.6523
correct_ang: 0.8889
------------------------------

------------------------------
Epoch 24/29
------------------------------
  (train)
loss total:  0.0821
loss_pos:    0.0079
loss_ang:    0.0742
correct_ang: 1.0000
  (valid)
loss total:  0.6760
loss_pos:    0.0189
loss_ang:    0.6571
correct_ang: 0.8889
------------------------------

------------------------------
Epoch 25/29
------------------------------
  (train)
loss total:  0.1014
loss_pos:    0.0068
loss_ang:    0.0945
correct_ang: 1.0000
  (valid)
loss total:  0.6633
loss_pos:    0.0181
loss_ang:    0.6451
correct_ang: 0.8889
------------------------------

------------------------------
Epoch 26/29
------------------------------
  (train)
loss total:  0.1366
loss_pos:    0.0068
loss_ang:    0.1298
correct_ang: 0.9636
  (valid)
loss total:  0.6444
loss_pos:    0.0139
loss_ang:    0.6305
correct_ang: 0.8889
------------------------------

------------------------------
Epoch 27/29
------------------------------
  (train)
loss total:  0.0478
loss_pos:    0.0070
loss_ang:    0.0408
correct_ang: 1.0000
  (valid)
loss total:  0.6377
loss_pos:    0.0106
loss_ang:    0.6271
correct_ang: 0.8889
------------------------------

------------------------------
Epoch 28/29
------------------------------
  (train)
loss total:  0.0357
loss_pos:    0.0046
loss_ang:    0.0311
correct_ang: 1.0000
  (valid)
loss total:  0.6467
loss_pos:    0.0102
loss_ang:    0.6364
correct_ang: 0.8889
------------------------------

------------------------------
Epoch 29/29
------------------------------
  (train)
loss total:  0.0397
loss_pos:    0.0070
loss_ang:    0.0327
correct_ang: 1.0000
  (valid)
loss total:  0.6626
loss_pos:    0.0118
loss_ang:    0.6508
correct_ang: 0.8889
------------------------------

Trained in 0m 37s
Best validation epoch total loss:  0.637700
  train:
[1.63446, 1.41804, 1.28079, 1.25994, 1.19091, 1.18159, 1.05284, 1.02449, 0.94713, 0.92412, 0.79997, 0.73252, 0.63081, 0.56731, 0.46964, 0.45073, 0.34947, 0.27553, 0.2633, 0.18729, 0.16039, 0.18227, 0.09808, 0.12332, 0.08205, 0.10136, 0.13665, 0.04775, 0.03571, 0.03971]
  valid:
[1.47955, 1.36303, 1.33313, 1.32483, 1.31625, 1.29745, 1.27339, 1.23649, 1.19035, 1.14071, 1.09975, 1.05628, 1.00791, 0.95355, 0.90277, 0.85289, 0.80673, 0.76596, 0.7332, 0.70802, 0.69383, 0.68486, 0.67026, 0.67021, 0.67603, 0.66328, 0.6444, 0.6377, 0.64666, 0.66257]

Visualizing performance of best model on validation set:
  16 / 18 angle accuracy
```
