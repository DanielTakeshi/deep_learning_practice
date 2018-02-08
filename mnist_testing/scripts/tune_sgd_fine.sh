#!/bin/bash
# TODO similar to RMSProp...
for (( i=1 ; i <= 20 ; i++ )); do
    python mnist_fc.py --optimizer sgd --fc_size 400 --lrate $e --l2_reg $w --seed $i > logs/sgd-tune/fc-400-lrate-$e-wd-$w-seed-$i
done
