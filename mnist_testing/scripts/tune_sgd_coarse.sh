#!/bin/bash
for e in 0.04 0.07 0.1 0.3 0.5 0.7; do
    for w in 0.0 0.000001 0.00001 0.0001 0.001; do
        for (( i=1 ; i <= 4 ; i++ )); do
            python mnist_fc.py --optimizer sgd --fc_size 400 --lrate $e --l2_reg $w --seed $i > logs/sgd-tune/fc-400-lrate-$e-wd-$w-seed-$i
        done
    done
done
