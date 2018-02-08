#!/bin/bash
# TODO find best settings and run with more random seeds.
for (( i=1 ; i <= 20 ; i++ )); do
    python mnist_fc.py --optimizer rmsprop --fc_size 400 --lrate $e --l2_reg $w --seed $i > logs/rmsprop-tune/fc-400-lrate-$e-wd-$w-seed-$i
done
