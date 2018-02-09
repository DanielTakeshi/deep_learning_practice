#!/bin/bash

for e in 0.0003 0.0006 0.0009 0.0012 0.0015 ; do
    for w in 0.0 0.00001 ; do
        for (( i=1 ; i<=20 ; i++ )) ; do
            python mnist_fc.py --optimizer rmsprop --fc_size 400 --lrate $e --l2_reg $w --seed $i > logs/rmsprop-tune/fc-400-lrate-$e-wd-$w-seed-$i
        done
    done
done
