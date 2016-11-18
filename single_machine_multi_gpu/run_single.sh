#!/bin/bash
  #--batch_size 64 \

rm -rf dir/train/*

python translate.py \
  --num_gpus=4 \
  --num_layers 2  --size 200 \
  --data_dir ./dir/data  --train_dir ./dir/train
