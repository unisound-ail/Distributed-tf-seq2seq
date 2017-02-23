#!/bin/bash
export CUDA_VISIBLE_DEVICES=""
#rm -rf ../dir/train/*
#rm -rf ../dir/train0/*
#rm -rf ../dir/train1/*
#rm -rf ../dir/train2/*
JOB_NAME=worker
python translate.py \
  --job_name ${JOB_NAME} --task_index 1 \
  --ps_hosts 172.17.0.6:2221 \
  --worker_hosts 172.17.0.5:2222,172.17.0.8:2222 \
  --batch_size 64 --num_layers 2  --size 200 \
  --data_dir ../dir/dataBk  --train_dir ../dir/train1

#--worker_hosts 10.10.10.39:2222,10.10.10.39:2223 \
