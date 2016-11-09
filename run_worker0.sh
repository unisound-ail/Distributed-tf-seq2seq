#!/bin/bash
JOB_NAME=worker
python translate.py \
  --job_name ${JOB_NAME} --task_index 0 \
  --ps_hosts 10.10.14.71:2221 \
  --worker_hosts 10.10.14.71:2222,10.10.14.71:2223 \
  --num_layers 2  --size 200 \
  --data_dir ./dir/data  --train_dir ./dir/train0
