# Distributed-tensorflow-sequence_to_sequence
A TensorFlow seq2seq model on Kubernetes

## Step 1 tensorflow(cpu) on kubernetes
https://github.com/xuerq/k8s-tensorflow/tree/master/examples/k8s_cpu_tensorflow

## Step 3 Seq2seq(CPU) on kubernetes
试验目的：将单机版tensorflow的seq2seq demo改为cpu分布式版本<br>
试验方法：在一台服务器上，创建了2个worker节点，1个ps节点<br>
   sh run_ps.sh: create ps节点<br>
   sh run_worker0.sh: create worker0节点<br>
   sh run_worker1.sh: create worker1节点<br>

试验结果：以下是两个worker节点log，perplexity交替下降，说明参数已经融合<br>
worker0:<br>
step-time 1.61 perplexity 735.61<br>
step-time 1.63 perplexity 130.96<br>
step-time 1.54 perplexity 20.42<br>
step-time 1.63 perplexity 5.40<br>

worker1:<br>
step-time 0.90 perplexity 3042.09<br>
step-time 0.93 perplexity 318.16<br>
step-time 0.90 perplexity 163.88<br>
step-time 0.90 perplexity 73.01<br>
step-time 0.85 perplexity 32.77<br>
step-time 0.87 perplexity 12.99<br>
step-time 0.85 perplexity 6.81<br>
step-time 0.91 perplexity 3.99<br>
step-time 0.85 perplexity 2.70<br>

对比单机gpu log<br>
step-time 0.29 perplexity 1089.04<br>
step-time 0.26 perplexity 220.07<br>
step-time 0.26 perplexity 120.95<br>
step-time 0.27 perplexity 66.51<br>
step-time 0.24 perplexity 26.14<br>
step-time 0.26 perplexity 18.38<br>
step-time 0.26 perplexity 12.17<br>
step-time 0.25 perplexity 6.98<br>
step-time 0.26 perplexity 5.28<br>
step-time 0.26 perplexity 4.62<br>

### add run seq2seq on k8s script(update 2016-11-12)<br>
1.delete mession<br>
  chmod +x delete.sh;./delete.sh<br>
2.run mession<br>
  chmod +x create.sh;./create.sh<br>

## Step 4 Seq2seq(GPU:one GPU in one pod) on kubernetes
update 2016-11-14<br>

物理节点(GPU)1： ps节点、worker节点1<br>
物理节点(GPU)2： worker节点2<br>

kubectl create -f worker_ps_GPU.yaml<br>

结论：<br>
1. seq2seq训练在gpu集群跑通<br>
2. 在一台物理节点上，目前只能启动一个gpu<br>

## Step 5 Seq2seq(GPU:multi GPUs) on physical machine
see ./single_machine_multi_gpu

## 性能测试
测试环境：分布式4容器（单机4容器），1ps 3worker节点<br>
性能结论：<br>
1. 分布式2显卡≈单机1显卡<br>
2. 分布式3显卡>单机1显卡<br>
         
### 单机1显卡
```shell
global step 200 learning rate 0.5000 step-time 0.32 perplexity 1427.92
  eval: bucket 0 perplexity 3541.55
  eval: bucket 1 perplexity 3248.93
  eval: bucket 2 perplexity 3501.10
  eval: bucket 3 perplexity 4820.27
global step 400 learning rate 0.5000 step-time 0.29 perplexity 218.10
  eval: bucket 0 perplexity 1791.22
  eval: bucket 1 perplexity 1978.34
  eval: bucket 2 perplexity 1763.81
  eval: bucket 3 perplexity 1777.52
global step 600 learning rate 0.5000 step-time 0.29 perplexity 114.04
  eval: bucket 0 perplexity 3294.74
  eval: bucket 1 perplexity 2020.29
  eval: bucket 2 perplexity 2141.23
  eval: bucket 3 perplexity 2458.65
global step 800 learning rate 0.5000 step-time 0.29 perplexity 62.45
  eval: bucket 0 perplexity 3397.24
  eval: bucket 1 perplexity 2761.49
  eval: bucket 2 perplexity 2629.58
  eval: bucket 3 perplexity 3044.73
global step 1000 learning rate 0.5000 step-time 0.29 perplexity 27.37         
```
###分布式3显卡
```shell
Gpu1
current step 200  step-time 0.66 sample-per-sec 97.15 perplexity 555.52
  eval: bucket 0 perplexity 4363.28
  eval: bucket 1 perplexity 2138.63
  eval: bucket 2 perplexity 2598.28
  eval: bucket 3 perplexity 1529.38
current step 400  step-time 0.53 sample-per-sec 120.66 perplexity 74.70
  eval: bucket 0 perplexity 3186.36
  eval: bucket 1 perplexity 3427.06
  eval: bucket 2 perplexity 1853.89
  eval: bucket 3 perplexity 1372.51
current step 600  step-time 0.52 sample-per-sec 123.49 perplexity 14.05
  eval: bucket 0 perplexity 6908.31
  eval: bucket 1 perplexity 10990.13
  eval: bucket 2 perplexity 6280.96
  eval: bucket 3 perplexity 2736.41
current step 800  step-time 0.53 sample-per-sec 121.09 perplexity 4.52
  eval: bucket 0 perplexity 96858.06
  eval: bucket 1 perplexity 120185.61
  eval: bucket 2 perplexity 44755.82
  eval: bucket 3 perplexity 26741.13
current step 1000  step-time 0.55 sample-per-sec 116.73 perplexity 1.68
  eval: bucket 0 perplexity 319729.32
  eval: bucket 1 perplexity 1067919.29
  eval: bucket 2 perplexity 261708.87
  eval: bucket 3 perplexity 147156.36
current step 1200  step-time 0.54 sample-per-sec 119.58 perplexity 2.48
  eval: bucket 0 perplexity 475635.35
  eval: bucket 1 perplexity 629923.40
  eval: bucket 2 perplexity 273512.36
  eval: bucket 3 perplexity 92860.20

Gpu2
current step 200  step-time 0.55 sample-per-sec 116.94 perplexity 8349.74
  eval: bucket 0 perplexity 2443.89
  eval: bucket 1 perplexity 1324.94
  eval: bucket 2 perplexity 1466.47
  eval: bucket 3 perplexity 1528.30
current step 400  step-time 0.53 sample-per-sec 121.12 perplexity 161.81
  eval: bucket 0 perplexity 1463.04
  eval: bucket 1 perplexity 1481.88
  eval: bucket 2 perplexity 1517.51
  eval: bucket 3 perplexity 1372.51
current step 600  step-time 0.50 sample-per-sec 126.95 perplexity 34.97
  eval: bucket 0 perplexity 2823.54
  eval: bucket 1 perplexity 4039.98
  eval: bucket 2 perplexity 2272.58
  eval: bucket 3 perplexity 1471.31
current step 800  step-time 0.50 sample-per-sec 127.64 perplexity 6.58
  eval: bucket 0 perplexity 21334.52
  eval: bucket 1 perplexity 31563.55
  eval: bucket 2 perplexity 13883.32
  eval: bucket 3 perplexity 5732.98
current step 1000  step-time 0.53 sample-per-sec 119.63 perplexity 3.36
  eval: bucket 0 perplexity 136168.92
  eval: bucket 1 perplexity 62709.63
  eval: bucket 2 perplexity 52735.24
  eval: bucket 3 perplexity 25050.97
current step 1200  step-time 0.54 sample-per-sec 119.55 perplexity 1.88
  eval: bucket 0 perplexity 162081.00
  eval: bucket 1 perplexity 501147.01
  eval: bucket 2 perplexity 183193.38
  eval: bucket 3 perplexity 148799.01
current step 1400  step-time 0.54 sample-per-sec 119.24 perplexity 2.42
  eval: bucket 0 perplexity 345981.59
  eval: bucket 1 perplexity 483963.88
  eval: bucket 2 perplexity 121277.70
  eval: bucket 3 perplexity 85902.29

Gpu3
current step 200  step-time 0.56 sample-per-sec 114.60 perplexity 3389.84
  eval: bucket 0 perplexity 1839.04
  eval: bucket 1 perplexity 2036.01
  eval: bucket 2 perplexity 1235.35
  eval: bucket 3 perplexity 2096.51
current step 400  step-time 0.52 sample-per-sec 122.57 perplexity 110.73
  eval: bucket 0 perplexity 3049.79
  eval: bucket 1 perplexity 3545.79
  eval: bucket 2 perplexity 2537.56
  eval: bucket 3 perplexity 1533.19
current step 600  step-time 0.51 sample-per-sec 124.89 perplexity 26.22
  eval: bucket 0 perplexity 5451.81
  eval: bucket 1 perplexity 8315.88
  eval: bucket 2 perplexity 2801.59
  eval: bucket 3 perplexity 2611.64
current step 800  step-time 0.52 sample-per-sec 123.09 perplexity 5.49
  eval: bucket 0 perplexity 14110.71
  eval: bucket 1 perplexity 53850.02
  eval: bucket 2 perplexity 32003.12
  eval: bucket 3 perplexity 6723.97
current step 1000  step-time 0.52 sample-per-sec 124.03 perplexity 2.62
  eval: bucket 0 perplexity 103068.39
  eval: bucket 1 perplexity 104944.29
  eval: bucket 2 perplexity 57259.02
  eval: bucket 3 perplexity 29859.50
current step 1200  step-time 0.54 sample-per-sec 118.26 perplexity 1.64
  eval: bucket 0 perplexity 227892.63
  eval: bucket 1 perplexity 551796.12
  eval: bucket 2 perplexity 240696.85
  eval: bucket 3 perplexity 147526.06
current step 1400  step-time 0.53 sample-per-sec 119.76 perplexity 2.45
  eval: bucket 0 perplexity 2051196.09
  eval: bucket 1 perplexity 1152949.78
  eval: bucket 2 perplexity 258289.87
  eval: bucket 3 perplexity 175813.99
```

## Todo:
- [x] 1. Tensorflow(CPU) example on kubernetes
- [x] 2. Tensorflow(GPU) example on kubernetes
- [x] 3. Seq2seq(CPU) on kubernetes
- [x] 4. Seq2seq(GPU:one GPU in one pod) on kubernetes
- [x] 5. Seq2seq(GPU:multi GPUs) on physical machine
- [ ] 6. Seq2seq(GPU:multi GPUs in one pod) on kubernetes

