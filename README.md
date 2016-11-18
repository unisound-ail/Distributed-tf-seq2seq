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

update 2016-11-12, add run seq2seq on k8s script<br>
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

## Todo:
- [x] 1. Tensorflow(CPU) example on kubernetes
- [x] 2. Tensorflow(GPU) example on kubernetes
- [x] 3. Seq2seq(CPU) on kubernetes
- [x] 4. Seq2seq(GPU:one GPU in one pod) on kubernetes
- [x] 5. Seq2seq(GPU:multi GPUs) on physical machine
- [ ] 6. Seq2seq(GPU:multi GPUs in one pod) on kubernetes

