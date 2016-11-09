# Distributed-tensorflow-sequence_to_sequence
A TensorFlow seq2seq model on Kubernetes

## Step 1 tensorflow(cpu) on kubernetes
https://github.com/xuerq/k8s-tensorflow/tree/master/examples/k8s_cpu_tensorflow

## Step 3 tensorflow(cpu) on kubernetes
试验目的：将单机版tensorflow的seq2seq demo改为cpu分布式版本<b>
试验方法：在一台服务器上，创建了2个worker节点，1个ps节点<b>
   sh run_ps.sh: create ps节点<b>
   sh run_worker0.sh: create worker0节点<b>
   sh run_worker1.sh: create worker1节点<b>

试验结果：以下是两个worker节点log，perplexity交替下降，说明参数已经融合<b>
worker0:<b>
step-time 1.61 perplexity 735.61<b>
step-time 1.63 perplexity 130.96<b>
step-time 1.54 perplexity 20.42<b>
step-time 1.63 perplexity 5.40<b>

worker1:<b>
step-time 0.90 perplexity 3042.09<b>
step-time 0.93 perplexity 318.16<b>
step-time 0.90 perplexity 163.88<b>
step-time 0.90 perplexity 73.01<b>
step-time 0.85 perplexity 32.77<b>
step-time 0.87 perplexity 12.99<b>
step-time 0.85 perplexity 6.81<b>
step-time 0.91 perplexity 3.99<b>
step-time 0.85 perplexity 2.70<b>

对比单机gpu log<b>
step-time 0.29 perplexity 1089.04<b>
step-time 0.26 perplexity 220.07<b>
step-time 0.26 perplexity 120.95<b>
step-time 0.27 perplexity 66.51<b>
step-time 0.24 perplexity 26.14<b>
step-time 0.26 perplexity 18.38<b>
step-time 0.26 perplexity 12.17<b>
step-time 0.25 perplexity 6.98<b>
step-time 0.26 perplexity 5.28<b>
step-time 0.26 perplexity 4.62<b>

## Todo:
- [x] Tensorflow(CPU) example on kubernetes
- [ ] Tensorflow(GPU) example on kubernetes
- [x] Seq2seq(CPU) on kubernetes
- [ ] Seq2seq(GPU:one GPU in one pod) on kubernetes
- [ ] Seq2seq(GPU:multi GPUs in one pod) on kubernetes

