# single_machine_multi_gpu
本实验将tensorflow的seq2seq demo改为单机多卡运行<br>
首先说明下本实验目的：<br>
1. 证明了，在不改变模型（单机单卡）定义的graph的前提下，可以将模型转变为多卡训练<br>
2. 证明了，单卡转为多卡训练后，训练效率是提升的<br>

## 运行方法：<br>

sh run_single.sh<br>
运行结果<br>
```
Fri Nov 18 03:28:46 2016
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 367.44                 Driver Version: 367.44                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 980     Off  | 0000:04:00.0     Off |                  N/A |
| 26%   41C    P2    64W / 180W |   3782MiB /  4037MiB |     40%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 980     Off  | 0000:05:00.0     Off |                  N/A |
| 26%   38C    P2    65W / 180W |   3782MiB /  4037MiB |     13%      Default |
+-------------------------------+----------------------+----------------------+
|   2  GeForce GTX 980     Off  | 0000:08:00.0     Off |                  N/A |
| 26%   37C    P2    67W / 180W |   3782MiB /  4037MiB |     39%      Default |
+-------------------------------+----------------------+----------------------+
|   3  GeForce GTX 980     Off  | 0000:09:00.0     Off |                  N/A |
| 26%   36C    P2    65W / 180W |   3782MiB /  4037MiB |     45%      Default |
+-------------------------------+----------------------+----------------------+
```
效率提升
4gpu : 
```
global step 200 learning rate 0.5000 step-time 0.33 sample-per-sec 782.24 perplexity 688.39
   eval: bucket 0 perplexity 279.72
  eval: bucket 1 perplexity 235.40
  eval: bucket 2 perplexity 261.60
  eval: bucket 3 perplexity 617.31
global step 400 learning rate 0.5000 step-time 0.32 -sampleper-sec 801.45 perplexity 425.08
  eval: bucket 0 perplexity 242.39
  eval: bucket 1 perplexity 262.32
  eval: bucket 2 perplexity 251.14
  eval: bucket 3 perplexity 275.40
global step 600 learning rate 0.5000 step-time 0.39 sample-per-sec 656.86 perplexity 306.08
  eval: bucket 0 perplexity 206.55
  eval: bucket 1 perplexity 233.25
  eval: bucket 2 perplexity 279.67
  eval: bucket 3 perplexity 297.41
global step 800 learning rate 0.5000 step-time 0.27 sample-per-sec 941.30 perplexity 278.51
  eval: bucket 0 perplexity 186.01
  eval: bucket 1 perplexity 244.04
  eval: bucket 2 perplexity 217.89
  eval: bucket 3 perplexity 233.93
global step 1000 learning rate 0.5000 step-time 0.27 sample-per-sec 936.27 perplexity 264.25
```
1gpu : 
```
global step 200 learning rate 0.5000 step-time 0.30 sample-per-sec 211.23 perplexity 1032.33
  eval: bucket 0 perplexity 225.31
  eval: bucket 1 perplexity 309.99
  eval: bucket 2 perplexity 327.17
  eval: bucket 3 perplexity 320.64
global step 400 learning rate 0.5000 step-time 0.25 sample-per-sec 252.13 perplexity 391.96
  eval: bucket 0 perplexity 111.15
  eval: bucket 1 perplexity 206.99
  eval: bucket 2 perplexity 237.04
  eval: bucket 3 perplexity 280.60
global step 600 learning rate 0.5000 step-time 0.26 sample-per-sec 243.87 perplexity 288.47
  eval: bucket 0 perplexity 79.37
  eval: bucket 1 perplexity 187.94
  eval: bucket 2 perplexity 199.98
  eval: bucket 3 perplexity 251.38
global step 800 learning rate 0.5000 step-time 0.28 sample-per-sec 231.45 perplexity 255.71
  eval: bucket 0 perplexity 111.10
  eval: bucket 1 perplexity 158.40
  eval: bucket 2 perplexity 198.32
  eval: bucket 3 perplexity 213.16
global step 1000 learning rate 0.5000 step-time 0.26 sample-per-sec 244.82 perplexity 269.83
  eval: bucket 0 perplexity 92.54
  eval: bucket 1 perplexity 156.65
  eval: bucket 2 perplexity 167.31
  eval: bucket 3 perplexity 193.36
global step 1200 learning rate 0.5000 step-time 0.26 sample-per-sec 242.49 perplexity 210.17
  eval: bucket 0 perplexity 1336.21
  eval: bucket 1 perplexity 496.46
  eval: bucket 2 perplexity 387.85
  eval: bucket 3 perplexity 293.78
global step 1400 learning rate 0.5000 step-time 0.26 sample-per-sec 244.88 perplexity 235.90

```

## 如何将单卡的代码改为多卡
### 为gpu单独创建model,注意这里并不改变单卡model的定义
```python
for i in xrange(FLAGS.num_gpus):
  with tf.device('/gpu:%d' % i):
    with tf.name_scope('TOWER_%d' % (i)) as scope:
      # Create model.
      print("Creating %d layers of %d units On Gpu:%d." % (FLAGS.num_layers, FLAGS.size, i))
      model_list[i] = create_model2(sess, False)
```
### 收集每个gpu单独计算的loss和梯度
```python
      step_losses.append(model_list[i].losses[bucket_id])
      gradient_norms = []
      params = tf.trainable_variables()
      gradients = tf.gradients(step_losses[i], params)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                    FLAGS.max_gradient_norm)
      gradient_norms.append(norm)
      # Keep track of the gradients across all towers.
      tower_grads.append(clipped_gradients)
```
### 完整代码
```python
for i in xrange(FLAGS.num_gpus):
  with tf.device('/gpu:%d' % i):
    with tf.name_scope('TOWER_%d' % (i)) as scope:
      # Create model.
      print("Creating %d layers of %d units On Gpu:%d." % (FLAGS.num_layers, FLAGS.size, i))
      model_list[i] = create_model2(sess, False)

      step_losses.append(model_list[i].losses[bucket_id])
      # Reuse variables for the next tower.
      tf.get_variable_scope().reuse_variables()

      gradient_norms = []
      params = tf.trainable_variables()
      gradients = tf.gradients(step_losses[i], params)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                    FLAGS.max_gradient_norm)
      gradient_norms.append(norm)
      # Keep track of the gradients across all towers.
      tower_grads.append(clipped_gradients)
```
### 训练部分，多卡独立计算，所以我们需要为每个gpu分别提供input
```python
while True:
  input_feed = {}
  for i in xrange(FLAGS.num_gpus):
    # Check if the sizes match.
    encoder_size, decoder_size = model_list[i].buckets[bucket_id]
    encoder_inputs[i], decoder_inputs[i], target_weights[i] = model_list[i].get_batch(train_set, bucket_id)

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    for l in xrange(encoder_size):
      input_feed[model_list[i].encoder_inputs[l].name] = encoder_inputs[i][l]
    for l in xrange(decoder_size):
      input_feed[model_list[i].decoder_inputs[l].name] = decoder_inputs[i][l]
      input_feed[model_list[i].target_weights[l].name] = target_weights[i][l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = model_list[i].decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([model_list[i].batch_size], dtype=np.int32)

  #for GpuIdx in xrange(FLAGS.num_gpus):
  current_step += 1
  # Gradients and SGD update operation for training the model.
  #params = tf.trainable_variables()
  start_time = time.time()
  _,step_loss = sess.run([train_op,step_losses],input_feed)
  step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
  loss += step_loss[0] / FLAGS.steps_per_checkpoint
```

### 注意1,allow_soft_placement参数必须设置，防止某些op无法在gpu上运行：
```python
# Start running operations on the Graph. allow_soft_placement must be set to  
# True to build towers on GPU, as some of the ops do not have GPU  
# implementations.  

config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
```

### 注意2,tf.get_variable_scope().reuse_variables()必须设置，保证不同gpu的模型间可以共享变量：
如果去掉会raise exception：变量重复定义
```python
Traceback (most recent call last):
  File "translate.py", line 417, in <module>
    tf.app.run()
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/platform/app.py", line 30, in run
    sys.exit(main(sys.argv))
  File "translate.py", line 414, in main
    train()
  File "translate.py", line 233, in train
    model_list[i] = create_model2(sess, False)
  File "translate.py", line 141, in create_model2
    forward_only=forward_only)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/models/rnn/translate/seq2seq_model.py", line 86, in __init__
    w = tf.get_variable("proj_w", [size, self.target_vocab_size])
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/variable_scope.py", line 873, in get_variable
    custom_getter=custom_getter)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/variable_scope.py", line 700, in get_variable
    custom_getter=custom_getter)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/variable_scope.py", line 217, in get_variable
    validate_shape=validate_shape)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/variable_scope.py", line 202, in _true_getter
    caching_device=caching_device, validate_shape=validate_shape)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/variable_scope.py", line 494, in _get_single_variable
    name, "".join(traceback.format_list(tb))))
ValueError: Variable proj_w already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:

  File "/usr/local/lib/python2.7/dist-packages/tensorflow/models/rnn/translate/seq2seq_model.py", line 86, in __init__
    w = tf.get_variable("proj_w", [size, self.target_vocab_size])
  File "translate.py", line 141, in create_model2
    forward_only=forward_only)
  File "translate.py", line 233, in train
    model_list[i] = create_model2(sess, False)
```
关于共享变量、variable_scope 以及 name_scope的作用和区别<br>
http://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow<br>
http://stackoverflow.com/questions/34215746/difference-between-variable-scope-and-name-scope-in-tensorflow<br>
