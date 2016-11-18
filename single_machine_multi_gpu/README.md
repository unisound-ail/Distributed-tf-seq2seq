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
```
4gpu : sample-per-sec 532.69<br>
1gpu : sample-per-sec 241.41<br>
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
