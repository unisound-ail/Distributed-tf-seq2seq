# TFCluster：编写 tensorflow 分布式程序的易用性封装
                        
## Step 1 编写 tensorflow 分布式程序时，import TFCluster.py

## Step 2 修改单机训练代码基础上，增加如下几行
```python
  tfd = TFCluster.TF_Dist()
  if FLAGS.job_name != "worker":
    return
  with tf.device(tfd.Get_replica_device_setter()):
    print("Creating %d layers of %d units On task %d." % 
          (FLAGS.num_layers, FLAGS.size,FLAGS.task_index))
    model = create_model_distributed(False)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    saver = tf.train.Saver()
  sv=tfd.Get_Supervisor(logdir=FLAGS.train_dir,
                        saver=saver,
                        global_step=global_step)
  with sv.managed_session(tfd.server.target) as sess:
```
## Step 3 查看单机和分布式代码的例子
translate.py
其中:
train_single():为单机代码
train():为分布式代码
