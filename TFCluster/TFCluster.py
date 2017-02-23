from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

class TF_Dist(object):
  def __init__(self):

    self.ps_hosts = FLAGS.ps_hosts.split(",")
    self.worker_hosts = FLAGS.worker_hosts.split(",")
    self.cluster = tf.train.ClusterSpec({"ps": self.ps_hosts, "worker": self.worker_hosts})
    self.server = tf.train.Server(self.cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        self.server.join()
  def Get_replica_device_setter(self):
    replica_device_setter=tf.train.replica_device_setter(
						worker_device="/job:worker/task:%d" % FLAGS.task_index,
						cluster=self.cluster)
    return replica_device_setter;
  def Run_ps(self):
    if FLAGS.job_name == "ps":
        self.server.join()
  def Get_Supervisor(self,
                graph=None,
                ready_op=0,
                ready_for_local_init_op=0,
                #is_chief=True,
                #init_op=0,
                init_feed_dict=None,
                local_init_op=0,
                logdir=None,
                summary_op=0,
                saver=0,
                global_step=0,
                save_summaries_secs=120,
                save_model_secs=600,
                recovery_wait_secs=30,
                stop_grace_secs=120,
                checkpoint_basename="model.ckpt",
                session_manager=None,
                summary_writer=0,
                init_fn=None):
    sv =tf.train.Supervisor(
                graph=graph,
                ready_op=ready_op,
                ready_for_local_init_op=ready_for_local_init_op,
                is_chief=(FLAGS.task_index == 0),
                init_op=tf.initialize_all_variables(),
                init_feed_dict=init_feed_dict,
                local_init_op=local_init_op,
                logdir=logdir,
                summary_op=summary_op,
                saver=saver,
                global_step=global_step,
                save_summaries_secs=save_summaries_secs,
                save_model_secs=save_model_secs,
                recovery_wait_secs=recovery_wait_secs,
                stop_grace_secs=stop_grace_secs,
                checkpoint_basename=checkpoint_basename,
                session_manager=session_manager,
                summary_writer=summary_writer,
                init_fn=init_fn)
    if (FLAGS.task_index == 0):
      print("Worker %d: Initializing session..." % FLAGS.task_index)
    else:
      print("Worker %d: Waiting for session to be initialized..." %
            FLAGS.task_index)
    #sess=sv.managed_session(self.server.target);
    return sv
