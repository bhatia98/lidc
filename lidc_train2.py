from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:35:48 2016
@author: kkb15
"""
import os.path
import time
import tensorflow as tf
import lidcTF


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/lidc/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Batch size.""")                         
tf.app.flags.DEFINE_integer('num_epochs', 1,
                            """Number of epochs.""")                                     
tf.app.flags.DEFINE_boolean('log_device_placement', False,
"""Whether to log device placement.""")



def train():
  """Train LIDC for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for LUNA.
    images, labels = lidcTF.train_inputs(FLAGS.batch_size,FLAGS.num_epochs)
                                                        
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = lidcTF.inference(images)

    # Calculate loss.
    loss = lidcTF.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = lidcTF.train(loss, global_step)
    
    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()
      
    # Initialize variables  
    init = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())
                       
    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    coord = tf.train.Coordinator()
    
    # Start queue runners
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph) 

    try:
      step = 0
      while not coord.should_stop():
          
          start_time = time.time()

          # Run one step of the model.  The return values are
          # the activations from the `train_op` (which is
          # discarded) and the `loss` op.  To inspect the values
          # of your ops or variables, include them in
          # the list passed to sess.run() and the value tensors
          # will be returned in the tuple from the call.
          _, loss_value = sess.run([train_op, loss])

          # print overview
          duration = time.time() - start_time
          if step % 100 == 0:
              num_examples_per_step = FLAGS.batch_size
              examples_per_sec = num_examples_per_step / duration
              sec_per_batch = float(duration)
              summary_str = sess.run(summary_op)
              summary_writer.add_summary(summary_str, step)
              format_str = ('Step %s: loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
              print (format_str % (step, loss_value,
                                   examples_per_sec, sec_per_batch))

          # Save the model checkpoint periodically.
          if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
              checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
              saver.save(sess, checkpoint_path, global_step=step)                                                 
                                                                                                       
          step += 1
          
    except tf.errors.OutOfRangeError:
      print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

      # Wait for threads to finish.
      coord.join(threads)
      sess.close()


def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
    tf.app.run()