from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:35:48 2016

@author: kkb15
"""

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import lidcTF

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/lidc/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10,
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

    # Get images and labels for LUNA.
    images, labels = lidcTF.train_inputs(FLAGS.batch_size,FLAGS.num_epochs)
                            
                            
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = lidcTF.inference(images)

    # Calculate loss.
    loss = lidcTF.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = lidcTF.train(loss, 0.02)
    
    
    init = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())
    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    coord = tf.train.Coordinator()
    # Start queue runners
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      step = 0
      while not coord.should_stop():
          print(step)
          start_time = time.time()

          # Run one step of the model.  The return values are
          # the activations from the `train_op` (which is
          # discarded) and the `loss` op.  To inspect the values
          # of your ops or variables, you may include them in
          # the list passed to sess.run() and the value tensors
          # will be returned in the tuple from the call.
          _, loss_value = sess.run([train_op, loss])

          duration = time.time() - start_time

          # Print an overview fairly often.
          if step % 100 == 0:
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                     duration))
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