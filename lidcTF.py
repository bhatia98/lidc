from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:46:17 2016

@author: kkb15
"""

#=============================================================================

"""Builds the LIDC-U network.
Summary of available functions:
 # Compute input images and labels for training. To run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()
 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)
 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)
 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring

import re
import numpy as np
import tensorflow as tf
from math import ceil
import tfvis as vis

import lidc_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
#tf.app.flags.DEFINE_integer('batch_size', 10,
#                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './tensorflow/',
                           """Path to the TFR data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the LIDC data set.
IMAGE_SIZE = 256
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 963
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 108


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 10.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01      # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'GT970'

filename = "../trfs/trainLungNorm.tfrecords"

"""=================== Helper functions ====================="""
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
                          
def get_weights(name, shape):
    #var = tf.Variable(tf.random_normal(shape), name="weights")
    var = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    tf.add_to_collection(name, var)
    return var

def get_biases(name, size):
    var = tf.constant(0.1, shape=[size])   
#    var = tf.Variable(tf.random_normal([size]), name="biases")
    tf.add_to_collection(name, var)
    return var


def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))



def _deconv_filler(shape):
# create bilinear upsampler
    width = shape[0]
    height = width
    f=ceil(width/2.0)
    c=(2*f-1-f%2)/(2.0*f)
    bilinear = np.zeros([width,height])
    for x in range(width):
        for y in range(height):
            value = (1-abs(x/f-c))*(1-abs(y/f-c))
            bilinear[x,y]=value
    
    nJ=shape[3]
    stdev=np.std(bilinear)    
    np.random.seed(0)
    noise = np.random.normal(0,stdev,shape[2]*nJ)
    
    weights = np.zeros(shape)
    for i in range(shape[2]):
       # for j in range(shape[3]):
       weights[:,:,i,i] = bilinear + noise[i] #noise[nJ*i+j]
    
    init = tf.constant_initializer(value=weights,dtype=tf.float32)
    filler = tf.get_variable(name="upscale",initializer=init,shape=weights.shape)
    
    return filler

""" ======================== Get input batches ================== """
def train_inputs(batchsize, numepochs):
  """For now, simple inputs. Distort for training later.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 2] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
#  data_dir = os.path.join(FLAGS.data_dir, '../tfrs/')
    
  filename_queue = tf.train.string_input_producer([filename], num_epochs=numepochs)
  images, labels = lidc_input.get_input(filename_queue, batchsize)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


""" ======================= Model ================================ """
def inference(images):
  """Build the LIDC model.
  Args:
    images: Images returned from inputs() [Batch_size 256 256 2].
  Returns:
    Logits.
  """
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = get_weights(scope.name, [3,3,2,32])
    biases = get_biases(scope.name, 32)
    conv1 = conv2d(images, kernel, biases)
    _activation_summary(conv1)
    vis.vis_kernel('conv1',kernel)
  pool1 = maxpool2d(conv1,k=2)

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = get_weights(scope.name, [3,3,32,64])
    biases = get_biases(scope.name, 64)
    conv2 = conv2d(pool1, kernel, biases)
    _activation_summary(conv2)
  pool2 = maxpool2d(conv2,k=2)

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = get_weights(scope.name, [3,3,64,128])
    biases = get_biases(scope.name, 128)
    conv3 = conv2d(pool2, kernel, biases)
    _activation_summary(conv3)
  pool3 = maxpool2d(conv3,k=2)
  
  with tf.variable_scope('conv4') as scope:
    kernel = get_weights(scope.name, [3,3,128,2])
    biases = get_biases(scope.name, 2)
    conv4 = conv2d(pool3, kernel, biases)
    _activation_summary(conv4)
    
  # deconv 1
  with tf.variable_scope('deconv1') as scope:
    filter_shape = [16, 16, 2, 2] 
    kernel = _deconv_filler(filter_shape)
    output_shape = [FLAGS.batch_size, 256, 256, 2]             
    deconv1 = tf.nn.conv2d_transpose(conv4, kernel, output_shape, [1, 8, 8, 1])
    _activation_summary(deconv1)   
    #deconv1tmp = tf.transpose(kernel,[0,1,3,2])
    #vis.vis_deconv('deconv1',deconv1tmp)   


  # softmax probabilities  
  with tf.variable_scope('softmax') as scope:
     softmax = tf.nn.softmax(deconv1)
     print('softmax shape:')
     print(softmax.get_shape())
     _activation_summary(softmax)
     tf.image_summary('label0', softmax[:,:,:,0:1])
  
  
  return softmax


""" =========================== Define losses ====================== """
def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss). Just xentropy for now.
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in LIDC model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


""" ======================== Training ============================= """
#def train(total_loss, learning_rate):
def train(total_loss, global_step):
  """Train LIDC model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


