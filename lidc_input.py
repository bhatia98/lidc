# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:27:28 2016

@author: kkb15
"""

# File to read TRFs and display images (for instance, images and segmentations)
import tensorflow as tf


def read_files(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
    })

    im = tf.decode_raw(features['image_raw'],tf.float32)
    lbl = tf.decode_raw(features['label'],tf.uint8)
    
    image = tf.reshape(im,[2,256,256])
    label = tf.reshape(lbl,[256,256])
#    # convert from [depth,height,width] to [height,width,depth]
    image = tf.transpose(image,[1,2,0])   
    
    return image, label

    
def generate_batch(image, label, min_queue_ex, batchsize):
    # returns shuffled batch of labels and images
    cap = min_queue_ex + 3 * batchsize # recommended
    images, labels = tf.train.shuffle_batch([image,label], 
                                            batch_size=batchsize,
                                            capacity=cap,
                                            min_after_dequeue=min_queue_ex)
                        
    tf.image_summary('images',images)
    return images, labels
    

def get_input(filename_q, batch_size):
    min_queue_examples = 400    
    #batch_size = 10
    # initialise variables
    i, l = read_files(filename_q)
    a, b = generate_batch(i,l,min_queue_examples,batch_size)
    return a,b
