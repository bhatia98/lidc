# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:27:28 2016

@author: kkb15
"""

# File to read TRFs and display images (for instance, images and segmentations)

import tensorflow as tf
#import matplotlib.pyplot as plt

filename = "../trfs/valLungNorm.tfrecords"

def read_files():
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        # traverse the Example format to get data
        width = example.features.feature['width'].int64_list.value[0]
        height = example.features.feature['height'].int64_list.value[0]
        depth = example.features.feature['depth'].int64_list.value[0]
        image = example.features.feature['image_raw'].bytes_list.value[0]
        label = example.features.feature['label'].bytes_list.value[0]

        im = tf.decode_raw(image,tf.float32)
        lbl = tf.decode_raw(label,tf.uint8)  
        image = tf.reshape(im,[depth,height,width])
        label = tf.reshape(lbl,[height,width])
        # convert from [depth,height,width] to [height,width,depth]
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
    

#with tf.Session() as sess: 

def get_input():
    min_queue_examples = 500    
    batch_size = 10
    # initialise variables
    i, l = read_files()
    a, b = generate_batch(i,l,min_queue_examples,batch_size)
    return a,b
    #image = sess.run(i) # convert to numpy array (could also use eval()?)
    #label = sess.run(l) # convert to numpy array
    #plt.imshow(image[:,:,0]) # plot the CT image [1,:,:] for MIP
    #plt.imshow(label)
