
# coding: utf-8

# In[ ]:

from __future__ import division
import numpy as np
import pandas as pd
import os
import glob
from skimage import io, color, measure
from skimage.util import img_as_float, img_as_ubyte
import tensorflow as tf
import time
from six.moves import xrange 
import argparse

# In[ ]:
mean_np_img = None
# In[ ]:

PATCH_DIM = 31
BATCH_SIZE = 100 # Must be a perfect square
NUM_CLASSES = 2
OUT_DIR = os.path.abspath("../../Data/DRIVE/tmp/model1")
IN_DIR = os.path.abspath("../../Data/DRIVE/test")
MODEL_PATH = os.path.abspath("../../Data/models/model1/model.ckpt-7999")
FCHU1 = 256
FORMAT = 'npz'
    
h = int(PATCH_DIM/2)
# We want to access the data chunk by chunk such that each chunk has
# approximately BATCH_SIZE pixels
stride = int(np.sqrt(BATCH_SIZE))

# In[ ]:

def get_path(directory):
    imgs = glob.glob(directory + '/images/*.tif')
    imgs.sort()
    #a = [x.split('/')[-1].split('.')[0] for x in train]
    
    mask = glob.glob(directory + '/mask/*.gif')
    mask.sort()
    #b = [x.split('/')[-1].split('.')[0] for x in mask]
    
    gt = glob.glob(directory + '/1st_manual/*.gif')
    gt.sort()
    #c = [x.split('/')[-1].split('.')[0] for x in gt]
    
    return map(os.path.abspath, imgs), map(os.path.abspath, mask), map(os.path.abspath, gt)

# In[ ]:
def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


# In[ ]:

def inference(images, keep_prob, fc_hidden_units1):
    """ Builds the model as far as is required for running the network
    forward to make predictions.

    Args:
        images: Images placeholder, from inputs().
        keep_prob: Probability used for Dropout in the final Affine Layer
        fc_hidden_units1: Number of hidden neurons in final Affine layer
    Returns:
        softmax_linear: Output tensor with the computed logits.
    """
    with tf.variable_scope('h_conv1') as scope:
        weights = tf.get_variable('weights', shape=[4, 4, 3, 64], 
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.05))
        
        variable_summaries(weights, scope.name+'/weights')
        variable_summaries(biases, scope.name+'/biases')
        
        # Flattening the 3D image into a 1D array
        x_image = tf.reshape(images, [-1,PATCH_DIM,PATCH_DIM,3])
        tf.image_summary('input', x_image, 10)
        
        z = tf.nn.conv2d(x_image, weights, strides=[1, 1, 1, 1], padding='VALID') + biases
        tf.histogram_summary(scope.name + '/pre_activations', z)
        
        h_conv1 = tf.nn.relu(z, name=scope.name)
        tf.histogram_summary(scope.name + '/activations', h_conv1)
        
    with tf.variable_scope('h_conv2') as scope:
        weights = tf.get_variable('weights', shape=[4, 4, 64, 64], 
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.05))
        
        variable_summaries(weights, scope.name+'/weights')
        variable_summaries(biases, scope.name+'/biases')
        
        z = tf.nn.conv2d(h_conv1, weights, strides=[1, 1, 1, 1], padding='SAME')+biases
        tf.histogram_summary(scope.name + '/pre_activations', z)
        
        h_conv2 = tf.nn.relu(z, name=scope.name)
        tf.histogram_summary(scope.name + '/activations', h_conv2)
    
    h_pool1 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name='h_pool1')
    tf.histogram_summary('h_pool1/activations', h_pool1)
    
    with tf.variable_scope('h_conv3') as scope:
        weights = tf.get_variable('weights', shape=[4, 4, 64, 64], 
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.05))
        
        variable_summaries(weights, scope.name+'/weights')
        variable_summaries(biases, scope.name+'/biases')
        
        z = tf.nn.conv2d(h_pool1, weights, strides=[1, 1, 1, 1], padding='SAME')+biases
        tf.histogram_summary(scope.name + '/pre_activations', z)
        
        h_conv3 = tf.nn.relu(z, name=scope.name)
        tf.histogram_summary(scope.name + '/activations', h_conv3)
        
    h_pool2 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name='h_pool2')
    tf.histogram_summary('h_pool2/activations', h_pool2)
    
    
    with tf.variable_scope('h_fc1') as scope:
        weights = tf.get_variable('weights', shape=[7**2*64, fc_hidden_units1], 
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=[fc_hidden_units1], initializer=tf.constant_initializer(0.05))
        
        variable_summaries(weights, scope.name+'/weights')
        variable_summaries(biases, scope.name+'/biases')
        
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights) + biases, name = 'h_fc1')
        tf.histogram_summary(scope.name + '/activations', h_fc1)
        
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        tf.histogram_summary(scope.name + '/dropout_activations', h_fc1_drop)
        
        
    with tf.variable_scope('h_fc2') as scope:
        weights = tf.get_variable('weights', shape=[fc_hidden_units1, NUM_CLASSES], 
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=[NUM_CLASSES])
        
        variable_summaries(weights, scope.name+'/weights')
        variable_summaries(biases, scope.name+'/biases')
        
        logits = (tf.matmul(h_fc1_drop, weights) + biases)
        tf.histogram_summary(scope.name + '/activations', logits)
        
    return logits


# In[ ]:

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    Args:
        batch_size: The batch size will be baked into both placeholders.
    Returns:
        images_placeholder: Images placeholder.
    """
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, PATCH_DIM**2*3))
    return images_placeholder


# In[ ]:

def softmax(logits):
    """ Performs softmax operation on logits
    
        Args:
            logits: logits from inference module
        Output:
            Softmax of logits    
    """
    return tf.nn.softmax(logits)

# In[ ]:

def nbd(image, point):
    """ Finds neighborhood around a point in an image
        
        Args: 
            image: Input image
            point: A point around which we would like to find the neighborhood
        
        Output:
            1d vector of size [PATCH_DIM*PATCH_DIM*3] which is a neighborhood
            aroud the point passed in the parameters list
    """
    i = point[0]
    j = point[1]
    h = int(PATCH_DIM/2)
    return image[i-h:i+h+1,j-h:j+h+1].reshape(-1)


# In[ ]:

def decode(test, mask_test):
    """Segments images in a directory using a given model and saves the images in a particular format
        to an output directory. The ensemble version of the decoder relies on this script to decode
        the same images using different models

        Args:
            test:       Paths to test images
            mask_test:  Paths to corresponding masks
    """

    begin = time.time()
    with tf.Graph().as_default():
        # Generate placeholders for the images and  keep_probability
        images_placeholder = placeholder_inputs(BATCH_SIZE)
        keep_prob = tf.placeholder(tf.float32)


        # Build a Graph that computes predictions from the inference model.
        logits = inference(images_placeholder, keep_prob, FCHU1)
        sm = softmax(logits)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        with tf.Session() as sess:
            saver.restore(sess, MODEL_PATH)
            
            # Once the model has been restored, we iterate through all images in the test set
            for im_no in xrange(len(test)):

                start_time = time.time()
                print "Working on image %d" % (im_no+1)
                image = io.imread(test[im_no])
                mask = img_as_float(io.imread(mask_test[im_no]))
                
                # We will start with a completely black image and update it chunk by chunk
                segmented = np.zeros(image.shape[:2])

                # We will use arrays to index the image and mask later
                cols, rows = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
                row_col = np.stack([rows,cols], axis = 2)
                # The neighborhood windows to be fed into the graph
                feed = np.zeros((BATCH_SIZE, PATCH_DIM**2*3))
                # The predicted classes for all the windows that were fed to the graph
                predictions = np.zeros(BATCH_SIZE)

                pixel_count = 0
            
                
                i = h+1
                while i < image.shape[0] - h-2:
                    j = h+1
                    while j < image.shape[1] - h-1:
                        # A small check is made to ensure that not all pixels are black

                        # Update i and j by adding stride but take care near the end
                        i_next = min(i+stride, image.shape[0]-h-1)
                        j_next = min(j+stride, image.shape[1]-h-1)
                        
                        if int(np.max(mask[i:i_next,j:j_next])) == 1:

                            pixel_count += BATCH_SIZE # This will not be true for border cases though
                                                      # but we don't care about the progress at the end

                            # Once we get a chunk, we flatten it and map a function that returns 
                            # the neighborhood of each point

                            #feed = np.array(map(lambda p: nbd(image, p), row_col[i:i_next, j:j_next].reshape(-1,2)))
                            #print " Feed shape = (%d, %d)" % feed.shape
                            chunk = np.array(map(lambda p: nbd(image, p), row_col[i:i_next, j:j_next].reshape(-1,2)))
                            feed[:len(chunk)] = chunk
                            
                            # Subtract training mean image
                            feed = feed - mean_np_img

                            # Get predictions and draw accordingly on black image    
                            predictions = sess.run([sm],
                                           feed_dict={images_placeholder: feed,
                                                      keep_prob: 1.0})
                            predictions = np.asarray(predictions).reshape(BATCH_SIZE, NUM_CLASSES)

                            # Uncomment following line for non-probability plotting
                            #predictions = np.argmax(predictions, axis=1)
                            predictions = predictions[:,1]

                            if not len(chunk) == BATCH_SIZE:
                                predictions = predictions[:len(chunk)]
                            segmented[rows[i:i_next, j:j_next], cols[i:i_next, j:j_next]] = predictions.reshape(i_next-i, j_next-j)

                            # Reset everything after passing feed to feedforward
                            feed = np.zeros((BATCH_SIZE, PATCH_DIM**2*3))
                            predictions = np.zeros(BATCH_SIZE)

                        j += stride
                    i += stride
                segmented = np.multiply(segmented,mask)
                    
                name = test[im_no].split('/')[-1].split('.')[0]
                if FORMAT == 'npz':
                    np.savez(os.path.join(OUT_DIR, name+'.npz'), segmented)
                elif FORMAT == 'jpg' or FORMAT == 'png':
                    segmented = segmented * (1.0/segmented.max())
                    io.imsave(os.path.join(OUT_DIR, name+'.'+FORMAT), segmented)
                else:
                    print "Unknown format. Saving as png."
                    segmented = segmented * (1.0/segmented.max())   
                    io.imsave(os.path.join(OUT_DIR, name+'.png'), segmented)

                current_time = time.time()
                print "Time taken - > %f" % (current_time - start_time)
                start_time = current_time

    print "Total time = %f mins" % ((time.time()-begin)/60.0)

def finish_parsing():

    global OUT_DIR, IN_DIR, MODEL_PATH, FCHU1, FORMAT

    parser = argparse.ArgumentParser(description=
                                     'Script to decode images using a single model')
    parser.add_argument("--fchu1", type=int, 
                help="Number of hidden units in FC1 layer. This should be identical to the one used in the model [Default - 256]")

    parser.add_argument("--out",
                        help="Directory to put rendered images to")
    parser.add_argument("--inp",
                        help="Directory containing images for testing")
    parser.add_argument("--model",
                        help="Path to the saved tensorflow model checkpoint")
    parser.add_argument("--format",
                        help="Format to save the images in. [Available formats: npz, jpg and png]")

    args = parser.parse_args()

    if args.fchu1 is not None:
        FCHU1 = args.fchu1
        print "New FCHU1 = %d" % FCHU1
    if args.out is not None:
        OUT_DIR = args.out
        print "New OUT_DIR = %s" % OUT_DIR
    if args.inp is not None:
        IN_DIR = args.inp
        print "New IN_DIR = %s" % IN_DIR
    if args.model is not None:
        MODEL_PATH = args.model
        print "New MODEL_PATH = %s" % MODEL_PATH
    if args.format is not None:
        FORMAT = args.format
        print "New FORMAT = %s" % FORMAT

def main():
    finish_parsing()    

    global mean_np_img

    mean_img = pd.read_pickle('../../Data/mean_img_no_class_bias.pkl')
    mean_np_img = np.asarray(mean_img)

    test, mask_test, _ = get_path(IN_DIR)

    # Make a directory to store the new images in
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    decode(test, mask_test)



if __name__ == "__main__":
    main()
