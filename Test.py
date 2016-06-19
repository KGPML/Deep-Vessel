
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


# In[ ]:

mean_img = pd.read_pickle('../Data/mean_img_no_class_bias.pkl')


# In[ ]:

PATCH_DIM = 31
BATCH_SIZE = 100 # Must be a perfect square
NUM_CLASSES = 2


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

train, mask_train, gt_train =  get_path('../Data/DRIVE/training')
test, mask_test, gt_test = get_path('../Data/DRIVE/test')


# In[ ]:

def inference(images, keep_prob, fc_hidden_units1=512):
    """ Builds the model as far as is required for running the network
    forward to make predictions.

    Args:
        images: Images placeholder, from inputs().
        keep_prob: Probability used for Droupout in the final Affine Layer
        fc_hidden_units1: Number of hidden neurons in final Affine layer
    Returns:
        softmax_linear: Output tensor with the computed logits.
    """
    with tf.variable_scope('h_conv1') as scope:
        weights = tf.get_variable('weights', shape=[4, 4, 3, 64], 
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.05))
        
        # Flattening the 3D image into a 1D array
        x_image = tf.reshape(images, [-1,PATCH_DIM,PATCH_DIM,3])
        z = tf.nn.conv2d(x_image, weights, strides=[1, 1, 1, 1], padding='VALID')
        h_conv1 = tf.nn.relu(z+biases, name=scope.name)
    with tf.variable_scope('h_conv2') as scope:
        weights = tf.get_variable('weights', shape=[4, 4, 64, 64], 
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.05))
        z = tf.nn.conv2d(h_conv1, weights, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2 = tf.nn.relu(z+biases, name=scope.name)
    
    h_pool1 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name='h_pool1')
    
    with tf.variable_scope('h_conv3') as scope:
        weights = tf.get_variable('weights', shape=[4, 4, 64, 64], 
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.05))
        z = tf.nn.conv2d(h_pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
        h_conv3 = tf.nn.relu(z+biases, name=scope.name)
        
    h_pool2 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name='h_pool2')
    
    with tf.variable_scope('h_fc1') as scope:
        weights = tf.get_variable('weights', shape=[7**2*64, fc_hidden_units1], 
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=[fc_hidden_units1], initializer=tf.constant_initializer(0.05))
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights) + biases, name = 'h_fc1')
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        
    with tf.variable_scope('h_fc2') as scope:
        weights = tf.get_variable('weights', shape=[fc_hidden_units1, NUM_CLASSES], 
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=[NUM_CLASSES])
        
        logits = (tf.matmul(h_fc1_drop, weights) + biases)
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

OUT_DIR = os.path.abspath("../Data/DRIVE/test_result")

# Make a directory to store the new images in
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
    
mean_np_img = np.asarray(mean_img)


# ### Decoding for all test images

# In[ ]:

h = int(PATCH_DIM/2)
# We want to access the data chunk by chunk such that each chunk has
# approximately BATCH_SIZE pixels
stride = int(np.sqrt(BATCH_SIZE))

begin = time.time()
start_time = time.time()
with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder = placeholder_inputs(BATCH_SIZE)

    # Build a Graph that computes predictions from the inference model.
    logits = inference(images_placeholder, 1.0, 512)
    sm = softmax(logits)

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    with tf.Session() as sess:
        model_save_path = os.path.abspath('../Data/models')
        saver.restore(sess, model_save_path+'/model.ckpt-99')
        
        # Once the model has been restored, we iterate through all images in the test set
        for im_no in xrange(len(test)):
            
            print "Working on image %d" % (im_no+1)
            image = io.imread(test[im_no])
            mask = img_as_float(io.imread(mask_test[im_no]))
            gt = img_as_float(io.imread(gt_test[im_no]))
            
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
                                       feed_dict={images_placeholder: feed})
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

                        if np.mod(pixel_count, 5000) < BATCH_SIZE:
                            print "%d / %d"%(pixel_count, image.shape[0]*image.shape[1])
                            current_time = time.time()
                            print "Time taken - > %f" % (current_time - start_time)
                            start_time = current_time
                    j += stride
                i += stride
            segmented = np.multiply(segmented,mask)
            segmented = segmented * (1.0/segmented.max())
                
            name = test[im_no].split('/')[-1].split('.')[0]
            io.imsave(os.path.join(OUT_DIR, name+'.jpg'), segmented)
print "Total time = %f mins" % ((time.time()-begin)/60.0)


# In[ ]:



