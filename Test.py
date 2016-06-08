
# coding: utf-8

# In[14]:

from __future__ import division
import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
from skimage import io, color
from skimage.util import img_as_float, img_as_ubyte
#import matplotlib.cm as cm
#from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')
import time
from six.moves import xrange 


# In[15]:

data = pd.read_pickle('../Data/mean_normalised_df_no_class_bias.pkl') 
mean_img = pd.read_pickle('../Data/mean_img_no_class_bias.pkl')


# In[27]:

PATCH_DIM = 31
BATCH_SIZE = 100 # Must be a perfect square
NUM_CLASSES = 2


# In[28]:

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
test, mask_test, mask_gt = get_path('../Data/DRIVE/test')


# In[29]:

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


# In[30]:

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    Args:
        batch_size: The batch size will be baked into both placeholders.
    Returns:
        images_placeholder: Images placeholder.
    """
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, PATCH_DIM**2*3))
    return images_placeholder


# In[31]:

def get_predictions(batch_x):
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        images_placeholder = placeholder_inputs(BATCH_SIZE)

        # Build a Graph that computes predictions from the inference model.
        logits = inference(images_placeholder, 1.0, 512)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        with tf.Session() as sess:
            saver.restore(sess, '../Data/model.ckpt')

            predictions = sess.run([logits],
                                   feed_dict={images_placeholder: batch_x})
            predictions = np.asarray(predictions).reshape(BATCH_SIZE, NUM_CLASSES)
            
            predicted_labels = np.argmax(predictions, axis=1)
            return predicted_labels


# In[32]:

image = io.imread(train[0])
mask = img_as_float(io.imread(mask_train[0]))
gt = img_as_float(io.imread(gt_train[0]))
mean_np_img = np.asarray(mean_img)


# In[33]:

"""
segmented = np.zeros(image.shape[:2])

rows = np.zeros(BATCH_SIZE, dtype='uint8')
cols = np.zeros(BATCH_SIZE, dtype='uint8')
feed = np.zeros((BATCH_SIZE, PATCH_DIM**2*3))
predictions = np.zeros(BATCH_SIZE)

count = 0
pixel_count = 0
h = int(PATCH_DIM/2)

start_time = time.time()
for i in range(h, image.shape[0] - h-1):
    for j in range(h, image.shape[1] - h-1):
        
        if int(np.sum(mask[i-h:i+h+1,j-h:j+h+1])/PATCH_DIM**2) == 1:
            pixel_count += 1
            if count < BATCH_SIZE-1:
                count += 1
                feed[count] = image[i-h:i+h+1,j-h:j+h+1].reshape(-1)
                rows[count] = i
                cols[count] = j
            else:
                # Subtract training mean image
                feed = feed - mean_np_img
                
                # Get predictions and draw accordingly on black image
                
                predictions = get_predictions(feed)
                segmented[rows,cols] = predictions
                
                # Reset everything after passing feed to feedforward
                rows = np.zeros(BATCH_SIZE, dtype='uint8')
                cols = np.zeros(BATCH_SIZE, dtype='uint8')
                feed = np.zeros((BATCH_SIZE, PATCH_DIM**2*3))
                predictions = np.zeros(BATCH_SIZE)
                count = 0
                if pixel_count%3000 == 0:
                    print "%d / %d"%(pixel_count, image.shape[0]*image.shape[1])
                    current_time = time.time()
                    print "Time taken - > %f" % (current_time - start_time)
                    start_time = current_time

segmented[0][0] = 0 # To nullify effects of final buffer
                
"""


# In[34]:

def nbd(image, point):
    i = point[0]
    j = point[1]
    h = int(PATCH_DIM/2)
    return image[i-h:i+h+1,j-h:j+h+1].reshape(-1)


# In[36]:

# We will start with a completely black image and update it chunk by chunk
segmented = np.zeros(image.shape[:2])

# We will use arrays to index the image and mask later
rows, cols = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
row_col = np.stack([rows,cols], axis = 2)
# The neighborhood windows to be fed into the graph
feed = np.zeros((BATCH_SIZE, PATCH_DIM**2*3))
# The predicted classes for all the windows that were fed to the graph
predictions = np.zeros(BATCH_SIZE)
# We want to access the data chunk by chunk such that each chunk has
# approximately BATCH_SIZE pixels
stride = int(np.sqrt(BATCH_SIZE))

pixel_count = 0
h = int(PATCH_DIM/2)

begin = time.time()
start_time = time.time()
with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder = placeholder_inputs(BATCH_SIZE)

    # Build a Graph that computes predictions from the inference model.
    logits = inference(images_placeholder, 1.0, 512)

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    with tf.Session() as sess:
        saver.restore(sess, '../Data/model.ckpt')
        
        i = h
        while i < image.shape[0] - h-1:
            j = h
            while j < image.shape[1] - h-1:
                # A small check is made to ensure that not all pixels are black
                #print "i = %d, j = %d" % (i,j)
                if int(np.max(mask[i:i+stride,j:j+stride])) == 1:

                    pixel_count += BATCH_SIZE # This will not be true for border cases though
                                              # but we don't care about the progress at the end

                    # Update i and j by adding stride but take care near the end
                    i_next = min(i+stride, image.shape[0]-h-1)
                    j_next = min(j+stride, image.shape[1]-h-1)

                    # Once we get a chunk, we flatten it and map a function that returns 
                    # the neighborhood of each point

                    chunk = np.array(map(lambda p: nbd(image, p), row_col[i:i_next, j:j_next].reshape(-1,2)))
                    if len(chunk) == BATCH_SIZE:
                        feed = chunk
                    else:
                        feed[:len(chunk)] = chunk
                    #print " Feed shape = (%d, %d)" % feed.shape
                    # Subtract training mean image
                    feed = feed - mean_np_img

                    # Get predictions and draw accordingly on black image    
                    predictions = sess.run([logits],
                                   feed_dict={images_placeholder: feed})
                    predictions = np.asarray(predictions).reshape(BATCH_SIZE, NUM_CLASSES)
            
                    predictions = np.argmax(predictions, axis=1)
                    segmented[rows[i:i_next, j:j_next], cols[i:i_next, j:j_next]] = predictions.reshape(i_next-i, j_next-j)

                    # Reset everything after passing feed to feedforward
                    feed = np.zeros((BATCH_SIZE, PATCH_DIM**2*3))
                    predictions = np.zeros(BATCH_SIZE)

                    if pixel_count%3000 == 0:
                        print "%d / %d"%(pixel_count, image.shape[0]*image.shape[1])
                        current_time = time.time()
                        print "Time taken - > %f" % (current_time - start_time)
                        start_time = current_time
                j += stride
            i += stride
print "Total time = %f mins" % ((time.time()-begin)/60.0)


# In[ ]:



