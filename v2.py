# In[24]:

import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
from skimage import io
from skimage.util import img_as_float, img_as_ubyte
import matplotlib.cm as cm
from matplotlib import pyplot as plt

# In[25]:

def get_path(directory):
    imgs = glob.glob(directory + '/images/*.tif')
    #a = [x.split('/')[-1].split('.')[0] for x in train]
    
    mask = glob.glob(directory + '/mask/*.gif')
    #b = [x.split('/')[-1].split('.')[0] for x in mask]
    
    gt = glob.glob(directory + '/1st_manual/*.gif')
    #c = [x.split('/')[-1].split('.')[0] for x in gt]
    
    return map(os.path.abspath, imgs), map(os.path.abspath, mask), map(os.path.abspath, gt)

train, mask_train, gt_train =  get_path('../Data/DRIVE/training')
test, mask_test, mask_gt = get_path('../Data/DRIVE/test')


# In[26]:

# Hyper Params
total_patches = 600
num_training_images = len(train)
patches_per_image = total_patches/num_training_images
patch_dim = 31                         
current_batch_ind = 0
print patches_per_image


# In[27]:

data = pd.read_pickle('../Data/mean_normalised_df.pkl') 
mean_img = pd.read_pickle('../Data/mean_img.pkl')
def next_batch(size, df):
    global current_batch_ind
    flag = False
    if current_batch_ind + size> len(df):
        print 'Next batch cannot be called because of insufficient remaining data'
        flag = True
    batch_x = np.zeros((size, patch_dim**2*3))
    batch_y = np.zeros((size,2), dtype = 'uint8')
    for i in range(current_batch_ind, current_batch_ind+size):
        batch_x[i - current_batch_ind] = df.loc[i][:-1]
        batch_y[i - current_batch_ind][int(df.loc[i][patch_dim**2*3])]=1
    current_batch_ind += size
    return (batch_x, batch_y), flag


# In[28]:

sess = tf.InteractiveSession()
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W, pad_type=1):
    if pad_type == 1:
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    else:
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
def max_pool_2x2(x,pad_type=1):
    if pad_type == 1:
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    else:
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')   


# In[29]:

x = tf.placeholder(tf.float32, shape=[None, patch_dim**2*3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])


# In[30]:

W_conv1 = weight_variable([4, 4, 3, 64])
b_conv1 = bias_variable([64])

x_image = tf.reshape(x, [-1,patch_dim,patch_dim,3]) # Check
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1,0) + b_conv1)


W_conv2 = weight_variable([4, 4, 64, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_pool1 = max_pool_2x2(h_conv2)


W_conv3 = weight_variable([4, 4, 64, 64])
b_conv3 = bias_variable([64])

h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
h_pool2 = max_pool_2x2(h_conv3)


# In[31]:

fc_neurons = 512

W_fc1 = weight_variable([7 * 7 * 64, fc_neurons])
b_fc1 = bias_variable([fc_neurons])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([fc_neurons, 2])
b_fc2 = bias_variable([2])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# In[32]:

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[33]:

num_folds = 5
test_accuracy = np.zeros(num_folds)
subset_size = len(data) / 5

for fold in range(num_folds):
    current_batch_ind = 0
    print 'Begin splitting'
    test_data = data[fold*subset_size:(fold+1)*subset_size]
    test_data = test_data.reset_index(drop = True)
    print 'Test data obtained'
    
    train_data = pd.concat([data[:fold*subset_size], data[(fold+1)*subset_size:]])
    train_data = train_data.reset_index(drop = True)
    print 'Train data obtained'
    
    sess.run(tf.initialize_all_variables())
    for i in range(16):
        batch, empty = next_batch(30, train_data)
        if empty:
            break
        if i%4 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %f"%(i, train_accuracy))
            #print("1 - %g"%(batch[1].mean()) )
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    current_batch_ind = 0
    small_cv_batch = 30
    fold_test_accuracy = np.zeros(subset_size/small_cv_batch)
    for j in range(subset_size/small_cv_batch):
        cv_batch, _ = next_batch(small_cv_batch, test_data)
        fold_test_accuracy[j] = accuracy.eval(feed_dict={x: cv_batch[0], y_: cv_batch[1], keep_prob: 1.0})
    test_accuracy[fold] = fold_test_accuracy.mean()
    print("test accuracy for fold %d = %g"%(fold+1, test_accuracy[fold]))
print 'Average test accuracy = %f' % (test_accuracy.mean())