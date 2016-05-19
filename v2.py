import numpy as np
import os
import glob
import tensorflow as tf
from skimage import io
from skimage.util import img_as_float, img_as_ubyte
import matplotlib.cm as cm
from matplotlib import pyplot as plt

def get_path(directory):
    imgs = glob.glob(directory + '/images/*.tif')
    #a = [x.split('/')[-1].split('.')[0] for x in train]
    
    mask = glob.glob(directory + '/mask/*.gif')
    #b = [x.split('/')[-1].split('.')[0] for x in mask]
    
    gt = glob.glob(directory + '/1st_manual/*.gif')
    #c = [x.split('/')[-1].split('.')[0] for x in gt]
    
    return map(os.path.abspath, imgs), map(os.path.abspath, mask), map(os.path.abspath, gt)

train, mask_train, gt_train =  get_path('../Data/DRIVE/training')
test, mask_test, gt_test = get_path('../Data/DRIVE/test')

# Hyper Params
total_patches = 60000
num_training_images = len(train)
patches_per_image = total_patches/num_training_images
patch_dim = 31                          # Dimension of window used for training
num_patches = 0                         # Patches used for training from the current image
current_img_index = 0                   # Index of the current image in 'train'
current_img = io.imread(train[0])    
current_mask = img_as_float(io.imread(mask_train[0]))
current_gt = img_as_float(io.imread(gt_train[0]))
print patches_per_image

# When we have extracted 'patches_per_image' number of patches from our current image
# we call this function to change the current image
def load_next_img(data,mask_data,gt_data):
    global num_patches, current_img_index, current_img, current_mask, current_gt
    num_patches = 0
    current_img_index +=1
    if current_img_index < len(data):
        current_img = io.imread(data[current_img_index])                     # 0-255
        current_mask = img_as_float(io.imread(mask_data[current_img_index])) # 0.0-1.0
        current_gt = img_as_float(io.imread(gt_data[current_img_index]))     # 0.0-1.0
    else:
        print 'No more images left in set'

# size should be a factor of patches_per_image    
def next_batch(size, data, mask_data, gt_data):
    global num_patches
    count = 0
    batch_x = np.zeros((size, patch_dim**2*3))
    batch_y = np.zeros((size,1), dtype = 'uint8')
    while count < size:
        i = np.random.randint(0,current_img.shape[0])
        j = np.random.randint(0,current_img.shape[1])
        h = (patch_dim - 1)/2
        if int(np.sum(current_mask[i-h:i+h+1,j-h:j+h+1])/patch_dim**2) == 1:
            batch_x[count] = np.reshape(current_img[i-h:i+h+1,j-h:j+h+1], -1)
            batch_y[count] = int(current_gt[i,j])
            count +=1
    num_patches += size
    if num_patches >= patches_per_image:
        load_next_img(data, mask_data, gt_data)
    return (batch_x, batch_y)

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



x = tf.placeholder(tf.float32, shape=[None, patch_dim**2*3])
y_ = tf.placeholder(tf.float32, shape=[None, 1])


W_conv1 = weight_variable([4, 4, 3, 64])
b_conv1 = bias_variable([64])

x_image = tf.reshape(x, [-1,patch_dim,patch_dim,3])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1,0) + b_conv1)


W_conv2 = weight_variable([4, 4, 64, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_pool1 = max_pool_2x2(h_conv2)


W_conv3 = weight_variable([4, 4, 64, 64])
b_conv3 = bias_variable([64])

h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
h_pool2 = max_pool_2x2(h_conv3)


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


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())



for i in range(600):
    batch = next_batch(100,train, mask_train, gt_train)
    if i%2 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %f"%(i, train_accuracy))
        print("1 - %g"%(batch[1].mean()) )
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


current_img_index = -1
test_batch = next_batch(2000, test, mask_test, gt_test)
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))

