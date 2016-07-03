
# coding: utf-8

# In[ ]:

from __future__ import division
import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
import time
from six.moves import xrange 
import shutil
import sys
import argparse


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

data = None
mean_img = None


# In[ ]:

# Hyper Params
TOTAL_PATCHES = None
NUM_IMAGES = None
PATCHES_PER_IMAGE = None
PATCH_DIM = None
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
TRAINING_PROP = 0.8
MAX_STEPS = 125
CKPT_STEP = 40
LOSS_STEP = 2
KEEP_PROB = 0.5
NUM_CLASSES = 2
FCHU1 = 512                # Fully connected layer 1 hidden units
MODEL_NAME = '1'


# In[ ]:

def next_batch(size, df, current_batch_ind):
    """Returns the next mini batch of data from the dataset passed
    
    Args:
        size: length of the current requested mini batch
        df: the data set consisting of the images and the labels
        current_batch_ind: the current position of the index in the dataset
    
    Returns:
        (batch_x, batch_y): A tuple of np arrays of dimensions 
                        [size, patch_dim**2*3] and [size, NUM_CLASSES] respectively 
        current_batch_ind: the updated current position of the index in the dataset
        df: when the requested batch_size+current_batch_ind is more than the length of the data set,
            the data is shuffled again and current_batch_ind is reset to 0, and this new data set is
            returned
    
    """
    
    if current_batch_ind + size> len(df):
        current_batch_ind = 0
        df = df.iloc[np.random.permutation(len(df))]
        df = df.reset_index(drop=True)
        
    batch_x = np.zeros((size, PATCH_DIM**2*3))
    batch_y = np.zeros((size, NUM_CLASSES), dtype = 'uint8')
    for i in range(current_batch_ind, current_batch_ind+size):
        batch_x[i - current_batch_ind] = df.loc[i][:-1]
        batch_y[i - current_batch_ind][int(df.loc[i][PATCH_DIM**2*3])]=1
        
    current_batch_ind += size  
    return (batch_x, batch_y), current_batch_ind, df


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

def calc_loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].
    Returns:
        loss: Loss tensor of type float.
    """
    labels = tf.to_float(labels)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.scalar_summary('cross_entropy_loss', loss)
    
    return loss


# In[ ]:

def training(loss, learning_rate=5e-4):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.
    Returns:
        train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    # Create the Adam optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# In[ ]:

def evaluation(logits, labels, topk=1):
    """Evaluate the quality of the logits at predicting the label.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
                  range [0, NUM_CLASSES).
        topk: the number k for 'top-k accuracy'
    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    
    correct = tf.nn.in_top_k(logits, tf.reshape(tf.slice(labels, [0,1], [int(labels.get_shape()[0]), 1]),[-1]), topk)
    # Return the number of true entries.
    accurate =  tf.reduce_sum(tf.cast(correct, tf.int32))
    accuracy =  tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.scalar_summary('accuracy', accuracy)
    
    return accurate


# In[ ]:

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
        batch_size: The batch size will be baked into both placeholders.
    Returns:
        images_placeholder: Images placeholder.
        labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, PATCH_DIM**2*3))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size, NUM_CLASSES))
    return images_placeholder, labels_placeholder


# In[ ]:

#UPDATE current_img_ind
def fill_feed_dict(data_set, images_pl, labels_pl, current_img_ind, batch_size, keep_prob):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
                  <placeholder>: <tensor of values to be passed for placeholder>,
                  ....
                }
    Args:
        data_set: The set of images and labels, from input_data.read_data_sets()
        images_pl: The images placeholder, from placeholder_inputs().
        labels_pl: The labels placeholder, from placeholder_inputs().
        current_img_ind: The current position of the index in the dataset
        keep_prob: Placeholder for dropout's keep_probability
    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
        current_img_ind: The updated position of the index in the dataset
        data_set: updated data_set
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.
    batch, current_img_ind, data_set= next_batch(batch_size, data_set, current_img_ind)

    feed_dict = {
      images_pl: batch[0],
      labels_pl: batch[1],
      keep_prob: KEEP_PROB
    }
    return feed_dict, current_img_ind, data_set


# In[ ]:

def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set, batch_size, keep_prob):
    """Runs one evaluation against the full epoch of data.
    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
                input_data.read_data_sets().
        keep_prob: Placeholder for dropout's keep_probability
    Output:
        precision: Accuracy of one evaluation of epoch data

    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = len(data_set) // batch_size
    num_examples = steps_per_epoch * batch_size
    current_img_ind = 0
    for step in xrange(steps_per_epoch):
        feed_dict, current_img_ind, data_set = fill_feed_dict(data_set, images_placeholder,
                               labels_placeholder, current_img_ind, batch_size, keep_prob)
        feed_dict[keep_prob] = 1.0
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
            (num_examples, true_count, precision))
    return precision


# In[ ]:

def finish_parsing():
    global BATCH_SIZE, LEARNING_RATE, TRAINING_PROP, MAX_STEPS, CKPT_STEP, LOSS_STEP, FCHU1, KEEP_PROB, MODEL_NAME
    
    parser = argparse.ArgumentParser(description=
                                     'Training script')
    parser.add_argument("--batch", type=int,
                        help="Batch Size [Default - 64]")
    parser.add_argument("--fchu1", type=int,
                        help="Number of hidden units in FC1 layer [Default - 512]")
    parser.add_argument("--learning_rate", type=float,
                        help="Learning rate for optimiser [Default - 5e-4]")
    parser.add_argument("--training_prop", type=float,
                        help="Proportion of data to be used for training data [Default - 0.8]")
    parser.add_argument("--max_steps", type=int,
                        help="Maximum number of iteration till which the program must run [Default - 100]")  
    parser.add_argument("--checkpoint_step", type=int,
                        help="Step after which an evaluation is carried out on validation set and model is saved [Default - 50]")
    parser.add_argument("--loss_step", type=int,
                        help="Step after which loss is printed [Default - 5]")
    parser.add_argument("--keep_prob", type=float,
                        help="Keep Probability for dropout layer [Default - 0.5]")
    parser.add_argument("--model_name",
                        help="Index of the model [Default - '1']")
    args = parser.parse_args()
    
    global total_patches, patch_dim, positive_proprtion
    if args.batch is not None:
        BATCH_SIZE = args.batch
        print "New BATCH_SIZE = %d" % BATCH_SIZE
    if args.model_name is not None:
        MODEL_NAME = args.model_name
        print "New MODEL_NAME = %s" % MODEL_NAME
    if args.fchu1 is not None:
        FCHU1 = args.fchu1
        print "New FCHU1 = %d" % FCHU1
    if args.learning_rate is not None:
        LEARNING_RATE = args.learning_rate
        print "New LEARNING_RATE = %.5f" % LEARNING_RATE
    if args.training_prop is not None:
        TRAINING_PROP = args.training_prop
        print "New TRAINING_PROP = %.2f" % TRAINING_PROP
    if args.max_steps is not None:
        MAX_STEPS = args.max_steps
        print "New MAX_STEPS = %d" % MAX_STEPS
    if args.checkpoint_step is not None:
        CKPT_STEP = args.checkpoint_step
        print "New CKPT_STEP = %d" % CKPT_STEP
    if args.loss_step is not None:
        LOSS_STEP = args.loss_step
        print "New LOSS_STEP = %d" % LOSS_STEP
    if args.keep_prob is not None:
        KEEP_PROB = args.keep_prob
        print "New KEEP_PROB = %.2f" % KEEP_PROB


# In[ ]:

def run_training():
    """Train for a number of steps."""
    # Tell TensorFlow that the model will be built into the default Graph.
    train_data = data[:int(TRAINING_PROP*len(data))]
    test_data = data[int(TRAINING_PROP*len(data)):]
    train_data = train_data.reset_index(drop = True)
    test_data = test_data.reset_index(drop = True)
    
    validation_accuracy = np.zeros((1,3))
    
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)
        
        keep_prob = tf.placeholder(tf.float32)
        # Build a Graph that computes predictions from the inference model.
        logits = inference(images_placeholder, keep_prob, FCHU1)

        # Add to the Graph the Ops for loss calculation.
        loss = calc_loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss, LEARNING_RATE)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = evaluation(logits, labels_placeholder)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        logs_path = os.path.abspath('../../Data/logs/')
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)
        summary_path = os.path.abspath(logs_path+'/model'+MODEL_NAME+'/')
        if os.path.exists(summary_path):
            shutil.rmtree(summary_path)
        os.mkdir(summary_path)

        summary_writer = tf.train.SummaryWriter(summary_path, sess.graph)
        
        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)

        current_img_ind = 0
        # And then after everything is built, start the training loop.
        for step in xrange(MAX_STEPS):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict, current_img_ind, train_data = fill_feed_dict(train_data,
                                 images_placeholder,
                                 labels_placeholder, current_img_ind=current_img_ind, 
                                                        batch_size=BATCH_SIZE, keep_prob=keep_prob)
            

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.
            _, summary_str, loss_value = sess.run([train_op, summary_op, loss],
                               feed_dict=feed_dict)
            # Update the events file.
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % LOSS_STEP == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                
            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % (CKPT_STEP) == 0 or (step + 1) == MAX_STEPS:
                model_path = os.path.abspath('../../Data/models/')
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                model_save_path = os.path.abspath(model_path+'/model'+MODEL_NAME+'/')
                if not os.path.exists(model_save_path):
                    os.mkdir(model_save_path)
                saver.save(sess, model_save_path+'/model.ckpt', global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                train_acc = do_eval(sess, eval_correct, images_placeholder, labels_placeholder, train_data, BATCH_SIZE, keep_prob)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                valid_acc = do_eval(sess, eval_correct, images_placeholder, labels_placeholder, test_data, BATCH_SIZE, keep_prob)
                
                validation_accuracy = np.append(validation_accuracy, np.array([[step, train_acc, valid_acc]]), axis=0)
    np.save(os.path.abspath('../../Data/models/model'+MODEL_NAME+'/')+ '/validation_accuracy', validation_accuracy)


# In[ ]:

def main():
    finish_parsing()
    
    global data, mean_img, TOTAL_PATCHES, NUM_IMAGES, PATCHES_PER_IMAGE, PATCH_DIM
    print 'Loading data'
    data = pd.read_pickle('../../Data/mean_normalised_df_no_class_bias.pkl') 
    mean_img = pd.read_pickle('../../Data/mean_img_no_class_bias.pkl')
    print 'Loading complete'
    
    
    train, mask_train, gt_train =  get_path('../../Data/DRIVE/training')
    test, mask_test, mask_gt = get_path('../../Data/DRIVE/test')

    
    # Changing some Hyper Params
    TOTAL_PATCHES = len(data)
    NUM_IMAGES = len(train)
    PATCHES_PER_IMAGE = TOTAL_PATCHES/NUM_IMAGES
    PATCH_DIM = int(np.sqrt((len(data.columns)-1)/3))
    
    
    run_training()


# In[ ]:

if __name__ == "__main__":
    '''
    sys.argv = ['v2_graph.py', '--batch', '64', '--fchu1', '128', '--learning_rate', '5e-4',
               '--training_prop', '0.9', '--max_steps', '20', 
                '--checkpoint_step', '10', '--loss_step', '2', '--keep_prob', '0.6',
                '--model_name', '3']
    '''
    main()

