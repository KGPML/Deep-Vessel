
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from skimage import io, draw
from skimage.util import img_as_float, img_as_ubyte
#%matplotlib inline
import os
import glob
import time


# In[2]:

def get_path(directory):
    """ Gets the filenames of all training, mask and ground truth images in the given 
        directory 
        Args:
            directory: The path to the root folder
        Output:
            imgs: List of paths to files containing images
            mask: List of paths to files containing masks of the images
            gt:   List of paths to files containing corresponding ground truth images
    """
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


# In[3]:

# Hyper Params
total_patches = 600
num_training_images = len(train)
patches_per_image = total_patches/num_training_images
patch_dim = 31                          # Dimension of window used for training
num_patches = 0                         # Patches used for training from the current image
current_img_index = -1                   # Index of the current image in 'train'
current_img = io.imread(train[0])    
current_mask = img_as_float(io.imread(mask_train[0]))
current_gt = img_as_float(io.imread(gt_train[0]))
positive_proprtion = 0.5


# In[4]:

def load_next_img(data,mask_data,gt_data):
    """When we have extracted 'PATCHES_PER_IMAGE' number of patches from our 
       current image we call this function to change the current image
       Args:
           data: The list of paths to the images
           mask_data: List of paths to the corresponding masks of images
           gt_data: List of paths to the corresponding ground truth images
       
    """
    global num_patches, current_img_index, current_img, current_mask, current_gt
    num_patches = 0
    if current_img_index < len(data)-1:
        current_img_index +=1
        print "Working on image %d"%(current_img_index + 1)
        current_img = io.imread(data[current_img_index])                     
        current_mask = img_as_float(io.imread(mask_data[current_img_index])) 
        current_gt = img_as_float(io.imread(gt_data[current_img_index])) 
        return True
    else:
        print 'No more images left in set'
        return False


# In[7]:

begin = time.time()
print "Creating DataFrame"
df = pd.DataFrame(index=np.arange(total_patches), columns = np.arange(patch_dim**2*3+1))
print "Dataframe ready"


# In[8]:

def save_img_data(data, mask_data, gt_data):
    """Extracts PATCHES_PER_IMAGE number of patches from each image
        
       It maintains a count of positive and negative patches and maintains
       the ratio POSITIVE_PROPORTION = pos/(pos+neg)
       Args:
           data: The list of paths to the images
           mask_data: List of paths to the corresponding masks of images
           gt_data: List of paths to the corresponding ground truth images
       
    """
    pos_count = 0
    neg_count = 0
    global df
    while pos_count +neg_count < patches_per_image: 
        # Choose a random point
        i = np.random.randint(patch_dim/2,current_img.shape[0]-patch_dim/2)
        j = np.random.randint(patch_dim/2,current_img.shape[1]-patch_dim/2)
        h = (patch_dim - 1)/2
        if int(np.sum(current_mask[i-h:i+h+1,j-h:j+h+1])/patch_dim**2) == 1:
            ind = current_img_index*patches_per_image+pos_count+neg_count
            
            # If a positive sample is found and positive count hasn't reached its limit
            if int(current_gt[i,j])==1 and pos_count < positive_proprtion*patches_per_image:
                df.loc[ind][0:-1] = np.reshape(current_img[i-h:i+h+1,j-h:j+h+1], -1)
                df.loc[ind][patch_dim**2*3] = int(current_gt[i,j])
                pos_count += 1
            # If a negative sample is found and negative count hasn't reached its limit
            elif int(current_gt[i,j])==0 and neg_count < (1-positive_proprtion)*patches_per_image:
                df.loc[ind][0:-1] = np.reshape(current_img[i-h:i+h+1,j-h:j+h+1], -1)
                df.loc[ind][patch_dim**2*3] = int(current_gt[i,j])
                neg_count += 1


# In[9]:

while load_next_img(train, mask_train, gt_train):
    start = time.time()
    save_img_data(train,mask_train, gt_train)
    print "Time taken for this image = %f secs" %( (time.time()-start))


# In[10]:

print "Mean Normalising"
last = len(df.columns) -1
mean_img = np.mean(df)[:-1]
labels = df[last]
mean_normalised_df = df - np.mean(df)
mean_normalised_df[last] = labels


# In[11]:

print "Randomly shuffling the datasets"
mean_normalised_df = mean_normalised_df.iloc[np.random.permutation(len(df))]
mean_normalised_df = mean_normalised_df.reset_index(drop=True)


# In[12]:

print "Writing to pickle"
mean_normalised_df.to_pickle('../Data/mean_normalised_df_no_class_bias.pkl')
mean_img.to_pickle('../Data/mean_img_no_class_bias.pkl')


# In[13]:

print "Total time taken = %f mins" %( (time.time()-begin)/60.0)

