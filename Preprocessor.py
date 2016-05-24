
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from skimage import io, draw
from skimage.util import img_as_float, img_as_ubyte
get_ipython().magic(u'matplotlib inline')
import os
import glob


# In[2]:

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


# In[4]:

# When we have extracted 'patches_per_image' number of patches from our current image
# we call this function to change the current image
def load_next_img(data,mask_data,gt_data):
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
df = pd.DataFrame(columns = np.arange(patch_dim**2*3+1))

def save_img_data(data, mask_data, gt_data):
    count = 0
    global df
    while count < patches_per_image:
        i = np.random.randint(0,current_img.shape[0])
        j = np.random.randint(0,current_img.shape[1])
        h = (patch_dim - 1)/2
        if int(np.sum(current_mask[i-h:i+h+1,j-h:j+h+1])/patch_dim**2) == 1:
            ind = current_img_index*patches_per_image+count
            df.loc[ind] = np.arange(patch_dim**2*3+1)
            df.loc[ind][0:-1] = np.reshape(current_img[i-h:i+h+1,j-h:j+h+1], -1)
            df.loc[ind][patch_dim**2*3] = int(current_gt[i,j])
            count +=1
'''
def save_img_data(data, mask_data, gt_data):
    count = 0
    #grid_length = np.floor(np.sqrt(patches_per_image))
    #print 'grid length = %d'%grid_length
    
    row_step = col_step = int(np.ceil(np.sqrt(0.7*current_mask.shape[0]*current_mask.shape[1]/(1.0*patches_per_image))) )   
    #print 'steps = %d'%row_step
    h = (patch_dim-1)/2
    i = h+1
    while i<current_mask.shape[0]-h-1-row_step :
        j = h+1
        while j<current_mask.shape[1]-h+1-col_step and count<patches_per_image:
            p = np.random.randint(0,row_step)
            q = np.random.randint(0,col_step)
            #print 'i = %d, j = %d, count = %d'%(i,j,count)
            #print 'p= %d, q = %d'%(p,q)
            if current_mask[i+p-h,j+q-h]==current_mask[i+p-h,j+q+h]== current_mask[i+p+h,j+q-h] == current_mask[i+p+h,j+q+h]>0.99: #To avoid floating point comparison
                ind = current_img_index*patches_per_image+count
                df.loc[ind] = np.arange(patch_dim**2*3+1)
                df.loc[ind][0:-1] = np.reshape(current_img[i+p-h:i+p+h+1,j+q-h:j+q+h+1], -1)
                df.loc[ind][patch_dim**2*3] = int(current_gt[i+p,j+q])
                count +=1
            
            j+=row_step
        i+=col_step
    #print count
    while count < patches_per_image:
        i = np.random.randint(50,current_img.shape[0]-50)
        j = np.random.randint(50,current_img.shape[1]-50)
        h = (patch_dim - 1)/2
        if current_mask[i-h,j-h]==current_mask[i-h,j+h]== current_mask[i+h,j-h] == current_mask[i+h,j+h]>0.99: #To avoid floating point comparison
            ind = current_img_index*patches_per_image+count
            df.loc[ind] = np.arange(patch_dim**2*3+1)
            df.loc[ind][0:-1] = np.reshape(current_img[i-h:i+h+1,j-h:j+h+1], -1)
            df.loc[ind][patch_dim**2*3] = int(current_gt[i,j])
            count +=1
'''


# In[5]:

while load_next_img(train, mask_train, gt_train):
    save_img_data(train,mask_train, gt_train)


# In[6]:

last = len(df.columns) -1
mean_img = np.mean(df)[:-1]
labels = df[last]
mean_normalised_df = df - np.mean(df)
mean_normalised_df[last] = labels


# In[7]:

mean_normalised_df = mean_normalised_df.iloc[np.random.permutation(len(df))]
mean_normalised_df = mean_normalised_df.reset_index(drop=True)


# In[9]:

mean_normalised_df.to_pickle('../Data/mean_normalised_df.pkl')
mean_img.to_pickle('../Data/mean_img.pkl')

