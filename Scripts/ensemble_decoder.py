
# coding: utf-8

# In[1]:

import os
import json
import sys
import argparse
from skimage import io
import time
import shutil
import glob
import numpy as np

# In[2]:

ARGS_PATH = os.path.abspath("../arglists/cl_args.json")
MODELS_PATH = os.path.abspath("../argslist/model_paths.json")
OUT_DIR = os.path.abspath("../../Data/DRIVE/ensemble_test_results")
IN_DIR = os.path.abspath("../../Data/DRIVE/test")
MAX_MODELS = 13
# In[3]:

def load_json(json_path):
    """Loads a json file with user defined arguments for all models
        
        Args:
            json_path: Input filepath
        Output:
            List of dictionaries, each dictionary representing parameters for
            each model
    
    """
    if not os.path.exists(json_path):
        print "No json file found"

    with open(json_path, 'r') as infile:
        return json.load(infile)



def finish_parsing():
    global ARGS_PATH, MODELS_PATH
    parser = argparse.ArgumentParser(description=
                                     'Training an ensemble of convnets')
    parser.add_argument("--a",
                        help="Path to JSON file containing model arguments")
    parser.add_argument("--m",
                        help="Path to JSON file containing saved model paths")

    parser.add_argument("--out",
                        help="Directory to put rendered images to")
    parser.add_argument("--inp",
                        help="Directory containing images for testing")

    args = parser.parse_args()
    if args.a is not None:
        ARGS_PATH = os.path.abspath(args.a)
        print "New ARGS_PATH = %s" % ARGS_PATH
    if args.m is not None:
        MODELS_PATH = os.path.abspath(args.m)
        print "New MODELS_PATH_PATH = %s" % MODELS_PATH
    if args.out is not None:
        OUT_DIR = os.path.abspath(args.out)
        print "New OUT_DIR = %s" % OUT_DIR
    if args.inp is not None:
        IN_DIR = os.path.abspath(args.inp)
        print "New IN_DIR = %s" % IN_DIR



def main():
    finish_parsing()
    model_args = load_json(ARGS_PATH) 
    model_name = [int(m['--model_name']) for m in model_args]
    model_fchu1 = [str(m['--fchu1']) for m in model_args]

    model_paths = load_json(MODELS_PATH)
    
    """
    This code decodes all test images for each model and puts them in a tmp folder
    in the format npz. It then later loads all these npz files and saves the mean as 
    a png image
    """
    TMP_DIR = os.path.abspath(OUT_DIR +'/tmp/')
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
    if os.path.exists(TMP_DIR):
        print "Temp path already exists. Delete this"
    os.mkdir(TMP_DIR)

    m_no = 0
    model_exists = [0 for i in range(MAX_MODELS)]

    while m_no < MAX_MODELS:
        m_no += 1
        if os.path.exists(model_paths[str(m_no)]):
            # Some models don't get created due to large parameter values and OOM errors.
            # Hence it is necessary to check if a model exists
            model_exists[m_no-1] = 1

            print '--'*40
            print 'Decoding using model ' + str(m_no) +'\n'

            args = 'python Test.py --inp ' + IN_DIR +' --out ' + TMP_DIR+'/model'+str(m_no)
            args = args + ' --fchu1 ' + model_fchu1[model_name.index(m_no)]
            args = args + ' --format npz '
            args = args + ' --model ' + model_paths[str(m_no)]

            print args

            os.system(args) 
    """
    Now that all the decoded arrays are in tmp file, we combine them one at a time
    """ 
    npz_paths = []  
    for i in range(MAX_MODELS):
        if model_exists[i] == 1:

            paths = map(os.path.abspath, glob.glob(os.path.join(TMP_DIR,'model' + str(i+1) + '/*.npz')))
            paths.sort()
            npz_paths.append(paths)
            

    new_file = np.load(npz_paths[0][0])
    new_file = new_file[new_file.keys()[0]]
    sum_img = np.zeros_like(new_file)

    print '--'*40
    print 'Combining'
    begin = time.time()
    for im_no in range(len(npz_paths[0])):
        for i in range(len(npz_paths)):
            new_file = np.load(npz_paths[i][im_no])
            new_file = new_file[new_file.keys()[0]]
            sum_img += new_file

        # Take average
        sum_img = sum_img/len(npz_paths)
        sum_img = sum_img/sum_img.max()

        name = npz_paths[0][im_no].split('/')[-1].split('.')[0]
        io.imsave(os.path.join(OUT_DIR, name+'.png'), sum_img)
    print 'Time taken = %.2f secs' % (time.time() - begin)
    shutil.rmtree(TMP_DIR)


# In[7]:

if __name__ == "__main__":
    main()

