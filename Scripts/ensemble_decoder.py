
# coding: utf-8

# In[1]:

import os
import json
import sys
import argparse


# In[2]:

ARGS_PATH = os.path.abspath("../arglists/cl_args.json")
MODELS_PATH = os.path.abspath("../argslist/model_paths.json")
OUT_DIR = os.path.abspath("../../Data/DRIVE/ensemble_test_results")
IN_DIR = os.path.abspath("../../Data/DRIVE/test")

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
        OUT_DIR = args.out
        print "New OUT_DIR = %s" % OUT_DIR
    if args.inp is not None:
        IN_DIR = args.inp
        print "New IN_DIR = %s" % IN_DIR


# In[6]:

def main():
    finish_parsing()
    model_args = load_json(ARGS_PATH) 
    model_name = [int(m['--model_name']) for m in model_args]
    model_fchu1 = [str(m['--fchu1']) for m in model_args]

    model_paths = load_json(MODELS_PATH)
    
    num_models = 0
    while num_models < 13:
        num_models += 1
        if os.path.exists(model_paths[str(num_models)]):
            args = 'python Test.py --inp ' + IN_DIR +' --out ' + OUT_DIR
            print '--'*40
            print 'Decoding using model ' + str(num_models) +'\n'
            args = args + ' --fchu1 ' + model_fchu1[model_name.index(num_models)]
            args = args + ' --format npz '
            args = args + ' --model ' + model_paths[str(num_models)]

            print args

            os.system(args)            

# In[7]:

if __name__ == "__main__":
    main()

