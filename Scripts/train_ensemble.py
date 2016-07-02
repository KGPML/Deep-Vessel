
# coding: utf-8

# In[1]:

import os
import json
import sys
import argparse


# In[2]:

ARGS_PATH = os.path.abspath("../arglists/cl_args.json")


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


# In[4]:

def get_args(model):
    """Parses a model's parameters from a dictionary
        
        Args:
            model: Dictionary containing parameters of one model
        Output:
            args: Exact command line call to be used

    """
    args = 'python v2_graph.py'
    for key in model.keys():
        args = args + ' ' + str(key) + ' ' + str(model[key])
    return args


# In[5]:

def finish_parsing():
    global ARGS_PATH
    parser = argparse.ArgumentParser(description=
                                     'Training an ensemble of convnets')
    parser.add_argument("--a",
                        help="Path to JSON file containing model arguments")
    args = parser.parse_args()
    if args.a is not None:
        ARGS_PATH = os.path.abspath(args.a)
        print "New ARGS_PATH = %s" % ARGS_PATH


# In[6]:

def main():
    finish_parsing()
    models = load_json(ARGS_PATH) 
    for m in models:
        print '--'*30
        print 'Beginning training for Model '+m['--model_name'] +'\n'
        arg_list = get_args(m)
        print arg_list
        os.system(arg_list)


# In[7]:

if __name__ == "__main__":
    sys.argv = ['train_ensemble.py', '--a', '../arglists/cl_args.json']
    main()

