
# coding: utf-8

# In[31]:

import os
import json
from pprint import pprint


# In[49]:

""" Our convolutional neural networks are combined in an ensemble of 13 networks to
    improve accuracy. The 13 models are run with different sets of parameters. This 
    program creates a json dump containing all the arguments to be passed to the 
    13 models, which the user can change from the json file later
"""
MAX_MODELS = 13
d = {
    '--batch': '64',
    '--fchu1': '128',
    '--learning_rate': '5e-4',
    '--training_prop': '0.9',
    '--max_steps': '20',
    '--checkpoint_step': '10',
    '--loss_step': '2',
    '--keep_prob': '0.6',
    '--model_name': '3'
    }
params_list = [d for i in range(MAX_MODELS)]
for i in range(MAX_MODELS):
    tmp = params_list[i].copy()
    tmp['--model_name'] = str(i+1)
    params_list[i] = tmp


# In[50]:

pprint(params_list[:3])


# In[53]:

"""We make a directory to store these JSON files. The user can make his/her custom 
    arg list by copying one of these JSON files and editing it
"""
args_path = os.path.abspath("../arglists")
if not os.path.exists(args_path):
    os.mkdir(args_path)
with open(args_path+'/cl_args.json', 'w') as outfile:
    json.dump(params_list, outfile, separators=(',',': '), indent = 4)

