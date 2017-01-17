#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 23:37:04 2017

@author: LinZhang
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle



pickle_file = 'notMNIST.pickle'
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)  # unpickle
    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    valid_dataset = data['valid_dataset']
    valid_labels = data['valid_labels']
    test_dataset = data['test_dataset']
    test_labels = data['test_labels']   
    f.close()
 
lr_classifer = LogisticRegression(verbose=1,max_iter = 10);
lr_classifer.fit(train_dataset.reshape((train_dataset.shape[0], -1), order='C'),train_labels)
lr_classifer.score(valid_dataset.reshape((train_dataset.shape[0], -1), order='C'),valid_labels)
lr_classifer.score(test_dataset.reshape((train_dataset.shape[0], -1), order='C'),test_labels)

print('done...')


