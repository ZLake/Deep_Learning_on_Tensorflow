#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:07:16 2017

@author: lakezhang
"""

#==============================================================================
# tf.nn.dynamic_rnn better than tf.nn.rnn, just use it!
# ref: http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/

# sequence_length is very important, tells the length of each batch so that tf won't run on padding 0s
#==============================================================================
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

# Create input data
X = np.random.randn(2, 10, 8)

# The second example is of length 6 
X[1,6:] = 0
X_lengths = [10, 6]

cell = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True)

outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)

result = tf.contrib.learn.run_n(
    {"outputs": outputs, "last_states": last_states},
    n=1,
    feed_dict=None)

assert result[0]["outputs"].shape == (2, 10, 64)
print(result[0]["outputs"])

# Outputs for the second example past past length 6 should be 0
assert (result[0]["outputs"][1,7,:] == np.zeros(cell.output_size)).all()
