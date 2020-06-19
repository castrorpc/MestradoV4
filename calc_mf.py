'''
Calculus of the membership functions
'''

import tensorflow as tf

@tf.function
def gaussian(mi, sigma, x):
    return tf.math.exp((-(x - mi)**2)/2*sigma**2)

@tf.function
def bell(a, b, c, x):
    return 1/(1 + tf.math.pow(tf.math.abs((x - c)/a), 2*b))

@tf.function
def triangular(a, b, c, x):
    return tf.mat.maximum(tf.math.minimum((x - a)/(b - a), (c - x)/(c - b)), 0)
