import numpy as np
from theano import function
from theano import tensor as T
from theano import config

a = T.vector()
b = T.log(a)
c = T.nnet.sigmoid(b)
d = T.sqrt(c)
e = T.concatenate((d, c), axis=0)
f = b * c * d
# This is the first bad line
g = e + f
h = g / c
fn = function([a], h, mode='FAST_COMPILE')
fn(np.ones((3,)).astype(a.dtype))
