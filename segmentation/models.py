from __future__ import absolute_import
from __future__ import print_function
import os
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu1,floatX=float32,optimizer=fast_compile,lib.cnmem=0.85'

import pylab as pl
import matplotlib.cm as cm
import itertools
import numpy as np
import theano.tensor as T
np.random.seed(1337) # for reproducibility

from keras.layers.noise import GaussianNoise
import keras.models as models
from keras.layers.core import Layer,Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.regularizers import ActivityRegularizer
#from keras.utils.visualize_util import plot

from keras import backend as K
import h5py as h5
import cv2
import numpy as np


class UnPooling2D(Layer):
    """A 2D Repeat layer"""
    def __init__(self, poolsize=(2, 2)):
        super(UnPooling2D, self).__init__()
        self.poolsize = poolsize

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1],
                self.poolsize[0] * input_shape[2],
                self.poolsize[1] * input_shape[3])

    def get_output(self, train):
        X = self.get_input(train)
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        output = X.repeat(s1, axis=2).repeat(s2, axis=3)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "poolsize":self.poolsize}

def create_encoding_layers(input_shape):
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return [
        ZeroPadding2D(input_shape=input_shape,padding=(pad,pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        #MaxPooling2D(pool_size=(pool_size, pool_size)),
    ]

def create_decoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return[
        #UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
    ]

def create_encoding_layers_small(input_shape):
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return [
        ZeroPadding2D(padding=(pad,pad),input_shape=input_shape),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu')
        #MaxPooling2D(pool_size=(pool_size, pool_size)),
    ]

def create_decoding_layers_small():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return[
        #UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization()
    ]

def segnet(input_shape=(3,360,480)):
	autoencoder = models.Sequential()
	# Add a noise layer to get a denoising autoencoder. This helps avoid overfitting

	#autoencoder.add(GaussianNoise(sigma=0.3))
	autoencoder.encoding_layers = create_encoding_layers(input_shape)
	autoencoder.decoding_layers = create_decoding_layers()
	for i,l in enumerate(autoencoder.encoding_layers):
	    autoencoder.add(l)
	    print(i,l.input_shape,l.output_shape)
	for l in autoencoder.decoding_layers:
	    autoencoder.add(l)
	    print(i,l.input_shape,l.output_shape)

	the_conv=(Convolution2D(num_classes, 1, 1, border_mode='valid',))
	autoencoder.add(the_conv)
	print (the_conv.input_shape,the_conv.output_shape)
	autoencoder.add(Reshape((num_classes,input_shape[1]*input_shape[2])))#, input_shape=(num_classes,360,480)))
	autoencoder.add(Permute((2, 1)))
	autoencoder.add(Activation('softmax'))
	return autoencoder

def segnet_small(input_shape=(3,90,120)):
	autoencoder = models.Sequential()
	# Add a noise layer to get a denoising autoencoder. This helps avoid overfitting

	#autoencoder.add(GaussianNoise(sigma=0.3))
	autoencoder.encoding_layers = create_encoding_layers_small(input_shape)
	autoencoder.decoding_layers = create_decoding_layers_small()
	for i,l in enumerate(autoencoder.encoding_layers):
	    autoencoder.add(l)
	    print(i,l.input_shape,l.output_shape)
	for l in autoencoder.decoding_layers:
	    autoencoder.add(l)
	    print(i,l.input_shape,l.output_shape)

	the_conv=(Convolution2D(num_classes, 1, 1, border_mode='valid',))
	autoencoder.add(the_conv)
	print (the_conv.input_shape,the_conv.output_shape)
	autoencoder.add(Reshape((num_classes,input_shape[1]*input_shape[2])))#, input_shape=(num_classes,360,480)))
	autoencoder.add(Permute((2, 1)))
	autoencoder.add(Activation('softmax'))
	return autoencoder

