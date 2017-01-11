from __future__ import absolute_import
from __future__ import print_function

import pylab as pl
import matplotlib.cm as cm
import itertools
import numpy as np
import theano.tensor as T
np.random.seed(1337) # for reproducibility

from keras.datasets import mnist
from keras.layers.noise import GaussianNoise
import keras.models as models
from keras.layers.core import Input, Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.regularizers import ActivityRegularizer
from keras.utils.visualize_util import plot

from keras import backend as K
import tensorflow as tf
import tensorflow.contrib.crf as crf

import cv2
import numpy as np

path = './CamVid/'
data_shape = 360*480

def normalized(rgb):
    #return rgb/255.0
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm

def binarylab(labels):
    x = np.zeros([360,480,12])
    for i in range(360):
        for j in range(480):
            x[i,j,labels[i][j]]=1
    return x

def prep_data():
    train_data = []
    train_label = []
    import os
    with open(path+'train.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    for i in range(len(txt)):
        train_data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])),2))
        train_label.append(binarylab(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[:,:,0]))
        print('.',end='')
    return np.array(train_data), np.array(train_label)

train_data, train_label = prep_data()
train_label = np.reshape(train_label,(367,data_shape,12))

class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]

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

def create_encoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return [
        ZeroPadding2D(padding=(pad,pad)),
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
def model_segnet():
    
    # Add a noise layer to get a denoising autoencoder. This helps avoid overfitting
    IN=x=Input(input_shape=(3,360,480))
    # autoencoder.add(Layer(input_shape=(3, 360, 480)))

    #autoencoder.add(GaussianNoise(sigma=0.3))
    encoding_layers = create_encoding_layers()
    decoding_layers = create_decoding_layers()
    for l in encoding_layers:
        x=l(x)
    for l in decoding_layers:
        x=l(x)

    x=(Convolution2D(num_classes, 1, 1, border_mode='valid',))(x)
    # import ipdb; ipdb.set_trace()
    x=(Reshape((num_classes,data_shape), input_shape=(num_classes,360,480)))(x)
    return [IN],[x]
def model_segnet_train(batch_size):
    inputs,outputs=model_segnet()
    x=Permute((2,1))(outputs[0])
    y=Input(input_shape=(data_shape,num_classes))
    sequence_lengths = np.full(batch_size, data_shape - 1, dtype=np.int32)
    sequence_lengths_t = tf.constant(sequence_lengths)
    log_likelihood, transition_params=crf.crf_log_likelihood(
        x, y, sequence_lengths_t)

    # Add a training op to tune the parameters.
    loss = tf.reduce_mean(-log_likelihood)

    #from keras.optimizers import SGD
    #optimizer = SGD(lr=0.01, momentum=0.8, decay=0., nesterov=False)
    # autoencoder.compile(loss="categorical_crossentropy", optimizer='adadelta')
    train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

    return [x, transition_params, train_op]

    # current_dir = os.path.dirname(os.path.realpath(__file__))
    # model_path = os.path.join(current_dir, "autoencoder.png")
    # plot(model_path, to_file=model_path, show_shapes=True)

    # nb_epoch = 100
    # batch_size = 14

    # history = autoencoder.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch,
    #                     show_accuracy=True, verbose=1, class_weight=class_weighting )#, validation_data=(X_test, X_test))

    # autoencoder.save_weights('model_weight_ep100.hdf5')
