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
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.regularizers import ActivityRegularizer
#from keras.utils.visualize_util import plot

from keras import backend as K
import h5py as h5
import cv2
import numpy as np

path = './household_data/rgbd-dataset/'
data_shape = 360*480
num_classes=53
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

def binarylab(labels,class_):
    x = np.zeros([360,480,num_classes])
    for i in range(360):
        for j in range(480):
            lab_=class_ if labels[i][j]!=0 else 0
            x[i,j,lab_]=1
    return x

def prep_data_gen():
    import os
    with open(path+'train.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    while True:
        train_data = []
        train_label = []
        i=np.random.randint(len(txt))
        if not os.path.exists(path+txt[i][0]):
            continue
        train_data.append(np.rollaxis(normalized(cv2.resize(cv2.imread(path + txt[i][0]), (480,360))),2))
        if not os.path.exists(path+txt[i][1]):
            del train_data[-1]
            continue
#        print (path+txt[i][1])
        train_label.append(binarylab(cv2.resize(cv2.imread(path + txt[i][1],0),(480,360)),int(txt[i][2])))
        if i%100==0:
            print(str(i))
        yield (train_data), (train_label)
def batch_data_gen(batch_size):
    the_gen=prep_data_gen()
    i=0
    while True:
        x,y=[],[]
        for t in range(batch_size):
            temp=next(the_gen)
            x+=temp[0]
            y+=temp[1]
 #       print (i)
        i+=1
#        print(np.array(y).shape)
        y=np.array(y)
        x,y=np.array(x),y.reshape((len(y),data_shape,num_classes))
        yield x,y

#train_data, train_label = prep_data()
#print ('saving data')
#with h5.File('train_data.h5','w') as f:
#    f.create_dataset('data',data=train_data)
#train_label = np.reshape(train_label,(train_label.shape[0],data_shape,num_classes))
#print('saving labels')
#with h5.File('train_label.h5','w') as f:
#    f.create_dataset('label',data=train_label)
#print('done saving')
#print('loading data')
#with h5.File('train_data.h5','r') as f:
#    train_data=f['data'][:]
#print('loading labels')
#with h5.File('train_label.h5','r') as f:
#    train_label=f['label'][:]
#print('done loading')
# class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]

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

autoencoder = models.Sequential()
# Add a noise layer to get a denoising autoencoder. This helps avoid overfitting
autoencoder.add(Layer(input_shape=(3,360, 480)))

#autoencoder.add(GaussianNoise(sigma=0.3))
autoencoder.encoding_layers = create_encoding_layers()
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
autoencoder.add(Reshape((num_classes,data_shape)))#, input_shape=(num_classes,360,480)))
autoencoder.add(Permute((2, 1)))
autoencoder.add(Activation('softmax'))
#from keras.optimizers import SGD
#optimizer = SGD(lr=0.01, momentum=0.8, decay=0., nesterov=False)
autoencoder.compile(loss="categorical_crossentropy", optimizer='adadelta',metrics=['accuracy'])
autoencoder.load_weights('model_weight_ep450.hdf5')

#current_dir = os.path.dirname(os.path.realpath(__file__))
#model_path = os.path.join(current_dir, "autoencoder.png")
#plot(model_path, to_file=model_path, show_shapes=True)

nb_epoch =  50
batch_size = 5

history = autoencoder.fit_generator(batch_data_gen(batch_size), 500, nb_epoch=nb_epoch)#,
                    #show_accuracy=True)#, class_weight=class_weighting )#, validation_data=(X_test, X_test))

autoencoder.save_weights('model_weight_ep600.hdf5')
