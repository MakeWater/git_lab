# -*- coding:utf-8 -*-
import numpy as np
from keras.layers import Input,Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model


# MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

data = np.load('coil_20.npy')
x_train = data[:len(data)/2] # just choose 10 class randomly from 20 objects.
x_test = x_train
print('x_train ''s shape is',x_train.shape)#############修改处


image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# network parameters(784,512,2048,10);(16384,4096,32768,10)
input_shape = (original_dim, ) # 784 or 16384
denseLayer1_dim = 512
denseLayer2_dim = 2048
latent_dim = 10
decoder_dim = 784
epochs = 50
batch_size = 32

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(denseLayer1_dim, activation='relu',name='encode_layer1')(inputs)
x = Dense(denseLayer2_dim,activation='relu'encode_layer2)(x)
encoded = Dense(latent_dim,activation='relu',name='encod_layer')(x)
x = Dense(denseLayer2_dim, activation='relu',name=decod_layer1)(encoded)
x = Dense(denseLayer1_dim, activation='relu',name='decod_layer2')(x)
decoded = Dense(decoder_dim,activation='sigmoid',name='decod_layer')(x) ################ 解码维度修改

# instantiate decoder model
encoder = Model(inputs, encoded, name='encoder')
autoencoder = Model(inputs, decoded, name='decoder')

autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
autoencoder.fit(x_train,x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test,x_test))
autoencoder.save_weights('autoencoder_embed.h5')
embeded = encoder.predict(x_train) ###########################修改处
np.save('embed_mnist.npy',embeded)