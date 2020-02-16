import cv2

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid, softmax
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

import h5py
import unicodedata
import json

# OUR FULL MODEL OF CRNN AND LSTM

# YOUR PART: TRY TO BUILD YOUR DIFFERENT TIMESTEPS AND BUILD TO DO IT, LET SAY 26 OR 32 INSTEAD OF 31 LIKE BELOW. 
def base_model(train_dir, test_dir):
    char_array = charset(train_dir, test_dir)
    # input with shape of height=118 and width=1875 
    inputs = Input(shape=(120,1900,1))   # (118,1875,3)

    # convolution layer with kernel size (3,3)
    conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)  # (120,1900,64)

    # poolig layer with kernel size (2,2) to make the height/2 and width/2  # (60,950,64)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

    conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)  # (60,950,128)

    # poolig layer with kernel size (2,2) to make the height/2 and width/2   # (30,475,128)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

    conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)   # (30,475,256)

    conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)   # (30,475,256)

    # poolig layer with kernel size (2,2) to make the height/2              # (15,95,256)
    pool_4 = MaxPool2D(pool_size=(2, 5))(conv_4)

    conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)  # (15,95,512)

    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(conv_5)

    conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)  # (15,95,512)
    batch_norm_6 = BatchNormalization()(conv_6)

    # poolig layer with kernel size (2,2) to make the height/2       # (15,95,512)
    pool_6 = MaxPool2D(pool_size=(5, 1))(batch_norm_6)

    # didn't have padding = "same" to reduce timestep - 1 because kernel size is # (1,95,512)
    conv_7 = Conv2D(512, (3,1), activation = 'relu')(pool_6)

    # to remove the first dimension of one: (1, 95, 512) to (95, 512) 
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

    # bidirectional LSTM layers with units=95
    blstm_1 = Bidirectional(LSTM(95, return_sequences=True, dropout = 0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(95, return_sequences=True, dropout = 0.2))(blstm_1)

    # this is our softmax character proprobility with timesteps (31, 63)
    outputs = Dense(len(char_array)+1, activation = 'softmax')(blstm_2)

    # model to be used at test time

    base_model = Model(inputs, outputs)
    
    return base_model 