import cv2

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences

import h5py
import unicodedata
import json

import pathlib
import glob
import os
import shutil

def preprocess_first(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11,11), 0)
    edged = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    edged = cv2.resize(edged, dsize=(1900, 120))
    
    return edged

def pre_for_model(new_folder, old_folder):
    all_path = get_path(new_folder)
    all_label = get_label_from_dir(new_folder, old_folder)
    label_list = []
    image_list = []
    
    for path in all_path:
        label = (path.split('/'))[-1]
        label = all_label[label]
        label_list.append(label)
        
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
    #     image = tf.image.convert_image_dtype(image, tf.float32)
    #     image = tf.image.per_image_standardization(image)
        image = image[:,:,1]
        image = np.expand_dims(image, axis=2)
        image_list.append(np.array(image))
    
    return image_list, label_list