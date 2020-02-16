import pathlib
import glob
import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def clean_ipynb_folder_if_exists(folder_path):
    folder = pathlib.Path(folder)
    ipynb_paths = [str(item) for item in folder.glob('**/*') if item.is_dir() and item.name.startswith('.ipynb')]
    if len(ipynb_paths) > 0:
        for eachdir in ipynb_paths:
            shutil.rmtree(eachdir)
            print("Removed", eachdir)
    else:
        print('No .ipynb_checkpoints to remove')

def get_path(folder_path):
    path = pathlib.Path(folder_path, '')

    all_file_path = path.glob('**/*')
    all_file_path = list(all_file_path)
    all_file_path = [str(path) for path in all_file_path]
    all_file_path = [m for m in all_file_path if m.lower().endswith(('.png', '.jpg', '.jpeg'))]
    clean_ipynb_folder_if_exists(folder_path)
    
    return all_file_path

def load_label(folder_path):
    labels = json.load(open(os.path.join(folder_path, 'labels.json')))
    return labels

def clean_label(all_file_path, all_labels):
    remove_path = []
    for i in all_file_path:
        key = i.split('/')[-1]
        if key not in all_labels:
            remove_path.append(i)
    
    return remove_path

def labelling(folder_path):
    all_img_labels = []
    all_labels = get_label(folder_path)
    all_file_path = get_path(folder_path)
    for i in all_file_path:
        key = i.split('/')[-1]
        if key in all_labels:
            all_img_labels.append(all_labels[key])
    
    return all_img_labels

def tokenize(folder_path):
    res = []
    
    all_img_labels = labelling(folder_path)
    for label in all_img_labels:
        for word in label:
            if word not in res:
                res.append(word)
    return res

def charset(train_dir, test_dir):

    train_res = tokenize(train_dir)
    test_res = tokenize(test_dir)
  
    for i in test_res:
        if i not in train_res:
            train_res.append(i)

    charset = ''.join(sorted(train_res))  
    return charset

def encode(train_dir, test_dir):  
    res = []
    
    char = charset(train_dir, test_dir)
    char2text = {values: key for key, values in enumerate(char)}
    train_labels = labelling(train_dir)
    test_labels = labelling(test_dir)
    train_labels.extend(test_labels)
    text = [i for i in train_labels]
    for line in text:
        for word in line:
            encode = char2text.get(word)
            res.append(encode)
 
    return np.asarray(res, dtype=np.int32)

def get_label_from_dir(new_folder, old_folder):
    path_img = get_path(new_folder)
    old_label = load_label(old_folder)
    new_label = {}
    for i in path_img:
        label = (i.split('/'))[-1]
        label = old_label[label]
        new_label[i.split('/')[-1]] = label
    
    return new_label