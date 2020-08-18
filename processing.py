import os
import sys
import json
import warnings
import argparse
import subprocess
import numpy as np
import pandas as pd
from glob import glob
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.exceptions import DataConversionWarning
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.utils.np_utils import to_categorical 
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
LOCAL_DATA_PATH = "/opt/ml/processing" 

def load_images(path):
    """Load images from local and resize them"""
    
    # ---- Read the condition's image and append
    img = image.load_img(path, target_size=(32, 32))
    
    # ---- Covert to array, and "preprocess it" so that keras model can read it
    img_array = image.img_to_array(img)
    
    return img_array


def _load_train_dataset(base_dir):
    """Load images from local, create dataframe and split in train and test"""
    
    # Read labels csv (labels + image metadata/paths)
    df = pd.read_csv(os.path.join(base_dir, "labels.csv"))

    # Add path of each image as a new column
    image_dict = {}
    for x in glob(os.path.join(base_dir, "images", "*.jpg")):
        image_dict[os.path.splitext(os.path.basename(x))[0]] = x

    df['path'] = df['image_id'].map(image_dict.get)
    
    # Pre-process labels: full name + categorize
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    df['cell_type'] = df['dx'].map(lesion_type_dict.get)
    df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes
    
    
    # Load images into the df
    df['image'] = df['path'].map(lambda x: load_images(x))
    
    # Train/Test split
    features = df['image']
    target = df['cell_type_idx']
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Create class weights
    cls_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

    # Convert images and the label vectors into numpy arrays    
    y_train = to_categorical(y_train, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)
    x_train = np.asarray(x_train.tolist()).reshape((x_train.shape[0],*(32,32,3)))
    x_test = np.asarray(x_test.tolist()).reshape((x_test.shape[0],*(32,32,3)))
    
    return x_train, y_train, x_test, y_test, cls_weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
    args, _ = parser.parse_known_args()
    
    base_dir = os.path.join(LOCAL_DATA_PATH, 'input/')
    x_train, y_train, x_test, y_test, cls_weight = _load_train_dataset(base_dir)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    
    np.save(os.path.join(LOCAL_DATA_PATH, "train/cls_weight.npy"), cls_weight)
    np.save(os.path.join(LOCAL_DATA_PATH, "train/x_train.npy"), x_train)
    np.save(os.path.join(LOCAL_DATA_PATH, "train/y_train.npy"), y_train)
    np.save(os.path.join(LOCAL_DATA_PATH, "test/x_test.npy"), x_test)
    np.save(os.path.join(LOCAL_DATA_PATH, "test/y_test.npy"), y_test)