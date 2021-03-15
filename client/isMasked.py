import os
import zipfile
import random
import shutil
import time
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd
from os import listdir
import cv2
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import imutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg

#CNN model 정의
model_mask = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5,5), activation='relu', input_shape=(112, 112, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),    
    tf.keras.layers.Dense(2, activation='softmax')
])
model_mask.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model_mask.load_weights("model_weight.h5")

def isMasked(face_img, Threshold):
    face_img=cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    normalized=face_img/255.0
    reshaped=np.reshape(normalized,(1,112,112,3))
    reshaped = np.vstack([reshaped])
    result=model_mask.predict(reshaped)
    print(result[0][1])
    
    if result[0][1] >= Threshold:
        return True
    else:
        return False