import os
import time
import random
import pathlib
import itertools
from glob import glob
from tqdm import tqdm_notebook, tnrange

# Import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
# %matplotlib inline
from skimage.color import rgb2gray
from skimage.morphology import label
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow, concatenate_images

# Import deep learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

print('Modules loaded')

# Function to create dataframe
def create_df(data_dir):
    images_paths = []
    masks_paths = []

    # Traverse through the directories
    for case_dir in glob(f"{data_dir}/*"):
        image_files = glob(f"{case_dir}/*.tif")
        for img_path in image_files:
            if "_mask" in img_path:
                masks_paths.append(img_path)
            else:
                images_paths.append(img_path)

    # Ensure images and masks are paired correctly
    images_paths = sorted(images_paths)
    masks_paths = sorted(masks_paths)

    df = pd.DataFrame(data={'images_paths': images_paths, 'masks_paths': masks_paths})

    return df

# Function to split dataframe into train, valid, test
def split_df(df):
    train_df, dummy_df = train_test_split(df, train_size=0.8, random_state=42)
    valid_df, test_df = train_test_split(dummy_df, train_size=0.5, random_state=42)
    return train_df, valid_df, test_df

# Function to create data generators
def create_gens(df, aug_dict):
    img_size = (128, 128)
    batch_size = 16

    img_gen = ImageDataGenerator(**aug_dict)
    msk_gen = ImageDataGenerator(**aug_dict)

    image_gen = img_gen.flow_from_dataframe(
        df,
        x_col='images_paths',
        class_mode=None,
        color_mode='rgb',
        target_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=1
    )

    mask_gen = msk_gen.flow_from_dataframe(
        df,
        x_col='masks_paths',
        class_mode=None,
        color_mode='grayscale',
        target_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=1
    )

    gen = zip(image_gen, mask_gen)

    for (img, msk) in gen:
        img = img / 255.0
        msk = msk / 255.0
        msk[msk > 0.5] = 1
        msk[msk <= 0.5] = 0

        yield (img, msk)

# Function to define U-Net model
def unet(input_size=(128, 128, 3)):
    inputs = Input(input_size)

    # Down-sampling
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Up-sampling
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Function to calculate dice coefficient
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# Directory setup
data_dir = 'dataset\lgg-mri-segmentation\kaggle_3m'
df = create_df(data_dir)
train_df, valid_df, test_df = split_df(df)

# Data augmentation settings
tr_aug_dict = dict(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generators
train_gen = create_gens(train_df, tr_aug_dict)
valid_gen = create_gens(valid_df, {})
test_gen = create_gens(test_df, {})

# Model setup
model = unet()
model.compile(optimizer=Adam(learning_rate=0.0001), loss=dice_loss, metrics=[dice_coef])

# Training
# callbacks = [ModelCheckpoint('best_model.h5', save_best_only=True, verbose=1)]
callbacks = [ModelCheckpoint('best_model.keras', save_best_only=True, verbose=1)]

history = model.fit(
    train_gen,
    steps_per_epoch=len(train_df) // 16,
    validation_data=valid_gen,
    validation_steps=len(valid_df) // 16,
    epochs=20,
    callbacks=callbacks
)

# Evaluate
print("Evaluation on test data:")
test_loss, test_dice = model.evaluate(test_gen)
print(f"Test Loss: {test_loss}, Test Dice Coefficient: {test_dice}")
