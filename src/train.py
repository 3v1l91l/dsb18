import os
from skimage.io import imsave
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import skimage.segmentation
from model import *
import pandas as pd
from keras.models import load_model
import random
from helper import *
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from skimage import transform
from imgaug import augmenters as iaa
from generators import *
import imgaug as ia
from zf import ZF_UNET_224
from keras.utils import to_categorical

seed = 42
random.seed = seed
np.random.seed(seed)

BATCH_SIZE = 8
def activator_masks(images, augmenter, parents, default):
    if augmenter.name in ["GaussianBlur", "Dropout"]:
        return False
    else:
        # default value for all other augmenters
        return default

def train():
    data = np.load('data.npz')
    X_train = data['X_train']
    y_train = data['y_train'].astype(np.float64)
    X_valid = data['X_valid']
    y_valid = data['y_valid'].astype(np.float64)
    X_train = X_train / 255
    X_valid = X_valid / 255
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_valid_cat = to_categorical(y_valid, num_classes=3)

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential([
        iaa.Fliplr(0.2),
        iaa.Flipud(0.2),
        # iaa.GaussianBlur((0, 2.0), name="GaussianBlur"),
        # iaa.PiecewiseAffine(scale=(0.01, 0.05)),
        # iaa.Affine(shear=(-16, 16))

        # sometimes(iaa.Affine(
        #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #     rotate=(-90, 90),  # rotate by -45 to +45 degrees
        #     shear=(-20, 20),  # shear by -16 to +16 degrees
        #     cval=0,  # if mode is constant, use a cval between 0 and 255
        #     mode='constant'  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        # ))
    ])
    hooks_masks= ia.HooksImages(activator=activator_masks)
    seq = seq.to_deterministic()

    # data_gen_args = dict(rotation_range=10.,
    #                      width_shift_range=0.1,
    #                      height_shift_range=0.1,
    #                      zoom_range=0.2,
    #                      horizontal_flip=True,
    #                      vertical_flip=True,
    #                      fill_mode='constant')
    # image_datagen_train = ImageDataGenerator(**data_gen_args)
    # mask_datagen_train = ImageDataGenerator(**data_gen_args)
    # image_datagen_valid = ImageDataGenerator(**data_gen_args)
    # mask_datagen_valid = ImageDataGenerator(**data_gen_args)
    # image_datagen_train.fit(y_train, augment=True, seed=seed)
    # mask_datagen_train.fit(y_train, augment=True, seed=seed)
    # image_datagen_valid.fit(y_train, augment=True, seed=seed)
    # mask_datagen_valid.fit(y_train, augment=True, seed=seed)
    #
    # image_generator_train = image_datagen_train.flow(
    #     X_train,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     seed=seed)
    # mask_generator_train = mask_datagen_train.flow(
    #     y_train,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     seed=seed)
    # image_generator_valid = image_datagen_valid.flow(
    #     X_train,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     seed=seed)
    # mask_generator_valid = mask_datagen_valid.flow(
    #     y_train,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     seed=seed)
    # train_generator = combine_generator(image_generator_train, mask_generator_train)
    # valid_generator = combine_generator(image_generator_valid, mask_generator_valid)
    #
    # model = get_unet_model()
    # # model.load_weights('model_weights.h5')
    # model.fit_generator(
    #     generator=train_generator,
    #     validation_data=valid_generator,
    #     steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
    #     validation_steps=X_valid.shape[0] // BATCH_SIZE,
    #     epochs=100,
    #     workers=4,
    #     callbacks=get_callbacks())

    model = get_unet_model(3)
    # model.load_weights('model.h5')
    # model = load_model('model.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    model.fit_generator(
        generator=generator(seq, X_train, y_train_cat, BATCH_SIZE),
        validation_data=generator(seq, X_valid, y_valid_cat, BATCH_SIZE, hooks=hooks_masks),
        steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
        validation_steps=X_valid.shape[0] // BATCH_SIZE,
        epochs=100,
        workers=4,
        callbacks=get_callbacks())

if __name__ == '__main__':
    train()
