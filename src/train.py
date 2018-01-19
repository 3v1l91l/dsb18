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

seed = 42
random.seed = seed
np.random.seed(seed)

IMG_WIDTH = IMG_HEIGHT = 256
IMG_CHANNELS = 3
BATCH_SIZE = 8

def combine_generator(gen1, gen2):
    while True:
        yield(gen1.next(), gen2.next())

def train():

    data = np.load('data.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    X_valid = data['X_valid']
    y_valid = data['y_valid']

    data_gen_args = dict(rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='wrap')
    image_datagen_train = ImageDataGenerator(**data_gen_args)
    mask_datagen_train = ImageDataGenerator(**data_gen_args)
    image_datagen_valid = ImageDataGenerator(**data_gen_args)
    mask_datagen_valid = ImageDataGenerator(**data_gen_args)
    image_datagen_train.fit(y_train, augment=True, seed=seed)
    mask_datagen_train.fit(y_train, augment=True, seed=seed)
    image_datagen_valid.fit(y_train, augment=True, seed=seed)
    mask_datagen_valid.fit(y_train, augment=True, seed=seed)

    image_generator_train = image_datagen_train.flow(
        X_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed)
    mask_generator_train = mask_datagen_train.flow(
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed)
    image_generator_valid = image_datagen_valid.flow(
        X_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed)
    mask_generator_valid = mask_datagen_valid.flow(
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed)
    train_generator = combine_generator(image_generator_train, mask_generator_train)
    valid_generator = combine_generator(image_generator_valid, mask_generator_valid)

    model = get_unet_model()
    model.fit_generator(
        generator=train_generator,
        validation_data=valid_generator,
        steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
        validation_steps=X_valid.shape[0] // BATCH_SIZE,
        epochs=100,
        callbacks=get_callbacks())

    # results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=100,
    #                     callbacks=get_callbacks())

    # model.fit_generator(
    #     train_datagen.flow(x_train, y_train, batch_size=batch_size),
    #     steps_per_epoch=x_train.shape[0] // batch_size,
    #     epochs=100,
    #     callbacks=get_callbacks(),
    #     validation_steps=x_valid.shape[0] // batch_size,
    #     validation_data=validation_datagen.flow(x_valid, y_valid, batch_size=batch_size)
    # )


    # preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
    # preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)


def make_submission():
    stage1_test_path = os.path.join('..', 'input', 'stage1_test')
    X_test = X_test, sizes_test = sizes_test, img_ids_test = img_ids_test

    model = get_unet_model()
    model.load_weights('model_weights.h5')
    # model = load_model('model.h5', custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})
    X_test, sizes_test, img_ids_test = test_data(stage1_test_path)
    preds_test = model.predict(X_test, verbose=1)

    # Threshold predictions
    preds_test_t = (preds_test > 0.5).astype(np.uint8)
    ix = random.randint(0, len(preds_test_t))

    # plt.figure(1)
    # plt.subplot(211)
    # plt.imshow(X_test[ix])
    # plt.subplot(212)
    # plt.imshow(np.squeeze(preds_test_t[ix]))
    # plt.show()
    # Create list of upsampled test masks
    preds_test_upsampled = []
    for i in range(len(preds_test_t)):
        preds_test_upsampled.append(resize(np.squeeze(preds_test_t[i]),
                                           (sizes_test[i][0], sizes_test[i][1]),
                                           mode='constant', preserve_range=True))

    new_test_ids = []
    rles = []
    for n, id_ in enumerate(img_ids_test):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('sub.csv', index=False)

if __name__ == '__main__':
    train()
    # make_submission()
