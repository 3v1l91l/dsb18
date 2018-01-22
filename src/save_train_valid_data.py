from sklearn.model_selection import train_test_split
import numpy as np
import os
import skimage
from skimage.io import imread, imshow, imread_collection, concatenate_images
from tqdm import tqdm
from skimage import transform
from helper import *
import matplotlib.pyplot as plt
import cv2

VALIDATION_SIZE = 0.2
# IMG_WIDTH = IMG_HEIGHT = 512
IMG_WIDTH = IMG_HEIGHT = 256

grid_size = 8
FILE_IMG_CHANNELS = 3

def _get_train_data(train_dir, normalize):
    excluded_img_ids = ['7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80']
    img_ids = os.listdir(train_dir)
    img_ids = [x for x in img_ids if not (x.startswith('.') or x in excluded_img_ids)]
    img_channels = 3
    if normalize:
        img_channels = 1

    X = np.zeros((len(img_ids), IMG_HEIGHT, IMG_WIDTH, img_channels), dtype=np.uint8)
    y = np.zeros((len(img_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    # X = Y = []
    for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
        image = imread(os.path.join(train_dir, img_id, 'images', img_id +".png"))[:,:,:FILE_IMG_CHANNELS]
        masks = imread_collection(os.path.join(train_dir, img_id, 'masks', '*.png')).concatenate()
        # fig = plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        image = transform.resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        image = image.astype(np.uint8)
        if normalize:
            image = normalize_image(image)

        # fig = plt.figure(figsize=(10, 10))
        # plt.imshow(np.squeeze(image))
        X[i] = image
        # X.append(image)

        overlay_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        # overlay_mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.bool)

        for mask in masks:
            mask[mask == 255] = 1
            mask = mask.astype(np.uint8)
            _, contours, hierarchy = cv2.findContours(mask.copy(),
                                                      cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
                                                      offset=(0, 0))
            mask_with_contours = cv2.drawContours(mask.copy(), contours, -1, (2, 0, 0))
            # mask = np.expand_dims(mask, axis=-1)
            # mask = transform.resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            mask_with_contours = np.expand_dims(mask_with_contours, axis=-1)
            mask_with_contours = transform.resize(mask_with_contours, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

            # fig = plt.figure()
            # plt.subplot(1,2,1)
            # plt.imshow(np.squeeze(mask))
            # plt.subplot(1,2,2)
            # plt.imshow(np.squeeze(mask_with_contours))

            overlay_mask = np.maximum(mask_with_contours, overlay_mask)

        # fig = plt.figure()
        # plt.imshow(np.squeeze(overlay_mask))
        y[i] = overlay_mask
        # Y.append(overlay_mask)
    # X = np.array(X)
    # Y = np.array(Y)
    # X = X / 255

    X_train, X_valid, y_train, y_valid, img_ids_train, img_ids_valid = train_test_split(X, y, img_ids, test_size=VALIDATION_SIZE)
    return X_train, X_valid, y_train, y_valid, img_ids_train, img_ids_valid


def _get_test_data(root_dir, normalize):
    img_ids = os.listdir(root_dir)
    img_ids = [x for x in img_ids if not x.startswith('.')]
    img_channels = 3
    if normalize:
        img_channels = 1
    X = np.zeros((len(img_ids), IMG_HEIGHT, IMG_WIDTH, img_channels), dtype=np.uint8)
    sizes = []
    X_orig = []
    for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
        image = skimage.io.imread(os.path.join(root_dir, img_id, 'images', img_id + ".png"))[:, :, :FILE_IMG_CHANNELS]
        X_orig.append(image)
        sizes.append([image.shape[0], image.shape[1]])
        image = transform.resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        image = image.astype(np.uint8)
        if normalize:
            image = normalize_image(image)
        X[i] = image


    # images = []
    # for img_id in img_ids:
    #     image = skimage.io.imread(os.path.join(root_dir, img_id, 'images', img_id + ".png"))
    #     images.append(image)
    # train_img_df = pd.DataFrame({'images': images})
    # print(train_img_df['images'].map(lambda x: x.shape).value_counts())
    # X = X / 255

    return X, sizes, img_ids, X_orig

def save_data():
    stage1_train_path = os.path.join('..', 'input', 'stage1_train')
    stage1_test_path = os.path.join('..', 'input', 'stage1_test')

    X_train, X_valid, y_train, y_valid, img_ids_train, img_ids_valid = _get_train_data(stage1_train_path, normalize=True)
    X_test, sizes_test, img_ids_test, X_test_orig = _get_test_data(stage1_test_path, normalize=True)
    np.savez('data.npz', X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,
             img_ids_train=img_ids_train, img_ids_valid=img_ids_valid,
             X_test=X_test, sizes_test=sizes_test, img_ids_test=img_ids_test, X_test_orig=X_test_orig)

if __name__ == '__main__':
    save_data()