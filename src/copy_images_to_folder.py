import os
from shutil import copyfile
from skimage.io import imread, imsave, imread_collection
import numpy as np
from helper import *

stage1_train_path = os.path.join('..', 'input', 'stage1_train')
dest_path = os.path.join('..', 'input', 'stage1_train_images')
if not os.path.exists(dest_path):
    os.makedirs(dest_path)

img_ids = os.listdir(stage1_train_path)
excluded_img_ids = ['7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80']
img_ids = [x for x in img_ids if not (x.startswith('.') or x in excluded_img_ids)]

for img_id in img_ids:
    image = imread(os.path.join(stage1_train_path, img_id, 'images', img_id + ".png"))[:, :, :3]
    image = normalize_image(image)
    masks = imread_collection(os.path.join(stage1_train_path, img_id, 'masks', '*.png')).concatenate()
    overlay_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.bool)
    for mask in masks:
        # mask = np.expand_dims(mask, axis=-1)
        overlay_mask = np.maximum(mask, overlay_mask)

    imsave(os.path.join(dest_path, img_id + '.png'), image)
    imsave(os.path.join(dest_path, img_id + '_mask.png'), overlay_mask)
