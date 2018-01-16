import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from glob import glob
from pathlib import Path
import skimage.io
import matplotlib.pyplot as plt
import skimage.segmentation
from model import *
from skimage import transform
from keras.callbacks import ModelCheckpoint
import pandas as pd

IMG_WIDTH = 256

def img_resize(img):
    return transform.resize(img, (IMG_WIDTH, IMG_WIDTH), preserve_range=True)

def load_data(root_dir):
    img_ids = os.listdir(root_dir)[:100]
    img_ids = [x for x in img_ids if not x.startswith('.')]
    images = masks = []
    for img_id in img_ids:
        image = skimage.io.imread(os.path.join(root_dir, img_id, 'images', img_id +".png"))
        mask = skimage.io.imread_collection(os.path.join(root_dir, img_id, 'masks', '*.png')).concatenate()
        # if image.shape[0] != IMG_WIDTH:
        #     image = img_resize(image)
        #     mask = list(map(img_resize, mask))
        overlay_mask = np.zeros((image.shape[0], image.shape[1]), np.bool)
        for index in range(len(mask)):
            overlay_mask[mask[index]>0] = True
        images.append(image)
        masks.append(overlay_mask)
    return images, masks

def test_data(root_dir):
    img_ids = os.listdir(root_dir)
    img_ids = [x for x in img_ids if not x.startswith('.')]
    images = []
    for img_id in img_ids:
        image = skimage.io.imread(os.path.join(root_dir, img_id, 'images', img_id + ".png"))
        images.append(image)
    train_img_df = pd.DataFrame({'images': images})
    print(train_img_df['images'].map(lambda x: x.shape).value_counts())


def train():
    stage1_train_path = os.path.join('..', 'input', 'stage1_train')
    stage1_test_path = os.path.join('..', 'input', 'stage1_test')
    # test_data(stage1_test_path)
    imgs_train, imgs_mask_train = load_data(stage1_train_path)

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])
    # model.load_weights('weights.h5')

if __name__ == '__main__':
    train()
