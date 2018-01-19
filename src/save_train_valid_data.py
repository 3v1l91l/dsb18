from sklearn.model_selection import train_test_split
import numpy as np
import os
import skimage
from skimage.io import imread, imshow, imread_collection, concatenate_images
from tqdm import tqdm
from skimage import transform

IMG_WIDTH = IMG_HEIGHT = 256
IMG_CHANNELS = 3


def _get_train_data(train_dir):
    excluded_img_ids = ['7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80']
    img_ids = os.listdir(train_dir)
    img_ids = [x for x in img_ids if not (x.startswith('.') or x in excluded_img_ids)]
    X = np.zeros((len(img_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    y = np.zeros((len(img_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    # X = Y = []
    for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
        image = imread(os.path.join(train_dir, img_id, 'images', img_id +".png"))[:,:,:IMG_CHANNELS]
        masks = imread_collection(os.path.join(train_dir, img_id, 'masks', '*.png')).concatenate()
        image = transform.resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X[i] = image
        # X.append(image)

        overlay_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        # overlay_mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.bool)

        for mask in masks:
            mask = np.expand_dims(mask, axis=-1)
            overlay_mask = np.maximum(transform.resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), overlay_mask)
        y[i] = overlay_mask
        # Y.append(overlay_mask)
    # X = np.array(X)
    # Y = np.array(Y)
    X = X / 255

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)
    return X_train, X_valid, y_train, y_valid


def _get_test_data(root_dir):
    img_ids = os.listdir(root_dir)
    img_ids = [x for x in img_ids if not x.startswith('.')]

    X = np.zeros((len(img_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes = []
    for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
        image = skimage.io.imread(os.path.join(root_dir, img_id, 'images', img_id + ".png"))[:, :, :IMG_CHANNELS]
        sizes.append([image.shape[0], image.shape[1]])
        image = transform.resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X[i] = image

    # images = []
    # for img_id in img_ids:
    #     image = skimage.io.imread(os.path.join(root_dir, img_id, 'images', img_id + ".png"))
    #     images.append(image)
    # train_img_df = pd.DataFrame({'images': images})
    # print(train_img_df['images'].map(lambda x: x.shape).value_counts())
    X = X / 255

    return X, sizes, img_ids

def save_data():
    stage1_train_path = os.path.join('..', 'input', 'stage1_train')
    stage1_test_path = os.path.join('..', 'input', 'stage1_test')

    X_train, X_valid, y_train, y_valid = _get_train_data(stage1_train_path)
    X_test, sizes_test, img_ids_test = _get_test_data(stage1_test_path)
    np.savez('data.npz', X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,
             X_test=X_test, sizes_test=sizes_test, img_ids_test=img_ids_test)


if __name__ == '__main__':
    save_data()