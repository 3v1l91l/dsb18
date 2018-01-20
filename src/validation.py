import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import random
from skimage import morphology, transform
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import skimage
from skimage.io import imread, imread_collection
from skimage.segmentation import relabel_sequential
from skimage.morphology import label
import cv2
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from model import *

IMG_HEIGHT = IMG_WIDTH = 256

def validate():
    data = np.load('data.npz')
    img_ids_valid = data['img_ids_valid']
    model = load_model('model.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

    prec = []
    for id in img_ids_valid:
        file = "../input/stage1_train/{}/images/{}.png".format(id, id)
        masks = "../input/stage1_train/{}/masks/*.png".format(id)
        image = imread(file)
        image = transform.resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        image = image.astype(np.uint8)
        masks = imread_collection(masks).concatenate()
        masks = np.array([transform.resize(x, (IMG_HEIGHT, IMG_WIDTH), mode='constant') for x in masks])
        labels = get_ground_truth_labels(masks)
        # image = normalize_image(image)
        # image = image / 255
        seg_prec = calculate_iou(image, labels, model)
        prec.append(seg_prec)
        if seg_prec< 0.3:
            print('img id: {}, prec: {}'.format(id, seg_prec))

    print('MEAN IOU: {}'.format(np.mean(np.array(prec))))

def get_ground_truth_labels(masks):
    height, width = masks[0].shape
    num_masks = masks.shape[0]

    # Make a ground truth label image (pixel value is index of object label)
    labels = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labels[masks[index] > 0] = index + 1
    return labels

def calculate_iou(image, labels, model):
    img = image[:,:,:3]
    img = img[np.newaxis,:,:]
    y_pred = model.predict(img/255, verbose=0)
    y_pred = np.squeeze(y_pred, axis=0)
    y_pred = np.squeeze(y_pred, axis=2)
    cutoff = 0.5
    y_pred_label = label(y_pred > cutoff)
    y_pred_label, _, inverse_map = skimage.segmentation.relabel_sequential(y_pred_label) # Relabel objects

    # fig = plt.figure(figsize=(10,10))
    # plt.imshow(np.squeeze(image))
    # plt.title("Original image")
    # fig = plt.figure(figsize=(10,10))
    # plt.imshow(labels)
    # plt.title("Ground truth masks")
    # fig = plt.figure(figsize=(10,10))
    # plt.imshow(y_pred)
    # plt.title("y_pred")
    # fig = plt.figure(figsize=(10,10))
    # plt.imshow(y_pred_label)
    # plt.title("y_pred_label")

    # fig = plt.figure(figsize=(10,10))
    # plt.hist(y_pred)
    # plt.title("hist y_pred")
    # image_ = y_pred > 0.5 #0.9999999
    # image_ = np.squeeze(image_).astype(np.uint8)
    # distance = ndi.distance_transform_edt(image_)
    # local_maxi = peak_local_max(distance, indices=False, labels=image_, min_distance=5)#, footprint=np.ones((3, 3)))
    # markers = ndi.label(local_maxi)[0]
    # watershed_labels = watershed(-distance, markers, mask=image_)
    # fig = plt.figure(figsize=(10,10))
    # plt.imshow(distance)
    # plt.title("distance")
    # fig = plt.figure(figsize=(10,10))
    # plt.imshow(watershed_labels)
    # plt.title("labels watershed")
    # # print(np.unique(watershed_labels))
    # y_pred = watershed_labels
    y_pred = y_pred_label

    # Compute number of objects
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))
    print("Number of true objects:", true_objects)
    print("Number of predicted objects:", pred_objects)

    # Compute intersection between all objects
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    # print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp / (tp + fp + fn)
        # print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    # print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

if __name__== '__main__':
    validate()