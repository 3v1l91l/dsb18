from skimage.morphology import label
import numpy as np
import cv2

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def normalize_image(in_rgb_img):
    grid_size = 8
    bgr = in_rgb_img[:, :, [2, 1, 0]]  # flip r and b
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size, grid_size))
    image = clahe.apply(lab[:, :, 0])
    if image.mean()>127:
        image = 255-image
    return image[...,np.newaxis]
