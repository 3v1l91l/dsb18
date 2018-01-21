from skimage import transform
from helper import *
from keras.models import load_model
import numpy as np
from model import *
from helper import *
import pandas as pd

def make_submission():
    data = np.load('data.npz')
    X_test = data['X_test']
    sizes_test = data['sizes_test']
    img_ids_test = data['img_ids_test']

    model = load_model('model.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    preds_test = model.predict(X_test/255, verbose=1)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)

    preds_test_upsampled = []
    for i in range(len(preds_test_t)):
        preds_test_upsampled.append(transform.resize(np.squeeze(preds_test_t[i]),
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
    make_submission()
