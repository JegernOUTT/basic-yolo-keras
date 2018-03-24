import os
import cv2
import csv
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

from preprocessing import *


ann_path = '/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/Downloads/lcrowdw/shop_var/bright/p_hd_hpc_hp/png/'
with open(os.path.join(ann_path, 'annotations.pickle'), 'rb') as f:
    data = pickle.load(f)


generator_config = {
    'IMAGE_H'         : 608,
    'IMAGE_W'         : 608,
    'GRID_H'          : 20,
    'GRID_W'          : 20,
    'BOX'             : 5,
    'LABELS'          : ['person'],
    'CLASS'           : 1,
    'ANCHORS'         : [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
    'BATCH_SIZE'      : 16,
    'TRUE_BOX_BUFFER' : 50,
}

train_imgs = load_queue_dataset()
lcrowdl = load_lcrowdl()
prw = load_prw_dataset()
np.random.shuffle(lcrowdl)
np.random.shuffle(prw)
train_imgs += lcrowdl[:600]
train_imgs += prw[:600]


batches = BatchGenerator(train_imgs, generator_config, jitter=True)


def imshow_grid(data, height=None, width=None, normalize=False, padsize=1, padval=0):
    '''
    Take an array of shape (N, H, W) or (N, H, W, C)
    and visualize each (H, W) image in a grid style (height x width).
    '''
    if normalize:
        data -= data.min()
        data /= data.max()

    N = data.shape[0]
    if height is None:
        if width is None:
            height = int(np.ceil(np.sqrt(N)))
        else:
            height = int(np.ceil( N / float(width) ))

    if width is None:
        width = int(np.ceil( N / float(height) ))

    assert height * width >= N

    # append padding
    padding = ((0, (width*height) - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((height, width) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((height * data.shape[1], width * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.show()

plt.rcParams['figure.figsize'] = (15, 15)

for i in range(10):
    imshow_grid(batches[i][0][0], normalize=True)
