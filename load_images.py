import os
import cv2
import torch
import math
from tqdm import tqdm


def load_images(path, scale):

    # find all files in directory
    files = os.listdir(path)
    N = len(files)


    # allocate memory
    image = cv2.imread(path + '/' + files[0], 1)
    sz = image.shape
    r = math.floor(sz[0] * scale)
    c = math.floor(sz[1] * scale)

    I = torch.zeros(N, 3, r, c)

    # read all files
    for i in tqdm(range(N)):

        # load image
        filename = os.path.join(path, files[i])
        im = cv2.imread(filename, 1) / 255.
        im = torch.tensor(im)
        im = im.permute(2, 0, 1)

        # optional downsampling step
        if (scale < 1):
            im = cv2.resize(im, (r, c), interpolation=cv2.INTER_CUBIC)
            im = torch.tensor(im)

        I[i, :, :, :] = im
    return I