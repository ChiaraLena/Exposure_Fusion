import argparse
import kornia
import numpy as np
import cv2
import os

import exposure_fusion as ef
import load_images as li

parser = argparse.ArgumentParser(description='Exposure fusion')

parser.add_argument('--path', help='path to the folder containing the images to fuse')
parser.add_argument('--contrast', default=0.5, help='weight of contrast parameter')
parser.add_argument('--saturation', default=0.5, help='weight of saturation parameter')
parser.add_argument('--well_exposed', default=1, help='weight of well-exposedness parameter')
parser.add_argument('--scale', default=1, help='scale is used to control downsampling (between 0 and 1)')
parser.add_argument('--output', default='output.png', help='name of the output image')
args = parser.parse_args()


I = li.load_images(os.path.join(os.getcwd(), args.path), args.scale)
R = ef.exposure_fusion(I, args.contrast, args.saturation,args.well_exposed)
R = kornia.utils.tensor_to_image(R)
R = np.uint8(R * 255)
cv2.imwrite(os.path.join(os.getcwd(), args.output), R)