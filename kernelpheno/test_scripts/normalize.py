from skimage.segmentation import slic
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu, gaussian
from skimage.color import label2rgb
from skimage.morphology import closing, square
from skimage.util import invert
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import os.path as osp


img_files = ["/home/apages/pysrc/KernelPheno/data/DSC05377.jpeg",
             "/home/apages/pysrc/KernelPheno/data/DSC05389.jpeg",
             "/home/apages/pysrc/KernelPheno/data/DSC05384.jpeg"]

br_thresh_sum = 0
num_images = 0

for path in img_files:
    col = imread(path)
    gray = rgb2gray(col)
    thresh = threshold_otsu(gray)

    num_images += 1
    br_thresh_sum += thresh

    print("Threshold: " + str(thresh))
    bw = closing(gray > thresh, square(3))

    masked = col.copy()
    masked[np.where(bw)] = [0,0,0]

    # plt.imshow(masked)
    # plt.show()

avg = br_thresh_sum / num_images
print("Average: " + str(avg))

for path in img_files:
    col = imread(path)
    gray = rgb2gray(col)
    thresh = threshold_otsu(gray)
    bw = closing(gray > thresh, square(3))

    diff = avg - thresh
    print("Image diff to avg: " + str(diff))

    normed = gray + diff

    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(6,6))
    ax[0].imshow(normed, cmap='gray')
    ax[1].imshow(gray, cmap='gray')
    plt.show()
