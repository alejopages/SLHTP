from colorsys import rgb_to_hsv
import os.path as osp
import logging
import pickle as pkl
from pprint import pprint

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from skimage.color import rgb2hsv, label2rgb

from .logger import get_logger
log = get_logger()

TEST_IMAGE_PATH = osp.abspath('./data/MADP_SB_8_003.tif')

def row_analysis(image, row, row_num, plt_axis=None):
    ''' Get the average measurements for a row of beans in image.
    args:
        row: a list of regionprops objects
        image: The rgb/grayscale image for which the row belongs
        validation_image: whether or not to generate an image with measurements
            painted on for validation
    returns:
        props: dict
            length: average length along the major axis
            width: average length along the minor axis
            rgb_avg: average rgb value
            hsv_avg: average hsv value
    '''

    length, width = 0, 0
    RGB_avg = 0
    area_avg = 0

    min_r, min_c, max_r, max_c = row[0].bbox

    for region in row:
        # average dims
        length += region.major_axis_length
        width  += region.minor_axis_length

        # average RGB
        r_0, c_0, r_1, c_1 = region.bbox
        RGB_avg += np.mean(image[r_0:r_1, c_0:c_1][region.filled_image], axis=0)

        # average area
        area_avg += region.filled_area # the objects should not have holes

        max_r = max_r if max_r > r_1 else r_1
        min_r = min_r if min_r < r_0 else r_0
        max_c = max_c if max_c > c_1 else c_1
        min_c = min_c if min_c < c_0 else c_0

        if plt_axis:
            area_coords = np.where(region.filled_image)
            mask = list(zip(
                [col + c_0 for col in area_coords[1]],
                [row + r_0 for row in area_coords[0]]
            ))
            area_patch = mpatches.Polygon(mask, edgecolor='b', alpha=0.6)
            plt_axis.add_patch(area_patch)

    row_len = len(row) # TODO: filter small objects
    if plt_axis:
        rect = mpatches.Rectangle((min_c, min_r), max_c - min_c, max_r - min_r,
                              fill=False, edgecolor='red', linewidth=2)
        plt_axis.add_patch(rect)
        plt_axis.text(min_c-20, min_r, row_num, fontsize=25)

    # scale rgb to 1.0, compute hsv
    RGB_avg /= row_len
    RGB_avg /= 255.
    HSV_avg = rgb_to_hsv(*RGB_avg)

    result = {
        'num_objs': row_len,
        'area': area_avg / row_len,
        'Shape_PC1': length / row_len,
        'Shape_PC2': width / row_len,
        'mean_R': RGB_avg[0],
        'mean_G': RGB_avg[1],
        'mean_B': RGB_avg[2],
        'mean_H': HSV_avg[0],
        'mean_S': HSV_avg[1],
        'mean_V': HSV_avg[2]
    }

    log.info(f"Number of objects in row: {row_len}")
    
    return result


# def test_row_analysis():

#     from skimage.io import imread
#     from preprocess import get_filter, sorted_regions

#     image = imread(TEST_IMAGE_PATH)

#     if osp.exists('.test_row_analyses.pkl'):
#         with open('.test_row_analyses.pkl', 'rb') as f:
#             rows = pkl.load(f)
#     else:
#         filter = get_filter(image)
#         rows = sorted_regions(filter)
#         with open('.test_row_analyses.pkl', 'wb') as f:
#             pkl.dump(rows, f)

#     pprint(row_analysis(rows[0], image))


# if __name__ == '__main__':
#     test_row_analysis()
