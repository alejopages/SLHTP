import os
import os.path as osp
import logging
import click

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from sklearn.cluster import KMeans
from skimage.io import imread
from skimage.morphology import binary_closing, binary_opening, square
from skimage.measure import label, regionprops
from skimage.filters import sobel

from .logger import get_logger
log = get_logger()

TEST_IMAGE_PATH = osp.abspath('./data/MADP_SB_8_003.tif')

# USER DEFINED VARIABLES
# BG_PXL_MEAN = np.array([55.70774974, 95.57235801, 52.66334754])
# BG_PXL_STD  = np.array([5.79046648, 7.40193659, 5.89953565])
BG_PXL_MEAN = np.array([49.54712426, 77.51730553, 49.30873155])
BG_PXL_STD  = np.array([ 7.7995182,  16.72836942,  7.24843234])

INITIAL_CLUSTERS = {
    4: np.array([[ 62.24681959, 108.64087395,  58.5843175 ],
                [196.05566408, 190.00799357, 171.15221035],
                [ 82.88725581,  47.97278208,  45.67010508],
                [145.66400941, 116.08091739,  91.09988348]])
}

CMAP = 'jet'

# Testing purposes
def show_image(image, cmap=None):
    fig = plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.imshow(image, aspect='auto', cmap=cmap)
    plt.show()


def region_valid(region):
    ''' Find and remove non-bean objects based on region properties '''
    # TODO: add properties checks such as size and convex hull to area ratio 
    return True


def sorted_regions(image):
    '''
    Desc: Sort image top to bottom, left to right
    args:
        image: black and white image
    returns:
        rows: list of lists of regionprops objects
    '''
    assert image.ndim == 2
    assert np.unique(image).shape[0] == 2

    # get region properties
    regions = regionprops(label(image))
    # affirm that regions are sorted top to bottom
    regions = sorted(regions, key=lambda x: x.bbox[0])

    # return variable where rows are stored
    rows = []
    # the inner interval bounds for the bounding box range
    upper, lower = regions[0].bbox[0], regions[0].bbox[2]
    # temp var used to store the row as it is built
    active = []

    for region in regions:
        if not region_valid(region):
            continue

        min_row, _, max_row, _ = region.bbox
        if min_row > lower:
            # this happens when a new row has been reached
            rows.append(sorted(active, key=lambda x: x.bbox[1]))
            active = []
            upper = min_row
            lower = max_row
        else:
            # reset inner interval bounds
            if min_row > upper:
                upper = min_row
            if max_row < lower:
                lower = max_row

        active.append(region)

    rows.append(sorted(active, key=lambda x: x.bbox[1]))
    if [] in rows: rows.remove([])

    return rows


def get_filter_thresh(image, opening_selem=10, closing_selem=20, std_factor=1.0,
    auto=False):
    '''
    Desc: Get image filter using thresh hold difference method.
    args:
        image: rgb image
        opening_selem: the square matrix size for removing white speckles
        closing_selem: the square matrix size for removing black speckles
        auto: whether to display the image at the end of the function and prompt user for inputs
    returns:
        filter: black and white image with background pixels as black
    '''
    log.info('Getting background filter')
    log.debug(f'Pixel Mean: {BG_PXL_MEAN}')
    log.debug(f'Pixel STD: {BG_PXL_STD}')

    # find where difference of background exceeds stddev
    bg_px_map = abs(np.subtract(image, BG_PXL_MEAN)) \
              <= std_factor * BG_PXL_STD
    bg_px_map = np.all(bg_px_map, axis=2)

    # compute filter image
    filter = np.zeros(image.shape[:2])
    filter[~bg_px_map] = 1

    clean = binary_opening(filter, selem=square(opening_selem))
    clean = binary_closing(clean, selem=square(closing_selem))

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, figsize = (15,10))
    ax1.imshow(image)
    ax1.set_title('Original')
    ax2.imshow(filter, cmap='gray')
    ax2.set_title('Threshold applied')
    ax3.imshow(clean, cmap='gray')
    ax3.set_title('Noise removed')
    plt.axis('off')
    fig.suptitle("Adjust threshold factor in command line to improve filter")
    
    if not auto:
        plt.show()

    return clean


def get_filter_kmeans(image, opening_selem=10, closing_selem=20, n_clusters=4,
                      auto=False):
    '''
    Desc: Get binary image background filter
    args:
        image: rgb image
        n_clusters: number of clusters to use in KMeans clustering
        opening_selem: the square matrix size for removing white speckles
        closing_selem: the square matrix size for removing black speckles
        auto: whether to display the image at the end of the function and prompt user for inputs
    returns:
        filter: binary background filter
    '''

    break_flag = False
    retrain_flag = False

    while True:
        if auto or click.confirm(f'Train model using K={n_clusters}?') or retrain_flag:
            retrain_flag = False
            if n_clusters in INITIAL_CLUSTERS:
                model = KMeans( n_clusters=n_clusters,
                                init=INITIAL_CLUSTERS[n_clusters],
                                n_jobs=-1)
            else:
                model = KMeans( n_clusters=n_clusters,
                                n_jobs=-1) 

            log.info('Training K Means model')
            labels = model.fit_predict(
                image.reshape(image.shape[0] * image.shape[1], image.shape[2])
            ).reshape(image.shape[:2])

            log.info("Storing model cluster centers for faster compute time later")
            INITIAL_CLUSTERS[n_clusters] = model.cluster_centers_

            # TESTING CODE:
            # model.cluster_centers_ = np.array([[49.65536133, 83.51262446, 46.98558262],
            #     [71.55258734, 77.93486518, 71.2245406 ],
            #     [30.19214964, 34.89643069, 32.21520259],
            #     [49.13650677, 53.383587,   50.99198278]])
            # labels = model.predict(
            #     image.reshape(image.shape[0] * image.shape[1], image.shape[2])
            # ).reshape(image.shape[:2])

            colormap = plt.get_cmap(name=CMAP, lut=n_clusters)
            color_map = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0., vmax=n_clusters),
                    cmap=CMAP)

            # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,15))
            # ax1.imshow(image)
            # ax2.imshow(_generate_mapped_image(image, labels, colormap))
            # plt.tight_layout()
            # plt.show()
        else:
            n_clusters = click.prompt('How many clusters? ', type=int)
            retrain_flag = True
            continue

        seg_map = _init_seg_map(model.cluster_centers_)

        while True:
            
            # initilize validation plot
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize = (15,10), sharey=True)
            fig.suptitle("Select label to color mappings")
            
            # formatter = plt.FuncFormatter(lambda val, loc: )

            ax1.set_title('Labels Mapped')
            ax2.set_title('Background [blk], Foreground [wht]')
            ax3.set_title('Noise removed')

            log.info("Generating filter")
            filter = _convert_to_binary(labels, seg_map)
            # Get user input on segmentation mapping

            log.info(f"Cleaning noise from image with selem: {opening_selem}")
            # clear speckled white pixels in the image
            clean = binary_opening(filter, selem=square(opening_selem))
            # clear speckled black pixels in the image
            clean = binary_closing(clean, selem=square(closing_selem))
            
            # generate colored label map
            log.info("Generating label map")
            mapped_image = _generate_mapped_image(image, labels, colormap)
            
            # Make figure
            ax1.imshow(mapped_image)
            ax2.imshow(filter, cmap='gray')
            
            # make colorbar
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', size='5%', pad=0.1)
            bounds = list(range(n_clusters+1))
            # norm = mpl.colors.BoundaryNorm(bounds, color_map)
            norm = mpl.colors.BoundaryNorm(bounds, n_clusters)

            fig.colorbar(color_map, cax=cax, norm=norm, boundaries=bounds, ticks=list(range(n_clusters)), cmap=CMAP)
            
            ax3.imshow(clean, cmap='gray')
            plt.draw()

            if not auto:
                plt.show()
                plt.close()
                rep, seg_map, opening_selem = \
                    _kmeans_label_prompt(labels, seg_map, opening_selem)
            
            if auto or not rep: 
                break_flag = True
                break
        if break_flag:
            break

    return clean


def _init_seg_map(cluster_centers):
    # initialize the background cluster map
    segmentation_map = np.ones((cluster_centers.shape[0],))
    init_bg_cluster = np.argmin(
        np.sum(abs(cluster_centers - BG_PXL_MEAN), axis=1)
    )
    segmentation_map[init_bg_cluster] = 0
    return segmentation_map


def _generate_mapped_image(image, labels, colormap):
    ''' Generate the colored label map '''
    label_map = image.copy()
    for label in np.unique(labels):
        label_map[labels == label] = list(map(lambda x: x * 255, colormap(label)[:3]))
    return label_map


def _kmeans_label_prompt(labels, seg_map, opening_selem):
    ''' A prompt for label to (back/fore)ground pixel map '''
    if click.confirm('Reset label/speckle params and try again? (\'N\' if you want to change number of clusters)'):
        for label in range(len(seg_map)):
            seg_map[label] = click.prompt(
                f'  Set label [{label}]: ',
                type=int, default=int(seg_map[label]))
        selem  = click.prompt(
            'Adjust size of white space to remove: ',
            type=int, default=opening_selem)
        return True, seg_map, selem
    else:
        return False, seg_map, opening_selem


def _convert_to_binary(labels, binary_map):
    '''
    Desc: Generate the binary background filter
    args:
        labels: ndarray with pixel labels
        binary_map: binary ndarray length of number of unique labels
    returns:
        filter: binary background filter
    '''
    filter = labels.copy()
    for label in np.unique(labels):
        filter[labels == label] = binary_map[label]
    return filter


### TEST FUNCTIONS
def test_get_filter_kmean():
    image = imread(TEST_IMAGE_PATH)
    filter = get_filter_kmeans(image)
    np.save('.test_get_filter_kmeans.npy', filter)
    plt.imshow(filter, cmap='gray')
    plt.show()


def test_sorted_regions():
    image = imread(TEST_IMAGE_PATH)
    if osp.exists('test_get_filter_kmeans.npy'):
        filter = np.load('test_get_filter_kmeans.npy')
    else:
        filter = get_filter(image)
    regions = sorted_regions(filter)

    fig, ax = plt.subplots(figsize=(8,8))
    image[~filter] = [0,0,0]
    ax.imshow(image)
    counter = 0

    for row_num, regions in regions.items():
        ax.text(0, regions[0].bbox[0] + 250, str(row_num), color='y')
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(minc - 10, minr-5, str(counter), color='r')
            counter += 1

    ax.set_axis_off()
    plt.show()


if __name__ == '__main__':
    # test_get_filter()
    test_sorted_regions()
