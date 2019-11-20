from skimage.segmentation import clear_border
from skimage.io import imread, imsave
from skimage.color import rgb2gray, gray2rgb
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.util import invert
from skimage import img_as_float, img_as_ubyte

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import os
import os.path as osp
import logging
import click
import subprocess as sp
import traceback as tb

from logger import get_logger
from utils import (
    get_image_regex_pattern,
    show_image,
    is_gray,
    create_name_from_path
)
from segmentation import segment_image, get_filter

log = get_logger(level=logging.DEBUG)

''' GENERATE DATASET '''
@click.command()
@click.argument('indir')
@click.argument('outdir')
@click.argument('anno_file')
@click.option(
    '-v',
    '--validation_split',
    nargs=1,
    default=0.2,
    type=float
)
@click.option(
    '-s',
    '--segment',
    is_flag=True,
    default=False,
    help="Whether or not to apply the segmentation filter to the image"
)

def dataset(indir, outdir, anno_file, validation_split, segment):
    '''
    Generate a dataset for CNN training

    params
    * indir: directory of preprocessed images
    * outdir: directory to output the images to
    * anno_file: the path to the file of annotations
    '''

    sp.run(['mkdir', '-p', outdir])

    fh = logging.FileHandler(osp.join(outdir, 'log'))
    fh.setLevel(logging.WARNING)
    log.addHandler(fh)

    data = osp.join(outdir, 'data')

    sp.run(['mkdir', '-p', data])
    sp.run(['mkdir', '-p', osp.join(data, 'train')])
    # sp.run(['mkdir', '-p', osp.join(data, 'valid')]) # handled by split_val

    for i in range(1, 6):
        sp.run(['mkdir', '-p', osp.join(data, 'train', str(i))])

    bbox_dir = osp.join(outdir, 'bboxes')
    bbox_err_dir = osp.join(outdir, 'err_bboxes')
    sp.run(['mkdir', '-p', bbox_dir])
    sp.run(['mkdir', '-p', bbox_err_dir])

    sp.run(['echo', 'filename,rating\n', '>',
            osp.join(outdir, 'annotations.csv')])
    bbox_err_count = 0

    try:
        annotations = pd.read_csv(anno_file)
        # convert the ratings string to list
        annotations['ratings'] = annotations['ratings'].apply(eval)

    except FileNotFoundError as fnfe:
        log.error(fnfe)
        exit()

    annotation_file = open(osp.join(outdir, 'annotations.csv'), 'a')
    annotation_file.write("filename,rating\n")
    annotations_summary = {
        '1': 0,
        '2': 0,
        '3': 0,
        '4': 0,
        '5': 0
    }

    for i, row in annotations.iterrows():
        log.info("Processing " + row['filename'])
        image_path = osp.join(indir, row['filename'])
        if not osp.isfile(image_path):
            log.error("Could not locate " + str(image_path))
            continue
        try:
            image = imread(image_path)
            if is_gray(image):
                image = gray2rgb(image)
        except Exception as e:
            log.error("Failed to load " + image_path)
            log.error(e)
            tb.print_exc()
            continue

        bboxes = get_sorted_bboxes(image)
        plot_bbx(image, bboxes)

        if len(bboxes) != row['num_objects']:
            log.error("Count of objects in image did not match: " + row['filename'])
            out_fname = osp.join(bbox_err_dir, row['filename'])
            bbox_err_count += 1
            plt.savefig(out_fname)
            plt.close('all')
            continue
        else:
            out_fname = osp.join(bbox_dir, row['filename'])
            plt.savefig(out_fname)
            plt.close('all')

        # squash 2d list to 1d
        ratings = [entry for line in row['ratings'] for entry in line]
        if segment:
            image = segment_image(image) 

        log.info("Getting thumbnails")
        for j, bbox in enumerate(bboxes):
            anno = ratings[j]
            minr, minc, maxr, maxc = bbox
            thumbnail = image[minr:maxr, minc:maxc]

            out_fname = osp.join(
                outdir, 'data', 'train',
                str(anno),
                str(j) + "_" + row['filename']
            )
            imsave(out_fname, thumbnail)
            annotation_file.write("{},{}\n".format(
                str(j) + "_" + row['filename'],
                anno)
            )
            annotations_summary[anno] += 1

    annotation_file.close()
    log.info("Generating validation dataset")
    split_val(data, validation_split)

    log.info("SUMMARY:")
    log.info("Number of bbox errors: " + str(bbox_err_count))
    for i in range(1,6):
        log.info("Number of annotations with rating {}: {}".format(
            i, annotations_summary[str(i)]))

    num_samples = sum(list(annotations_summary.values()))
    log.info("Total number of images: {}".format(num_samples))

    return


def split_val(datadir, split):
    ''' Generate the validation dataset '''

    assert (0 < split < 1.0)
    assert osp.isdir(datadir)

    sp.run(['mkdir', '-p', osp.join(datadir, 'valid')])

    for i in range(1, 6):
        sp.run(['mkdir', '-p', osp.join(datadir, 'valid', str(i))])

    PATTERN = get_image_regex_pattern()

    datadir = osp.abspath(datadir)
    train_dir = osp.join(datadir, 'train')
    validation_files = {}

    for dirpath, dirnames, filenames in os.walk(train_dir):
        image_files = [file for file in filenames if PATTERN.match(file)]
        num_examples = round(len(image_files) * split)

        if num_examples < 1: continue

        validation_files[osp.basename(dirpath)] = \
                np.random.choice(
                    filenames,
                    size = num_examples
                )

    for rating, filenames in validation_files.items():
        for filename in filenames:
            log.info("Moving file to validation set " + osp.join(rating, filename))
            file = osp.join(train_dir, rating, filename)
            try:
                sp.run(['mv', file,
                    osp.join(
                        datadir,
                        'valid',
                        osp.basename(osp.dirname(file)),
                        osp.basename(file)
                    )]
                )
            except sp.CalledProcessError as cpe:
                log.error(cpe)


''' CONVERSION FROM MISC IMAGE FORMATS TO JPG OR OTHERWISE SPECIFIED FORMAT '''

@click.command()
@click.argument(
    'dir'
)
@click.option(
    '-f',
    'format',
    help='Format to convert the images to',
    default='jpg',
    show_default=True
)
@click.option(
    '--copyto',
    help='Copy images to this directory before converting',
    default=False
)
def convert(dir, format, copyto):
    ''' Convert images to specified format '''

    if not os.path.isdir(dir):
        log.error('The input dir does not exist')
        return

    if copyto:
        sp.run(['mkdir', '-p', copyto])
        sp.run(['cp', '-r', dir, copyto])

    sp.run(['mogrify', '-format', format, osp.join(dir, '*')])

    return


''' SEGMENTATION FUNCTION '''

@click.command()
@click.argument(
    'indir'
)
@click.argument(
    'outdir'
)
@click.option(
    '-e',
    '--extension',
    help='The image extension',
    type=click.Choice(['jpg', 'jpeg', 'png', 'tif', 'tiff']),
    multiple=True,
    default=None
)
def segment(indir, outdir, extension):
    '''
    Segment the kernels in the image or images

    params
    * indir:    directory with images to segment
    * outdir:   directory to output images (will create if doesn't exist)
    * type:     gray or rgb
    '''
    sp.run(['mkdir', '-p', outdir])
    PATTERN = get_image_regex_pattern()
    for image_path in os.listdir(indir):
        if not PATTERN.match(image_path): continue
        try:
            image = imread(osp.join(indir, image_path))
            seg_image = segment_image(image)
            out_fname = create_name_from_path(image_path, out_dir=outdir)
            imsave(out_fname, seg_image)
        except Exception as e:
            log.error('Failed to process ' + osp.basename(image_path))
            log.error(e)
            tb.print_exc()
            continue


''' NORMALIZE IMAGES '''
@click.command()
@click.argument(
    'indir'
)
@click.argument(
    'outdir'
)
@click.argument(
    'type',
    type=click.Choice(['rgb', 'gray'])
)
@click.option(
    '--plot',
    help='Plot the comparison between normed and real'
)
def normalize(indir, outdir, type, plot):
    '''
    Perform mean scaling normalization method

    params
    * indir:    directory with images to normalize
    * outdir:   directory to output images (will create if doesn't exist)
    * type:     image type for normalization
    '''

    ###########################################################################
    # TO BE UPDATED WHEN RGB NORMALIZATION IMPLEMENTED
    if type == 'rgb':
        log.info('Normalization for color images has not yet been implemented')
        return
    ###########################################################################

    sp.run(['mkdir', '-p', outdir])

    PATTERN = get_image_regex_pattern()
    bg_avg = get_bg_avg(indir, PATTERN, type)

    for image_path in os.listdir(indir):
        if not PATTERN.match(image_path): continue
        log.info('Processing ' + image_path)

        try:
            log.info('Loading image')
            if type == 'gray':
                image = imread(osp.join(indir, image_path), as_gray=True)
            else:
                image = imread(osp.join(indir, image_path))
                if len(image.shape) == 2:
                    image = gray2rgb(image)

            normed = norm(image, bg_avg)

            if plot:
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6,6))
                fig.set_title('Normalized Image Comparison')
                if type == 'gray':
                    ax[0].imshow(image, cmap='gray')
                    ax[1].imshow(normed, cmap='gray')
                else:
                    ax[0].imshow(image)
                    ax[1].imshow(normed)
                figname = create_name_from_path(image_path, ('fig'), out_dir=outdir)
                plt.savefig(figname)

            out_fname = create_name_from_path(image_path, out_dir=outdir)
            log.info('Saving file: ' + out_fname)
            imsave(out_fname, normed)

        except Exception as e:
            log.error('Failed to process ' + osp.basename(image_path))
            log.error(e)
            tb.print_exc()
            continue

    return


''' PLOT BOUNDING BOXES '''
@click.command()
@click.argument(
    'indir'
)
@click.argument(
    'outdir'
)
def plot_bbox(indir, outdir):
    '''
    Plot the bounding boxes for each image given in indir

    params
    * indir:    directory with images to normalize
    * outdir:   directory to output images (will create if doesn't exist)
    '''
    sp.run(['mkdir', '-p', outdir])

    PATTERN = get_image_regex_pattern()
    for image_name in os.listdir(indir):
        if not PATTERN.match(image_name): continue
        log.info('Processing ' + osp.basename(image_name))

        try:
            image = imread(osp.join(indir, image_name))

            bboxes = get_sorted_bboxes(image)
            plot_bbx(image, bboxes)
            out_fname = osp.join(outdir, image_name)
            plt.savefig(out_fname)
        except Exception as e:
            log.error('Failed to process ' + osp.basename(image_path))
            log.error(e)
            tb.print_exc()
            continue


''' FUNCTIONS '''

def norm(image, bg_avg):
    ''' Normalize each by subtracting the background pixel mean differenc '''
    log.info('Normalizing image')

    gray = True if is_gray(image) else False
    image = img_as_float(image)
    filter = get_filter(image)
    masked = image.copy()

    if gray:
        masked[~filter] = 0
    else:
        masked[~filter] = [0,0,0]

    diff = bg_avg - np.mean(masked, axis=(0,1))

    log.info('Background diff: ' + str(diff))

    normed = image + diff

    if gray:
        normed[normed > 1.0] = 1.0
        normed[normed < -1.0] = -1.0
    else:
        # TODO: find out how to scale the color normalized to [0,255]
        pass

    return normed


def test_norm():
    image = imread('/home/apages/pysrc/KernelPheno/data/sample_images/DSC05389.jpeg', as_gray=True)
    PATTERN = get_image_regex_pattern()
    bg_avg = get_bg_avg('/home/apages/pysrc/KernelPheno/data/sample_images', PATTERN, type='gray')
    normed = normalize(image, bg_avg)
    plt.imshow(normed, cmap='gray')
    plt.show()
    return


def get_bg_avg(indir, PATTERN, type):
    ''' Get the background mean pixel values '''

    log.info('Gettind background pixel average')

    if type == 'gray':
        sum = 0
    else:
        sum = np.array([0,0,0], dtype=float)

    img_count = 0
    for image_path in os.listdir(indir):
        log.info('Processing ' + image_path)
        if not PATTERN.match(image_path): continue
        try:
            if type == 'gray':
                image = imread(osp.join(indir, image_path), as_gray=True)
            else:
                image = imread(osp.join(indir, image_path))
        except Exception as e:
            log.error('Failed to process ' + image_path)
            log.error(e)
            continue

        image = img_as_float(image)

        filter = get_filter(image)
        masked = image.copy()

        if type == 'gray':
            masked[~filter] = 0
        else:
            masked[~filter] = [0,0,0]

        mean = np.mean(masked, axis=(0,1))
        sum += mean
        img_count += 1
    try:
        mean = sum / float(img_count)
        log.info('All image background average: ' + str(mean))
    except ZeroDivisionError as zde:
        log.error('Zero division error, must not have had any images in indir')
        raise zde

    return mean


def plot_bbx(image, bboxes):
    ''' Plot red bounding boxes and return plot object '''

    fig, ax = plt.subplots(figsize=(6,6))

    if is_gray(image):
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)

    for i, (minr, minc, maxr, maxc) in enumerate(bboxes):
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(minc - 20, minr - 20, str(i))

    return plt


def get_sorted_bboxes(image):
    ''' Generate the sorted bounding boxes '''
    log.info('Getting sorted bounding boxes')
    filter = get_filter(image)
    cleared = clear_border(invert(filter))
    label_image = label(cleared)
    coords = []
    for region in regionprops(label_image, coordinates='rc'):
        if (region.area < 1000) \
            or (region.area > 100000) \
            or ((region.major_axis_length / region.minor_axis_length) < 0.2) \
            or ((region.minor_axis_length / region.major_axis_length) < 0.2):
            continue

        coords.append(region.bbox) # minr, minc, maxr, maxc

    sorted_bbxs = _sort_bbxs(coords, image.shape[0])

    return sorted_bbxs


def _sort_bbxs(regions, num_rows):
    ''' Sort bboxes left to right, top to bottom '''

    def overlap(el1, el2):
        ''' determine if bounding boxes overlap along a row '''
        upper_max = max(el1[0], el2[0])
        lower_min = min(el1[2], el2[2])
        return (upper_max < lower_min)

    rows = []

    while(len(regions)):
        sorted_by_y = sorted(regions, key=lambda x: x[0])
        first_el = sorted_by_y[0]
        rows.append([first_el])
        regions.remove(first_el)
        sorted_by_y.pop(0)
        for el in sorted_by_y:
            if overlap(el, first_el) or overlap(el, rows[-1][-1]):
                rows[-1].append(el)
                regions.remove(el)

    sorted_bbxs = []
    for row in rows:
        sorted_bbxs += sorted(row, key=lambda x: x[1])
    return sorted_bbxs


# def _get_background_filter(image):
#     ''' Get's the binary filter of the segmented image '''
#     log.info('Getting image background filter')
#     if not is_gray(image):
#         image = rgb2gray(image)
#     thresh = threshold_otsu(image)
#     bw = closing(image > thresh, square(3))
#     return invert(bw)


if __name__ == '__main__':
    split_val("/home/apages/pysrc/KernelPheno/data/datasets/unprocessed/data", 0.2)
    # generate_dataset(r'/home/apages/SCHNABLELAB/pysrc/KernelPheno/data/images',
    #                  r'/home/apages/SCHNABLELAB/pysrc/KernelPheno/data/dataset',
    #                  r'/home/apages/SCHNABLELAB/pysrc/KernelPheno/report.clean.csv')
