import os
import os.path as osp
import pickle as pkl
import subprocess as sp

import click
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import numpy as np

from .preprocess import (
    get_filter_kmeans,
    get_filter_thresh,
    sorted_regions
)
# from .inout import csv_template
from .row_analyses import row_analysis
from .logger import get_logger


log = get_logger()


STARTING_THRESH = 3.5
STARTING_SELEM  = 13


@click.command()
@click.argument('imgdir')
@click.argument('outdir')
@click.option(
    '-m', '--method',
    help='The segmentation method to use',
    nargs=1,
    type=click.Choice(['region', 'kmeans', 'thresh']),
    required=True
)
@click.option(
    '-d', '--no-display',
    help='Don\'t display the figures. Process can be quicker but don\'t use this unless you\'ve been through the analysis already.',
    default=False,
    is_flag=True
)
@click.option(
    '--n_clusters',
    help='Number of clusters to use with KMeans method',
    nargs=1,
    default=4
)
@click.option(
    '-r', '--reset',
    help='Recompute cached image paramters and recompute filters',
    default=False,
    is_flag=True
)
def rows(outdir, imgdir, method, no_display, reset, n_clusters):
    ''' Run row analyses on imgdir, output figures and export to
    outdir '''

    if not osp.isdir(outdir):
        log.debug(f"Creating output directory {outdir}")
        os.mkdir(osp.abspath(outdir))
    if not osp.isdir(osp.join(outdir, 'pickles')):
        os.mkdir(osp.join(outdir, 'pickles'))
    if not osp.isdir(osp.join(outdir, 'temp')):
        os.mkdir(osp.join(outdir, 'temp'))

    columns = [
        'Image#', 
        'Row#',
        'num_objs',
        'area',
        'Shape_PC1',
        'Shape_PC2',
        'mean_R',
        'mean_G',
        'mean_B',
        'mean_H',
        'mean_S',
        'mean_V'
    ]
    data = pd.DataFrame(columns=columns)

    # set defaults
    resp_thresh = STARTING_THRESH
    resp_selem  = STARTING_SELEM
    rep = False
    for img_file in os.listdir(imgdir):
        # img_num = int(img_file.split('.')[0][4:]) # this line applies specifically to the naming convention on Nate's bean project
        img_num = img_file # this line makes it so that the Image# field in the ouput csv is the file name.
        log.info(f"Processing {img_file}")
        cached_analysis_file = osp.join(outdir, 'pickles', '.'.join(img_file.split('.')[0:-1]) + '.pkl')
        if osp.exists(cached_analysis_file) and not reset:
            log.info(f"Loading data from previous analysis on {img_file}")
            with open(cached_analysis_file, 'rb') as analysis_file:
                row_data = pkl.load(analysis_file)
            data = _append_row(data, row_data, img_num)
            continue

        image = imread(osp.join(imgdir, img_file))

        plt.imshow(image)
        plt.show()

        resp_selem = 13 # Default binary closing matrix size

        if method == 'kmeans':
            while True:
                filter_file = osp.join(outdir, 'temp', osp.basename(img_file)[:-3] + 'npy')
                if osp.exists(filter_file) and not (reset or rep):
                    filter = np.load(filter_file)
                    _, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,15))
                    ax1.imshow(filter, cmap='gray')

                else:
                    filter = get_filter_kmeans(image,
                        n_clusters=n_clusters,
                        opening_selem=resp_selem,
                        display=not no_display)
                    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,15))
                    ax1.imshow(filter)

                rows = sorted_regions(filter)

                if no_display:
                    ax2=None
                else:
                    ax2.imshow(image)

                row_data = {}
                for row_num, row in enumerate(rows):
                    row_data[row_num] = row_analysis(image, row, row_num, plt_axis=ax2)

                plt.savefig(osp.join(outdir, osp.basename(img_file)))
                if not no_display:
                    plt.tight_layout()
                    plt.show()

                if _kmeans_reset_prompt():
                    continue
                else:
                    np.save(filter_file, filter)
                    break

        elif method == 'thresh':
            while True:
                filter = get_filter_thresh(image,
                    std_factor=resp_thresh,
                    opening_selem=resp_selem,
                    display=not no_display)
                # only prompt if displaying image

                rep, resp_thresh, resp_selem = _thresh_reset_prompt(resp_thresh, resp_selem)
                if rep: continue

                rows = sorted_regions(filter)

                if no_display:
                    ax=None
                else:
                    fig, ax = plt.subplots()
                    ax.imshow(image)

                row_data = {}

                for row_num, row in enumerate(rows):
                    row_data[row_num+1] = row_analysis(image, row, row_num, plt_axis=ax)

                plt.savefig(osp.join(outdir, osp.basename(img_file)))
                if not no_display:
                    plt.tight_layout()
                    plt.show()

                rep, resp_thresh, resp_selem = _thresh_reset_prompt(resp_thresh, resp_selem)
                if rep:
                    continue
                else:
                    break
        else:
            click.echo(f'Invalid method designation: {method}')
            return False

        data = _append_row(data, row_data, img_num)

        log.info("Caching row analysis")
        with open(cached_analysis_file, 'wb') as analysis_file:
            pkl.dump(row_data, analysis_file)

    log.info("Exporting data to csv")
    data = data.sort_values(['Image#', 'Row#'])
    data[columns].to_csv(osp.join(outdir, "export.csv"))
    

def _append_row(data, row, img_num):
    for row_num, row_data in row.items():
        temp = row_data.copy()
        temp['Row#'] = row_num
        temp['Image#'] = img_num
        data = data.append(pd.DataFrame.from_dict([temp]), sort=True)
    return data


@click.command()
@click.argument('export')
@click.argument('genotypes')
def add_genotypes(export, genotypes):
    ''' Add the genotypes from csv to the bean export generated by rows '''
    export = osp.abspath(osp.expandvars(export))
    genotypes = osp.abspath(osp.expandvars(genotypes))
    if not osp.exists(export):
        log.error(f"Export file does not exist: {export}")
        exit()
    if not osp.exists(genotypes):
        log.error(f"Genotypes file does not exist: {genotypes}")
        exit()

    export_df = pd.read_csv(export)
    genotypes_df = pd.read_csv(genotypes)
    assert 'Genotype' in genotypes_df.columns

    genotypes_df['Row#'] = genotypes_df['Row#'].astype(export_df.dtypes['Row#'])
    genotypes_df['Image#'] = genotypes_df['Image#'].astype(export_df.dtypes['Image#'])

    result_columns = ['Genotype'] + list(export_df.columns)

    result = pd.DataFrame(columns=result_columns)

    for idx, row in export_df.iterrows():
        try:
            image_num = row['Image#']
            row_num = row['Row#'] + 1

            log.info(f"Processing Image# {image_num} and Row# {row_num}")
            genotype = genotypes_df[(genotypes_df['Image#'] == image_num) & (genotypes_df['Row#'] == row_num)]['Genotype']
            log.info(f'Genotype: {genotype}')
            if len(genotype) > 1:
                if genotype.unique().shape[0] > 1:
                    log.warning(f"Found > 1 genotypes for Image# {image_num} and Row# {row_num}")
            elif len(genotype) == 0:
                log.warning(f"Could not find genotype for Image# {image_num} and Row# {row_num}")
                continue
            row['Genotype'] = ",".join(genotype.values)
            result = result.append(row[result.columns])

        except KeyError as ke:
            log.error(f"Could not find a genotype for Image# {image_num} and Row# {row_num}. Skipping")
            continue        

    print(result.head())
    result[result_columns].to_csv('export.csv')
    return


def _kmeans_reset_prompt():
    return click.confirm('Reset params and try again? ')


def _thresh_reset_prompt(resp_thresh, resp_selem):
    if click.confirm('Reset params and try again? '):
        thresh = click.prompt(
            f'Adjust threshold: (REAL)',
            type=float, default=resp_thresh)
        selem  = click.prompt(
            f'Adjust opening selem: (INTEGER)',
            type=int, default=resp_selem)
        return True, thresh, selem
    else:
        return False, resp_thresh, resp_selem
