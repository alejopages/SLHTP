'''
DESC: This command crops off the labels that we included at the top
of all the ear images. These labels are important for verifying that
the filename matches the sample id, however it needs to be removed
before entering the pipeline
'''

import cv2
import numpy as np
import os
import os.path as osp
import subprocess as sp
import click
import logging
import re
import tifffile as tiff
from .util import parse_path_str

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.DEBUG)

@click.command()
@click.argument(
	'img_type',
	required=True,
	type=click.Choice(['ear', 'cob', 'kernel'])
)
@click.argument(
	'input',
	required=True,
	nargs=1
)
@click.option(
	'--offset',
	required=False,
	help='Percentage to remove from the top',
	default=0.11,
	nargs=1,
	type=float
)
@click.option(
	'--image_file',
	'-i',
	required=False,
	default=False,
	help='Specify single image file to crop inside input path'
)
@click.pass_context
def crop(ctx, input, offset, img_type, image_file):
	''' Crop header from images for processing in cyverse DE '''

	input = parse_path_str(input)

	output = osp.join(
		ctx.obj['local_base'],
		'images',
		img_type,
		osp.basename(input)
	)

	sp.run(['mkdir','-p', output])

	log.info('Arguments and options:')
	log.info('Input dir:  		' + input)
	log.info('Output dir: 		' + output)
	log.info('Offset Perc: 		' + str(offset))
	log.info('Image: 	   		{}'.format('True' if image_file else 'False'))
	log.info('')

	PATTERN = re.compile(r'.*\.(tif|tiff)')

	if image_file:
		if osp.exists(osp.join(input, image_file)):
			files = [image_file]
		else:
			log.error("Image file does not exist " + image_file)
			return
	else:
		log.info('Getting images from:	' + input)
		files = [fp for fp in os.listdir(input) \
				 if PATTERN.match(fp)]

	log.info('{} images will be cropped'.format(len(files)))
	log.info('\n')
	for fp in files:
		log.info('Cropping ' + fp)
		try:
			img = tiff.imread(osp.join(input, fp))
			top_row = int(img.shape[0] * offset)
			img = img[top_row:, :, :]
			log.info("Writing to " + osp.join(output, fp))
			tiff.imsave(osp.join(output, fp), img)
		except Exception as e:
			log.error('Failed to crop: ' + fp)
			log.error(e)

	return
