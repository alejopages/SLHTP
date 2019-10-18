import subprocess as sp
import logging
import click
import os
import os.path as osp

from .util import parse_path_str


log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.DEBUG)


@click.command()
@click.argument(
    'analysis_type',
    type=click.Choice(['ear', 'cob', 'kernel'])
)
@click.argument(
    'upload_dir'
)
@click.pass_context
def upload_images(ctx, analysis_type, upload_dir):
    """ Upload images to cyverse DE. Upload will go to iplant images dir """
    upload_path = osp.join(
        ctx.obj.get('local_base'),
        'images',
        analysis_type,
        upload_dir
    )
    iplant_image_dir = osp.join(
        ctx.obj['iplant_images'],
        analysis_type
    )

    log.info("Arguments and Options:")
    log.info("Path to upload directory:      " + upload_path)

    num_files = len(os.listdir(upload_path))
    log.info("{} files will be uploaded".format(num_files))

    try:
        # Create the iplant directory
        # -p prevents the dir from being overwritten
        sp.run(['imkdir', '-p', iplant_image_dir])
        # Change working directoy to iplant directory
        sp.run(['icd', iplant_image_dir])
        # upload the images
        sp.run(['iput', '-r', '-v', upload_path])

    except sp.CalledProcessError as e:
        log.error(e)
        exit()
    except IOError as ioe:
        log.error(ioe)
        exit()
    return
