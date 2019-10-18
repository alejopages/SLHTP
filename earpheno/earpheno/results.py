import subprocess as sp
import logging
import os
import os.path as osp
import click
from glob import glob
import pandas as pd


from .util import parse_path_str


log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.DEBUG)


@click.command()
@click.argument(
    'type',
    type=click.Choice(['ear', 'cob', 'kernel'])
)
@click.argument(
    'iplant_analysis_dir',
    required=True
)
@click.argument(
    'iplant_results_dir',
    required=True
)
@click.pass_context
def get_results(ctx, type, iplant_analysis_dir, iplant_results_dir):
    ''' Get and process results files. Run after running analysis AND json
    compilation '''

    iplant_analysis_dir = parse_path_str(iplant_analysis_dir, local=False)

    local_output = osp.join(
        ctx.obj['local_base'], 'results', type,
        osp.basename(iplant_analysis_dir)
    )
    iplant_analysis = osp.join(
        ctx.obj['iplant_analyses'],
        iplant_analysis_dir
    )
    iplant_output = osp.join(
        ctx.obj['iplant_results'],
        iplant_results_dir
    )
    
    log.info('Arguments and Options:')
    log.info('iplant results Directory:         ' + iplant_output)
    log.info('iplant analysis Directory:        ' + iplant_analysis)
    log.info('local output directory:           ' + local_output)
    log.info('')

    try:
        sp.run(['mkdir', '-p', local_output])
        os.chdir(local_output)
        sp.run([osp.join(osp.dirname(__file__), 'shell', 'get_tifs.sh'),
            iplant_analysis, local_output])

        # get results from cyverse DE
        sp.run(['iget', '-v', '-r', '-f', 
            osp.join(iplant_output, 'output')])

    except sp.CalledProcessError as e:
        log.error(e)
        exit()
    except IOError as ioerr:
        log.error(ioerr)
        exit()

    # TODO: verify that the number of returned tifs match the number of uploaded images


@click.command()
@click.argument(
    'type',
    type=click.Choice(['ear', 'cob', 'kernel'])
)
@click.argument(
    "iplant_analysis"
)
@click.pass_context
def get_tifs(ctx, iplant_analysis, type):
    """ Search for tifs in iplant dir and download to local directory """
    
    local_output = osp.join(
        ctx.obj['local_base'],
        type,
        iplant_analysis
    )
    iplant_path = osp.join(
        ctx.obj['iplant_analyses'],
        iplant_analysis
    )

    # Argument and Option summary:
    log.info("Arguments and Options:")
    log.info("iplant analysis directory:         " + iplant_path)
    log.info("local output directory:            " + local_output)
    log.info("")

    try:
        # make local dir
        sp.run(['mkdir', '-p', local_output])

        # call local script to get the goods
        sp.run([osp.join(osp.dirname(__file__), 'shell', 'get_tifs.sh'),
            iplant_path, local_output])

    except sp.CalledProcessError as e:
        log.error(e)
        exit()
    except IOError as e:
        log.error(e)
        exit()
