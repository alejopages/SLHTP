import click
from os.path import abspath, expandvars, expanduser
import os
import os.path as osp
import logging
import json
from pprint import pprint

from .util import parse_path_str, rc_filepath


log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.DEBUG)


@click.command()
@click.pass_context
def init(ctx):
    ''' initialize the earpheno project '''
    log.info("Please note that icommands must be installed and initialized")

    log.info("Creating ~/.earphenorc")

    '''
    Get the local directory for project data
    EXPLANATION:
    This program requires a lot of paths as arguments, typing in absolute and
    relative paths is repetitive and cumbersome so I included this rc file
    to simplify running commands and managing the project
    '''
    while True:
        local_base = click.prompt('Provide a valid local path for the'\
            + ' data of this project')
        local_base = parse_path_str(local_base)
        log.info('Local path: ' + local_base)
        if click.confirm('Is this the correct path? '):
            break
        else:
            continue

    # SET UP LOCAL DIRECTORY STRUCTURE
    '''
    Creates the following directory structure:
    ================
    local_base
    -- results
    -- -- ear
    -- -- cob
    -- -- kernels
    -- images
    -- -- ear
    -- -- cob
    -- -- kernels
    ================
    '''

    log.info('Setting up local directory structure')
    if not osp.isdir(local_base):
        try:
            os.mkdir(local_base)
        except OSError as ose:
            log.error(ose)
            log.error("You may have set an invalid local path, if so run \'earpheno init\' again")
            quit()
    elif not click.confirm('Local base dir already exists. Use this? '):
        quit()

    for dire in ['results', 'images']:
        if not osp.isdir(osp.join(local_base, dire)):
            print("Making " + osp.join(local_base, dire))
            os.mkdir(osp.join(local_base, dire))
        for type in ['ear', 'kernel', 'cob']:
            if not osp.isdir(osp.join(local_base, dire, type)):
                os.mkdir(osp.join(local_base, dire, type))

    while True:
        iplant_home = click.prompt('Provide the absolute path to your iplant'\
            + ' earpheno project directory on cyverse DE. Example: [/iplant/home/johndoe/earpheno]')
        iplant_home = iplant_home
        if iplant_home[0] != '/':
            log.info('The first character should be \'/\' (root) for absoute path')
            continue
        log.info('iplant home path: ' + iplant_home)
        if click.confirm('Is this the correct path? '):
            break
        else:
            continue

    while True:
        iplant_images = click.prompt('Provide the path to your iplant images'\
            + ' directory RELATIVE to your iplant home directory',
            default='images')
        iplant_images = osp.join(iplant_home, iplant_images)
        log.info('iplant results path: ' + iplant_images)
        if click.confirm('Is this the correct path? '):
            break
        else:
            continue

    while True:
        iplant_analyses = click.prompt('Provide the path to your iplant analyses'\
            + ' directory RELATIVE to your iplant home directory',
            default='analyses')
        iplant_analyses = osp.join(iplant_home, iplant_analyses)
        log.info('iplant analyses path: ' + iplant_analyses)
        if click.confirm('Is this the correct path? '):
            break
        else:
            continue

    '''
    Get iplant results path:
    Explanation: the results path is where the iplant json compilation files would go.
    Anyone may put them in a different place. iplant defaults to 'analyses' directory for all analyses
    '''
    while True:
        iplant_results = click.prompt('Provide the path to your iplant results'\
            + ' directory RELATIVE to your iplant home directory',
            default='results')
        iplant_results = osp.join(iplant_home, iplant_results)
        log.info('iplant results path: ' + iplant_results)
        if click.confirm('Is this the correct path? '):
            break
        else:
            continue


    rc_obj = {
        'local_base':       local_base,
        'iplant_home':      iplant_home,
        'iplant_analyses':  iplant_analyses,
        'iplant_results':   iplant_results,
        'iplant_images':    iplant_images
    }

    with open(rc_filepath, 'w') as rc_file:
        json.dump(rc_obj, rc_file)

    return


@click.command()
@click.pass_context
def show_user(ctx):
    ''' Show user configurations or re-initialize earpheno '''
    log.info("User configurations: ")
    print()
    pprint(ctx.obj)
    print()
