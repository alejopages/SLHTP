import click
import os
import os.path as osp

import logging
import json

from .util import rc_filepath

from .initialize        import (init, show_user)
from .preprocess        import crop
from .upload            import upload_images
from .results           import (get_results, get_tifs)
from .reports           import report


log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.DEBUG)


@click.group()
@click.pass_context
def earpheno(ctx):
    """ Tools for managing the ear phenotyping project """
    if osp.exists(rc_filepath):
        with open(rc_filepath, 'r') as rc_file:
            try:
                ctx.obj = json.load(rc_file)
            except json.decoder.JSONDecodeError as je:
                log.error("Could not load rc file [~/.earphenorc].")
                quit()
    else:
        click.echo("Run `earpheno init` first to initialize earpheno")
    pass


earpheno.add_command(show_user)
earpheno.add_command(init)
earpheno.add_command(crop)
earpheno.add_command(upload_images)
earpheno.add_command(get_results)
earpheno.add_command(report)
earpheno.add_command(get_tifs)


def start():
    earpheno(obj={})


if __name__ == '__main__':
    start()
