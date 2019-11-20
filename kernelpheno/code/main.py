from .logger import get_logger

import click
import logging

log = get_logger(level=logging.DEBUG)

@click.group()
def KernelPheno():
    ''' Kernel Vitreousness project management and phenotyping tools '''
    pass


''' COMMAND IMPORTS '''
from .zooexp import *
from .preprocess import (
    generate_dataset,
    split_validation,
    convert,
    segment,
    normalize,
    plot_bbox
)

if __name__ == '__main__':
    KernelPheno()
