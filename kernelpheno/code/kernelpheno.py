import click

@click.group()
def KernelPheno():
    ''' Kernel Vitreousness project management and phenotyping tools '''
    pass


''' COMMAND IMPORTS '''
from zooexp import zooexp
from preprocess import (
    dataset,
    convert,
    segment,
    normalize,
    plot_bbox
)
from models.alexnet import train_alex


KernelPheno.add_command(zooexp)
KernelPheno.add_command(dataset)
KernelPheno.add_command(convert)
KernelPheno.add_command(segment)
KernelPheno.add_command(normalize)
KernelPheno.add_command(plot_bbox)
KernelPheno.add_command(train_alex)


if __name__ == '__main__':
    KernelPheno()
