import click
import logging

from logger import get_logger


log = get_logger(level=logging.INFO)


@click.group()
@click.option(
    '-q', '--quiet',
    default=False,
    is_flag=True
)
@click.pass_context
def Beans(ctx, quiet):
    ''' Bean phenotyping pipeline '''
    if quiet:
        log.setLevel(logging.ERROR)
    pass


from commands import (
    rows,
    add_genotypes
)


Beans.add_command(rows)
Beans.add_command(add_genotypes)


def start():
    Beans(obj={})


if __name__ == '__main__':
    Beans()
