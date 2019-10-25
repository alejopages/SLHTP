import os
import os.path as osp
from os.path import expandvars, abspath, expanduser


rc_filepath = osp.join(os.environ['HOME'], '.earphenorc')


def parse_path_str(path_str, local=True):
    return abspath(expandvars(expanduser(path_str))) if local \
        else expandvars(path_str)


