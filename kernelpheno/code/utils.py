import re
import os.path as osp
import matplotlib.pyplot as plt
import logging

from logger import get_logger

log = get_logger(level=logging.DEBUG)

def get_image_regex_pattern(extension=()):
    '''
    Returns a regex object useful for grabbing files with image filename extensions
    '''
    if extension == ():
        return re.compile(r".*\.(tif|tiff|jpg|jpeg|png)")

    patter_str = r".*\.(" + extension[0]
    for ext in extension[1:]:
        pattern_str += "|" + ext
    patter_str += ")"

    return re.compile(patter_str)


def create_name_from_path(file_path, pre_ext=[], out_dir=False):
    '''
    Inserts custom tags seperated by dots between filename and extension
    '''
    if type(pre_ext) != list:
        pre_ext = [pre_ext]

    extensions = osp.basename(file_path).split(".")
    for i, ext in enumerate(pre_ext):
        if ext in extensions:
            log.debug(
                "File: " + osp.basename(file_path)\
                + " already has extension: " \
                + ext
            )
            log.debug("Ommitting from resulting filename")
            pre_ext.pop(i)

    if pre_ext != []:
        exts = "." + ".".join(pre_ext) + "."
    else:
        exts = "."
    out_path    = ".".join(file_path.split(".")[:-1]) \
                +  exts \
                + file_path.split(".")[-1]
    if out_dir:
        out_path = osp.join(
            out_dir,
            osp.basename(out_path)
        )

    return out_path


def is_gray(image):
    return True if image.ndim == 2 else False


def show_image(image):
    '''
    Simply plots the image
    '''
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()
