# Author: Cecil Wang (cecilwang@126.com)

import os


def check_file_extension(filepath, extension):
    assert os.path.splitext(os.path.basename(filepath))[1] == '.'+extension


def ls_sorted_dir(dir):
    filepaths = sorted(os.listdir(dir), key=lambda x:int(x.split('.')[0]))
    filepaths = [os.path.join(dir, x) for x in filepaths]
    return filepaths
