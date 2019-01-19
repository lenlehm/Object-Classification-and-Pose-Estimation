
"""
Utilities
[1] readtxt(filePath, delimiter)
    return: array of int
[2] getposes(filePath)
    return: quartenions in file, e.g. [-0.28184579021235323, -0.6032481990846498, 0.6534595646367771, -0.3600627142949052]
"""

import os
import errno
import logging
import string
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def readtxt(txtfile, delim):
    with open(txtfile) as f:
        data_list = f.readlines()
        tmpstr = ''.join(x.strip() for x in data_list)
        data_list = tmpstr.split(delim)
        data_list = [int(x.strip()) for x in data_list]
        return data_list

def getposes(pose_txt):
    delim = '#'
    with open(pose_txt) as f:
        data_list = f.readlines()
        tmp_list = [x.strip(delim).strip() for x in data_list]

    pose_quart = []
    # print(len(tmp_list))
    # print(tmp_list[-1])
    # added condition because some have an empty split at the end
    if len(tmp_list)%2 != 0:
        MAX_NUM = len(tmp_list)-1
    else:
        MAX_NUM = len(tmp_list)
    for i in range(0, MAX_NUM, 2):
        per_line = tmp_list[i+1].split()
        per_line = [float(x) for x in per_line]
        pose_quart.append(per_line)
    return pose_quart
