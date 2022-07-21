import colorsys
import itertools
import os
import random
import sys

import IPython.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines, patches
from matplotlib.patches import Polygon
from skimage.measure import find_contours


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    print(colors)
    return colors

if __name__ == '__main__':
    random_colors(10)
