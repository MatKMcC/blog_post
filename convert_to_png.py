"""
This script converts all files in the tex directory (svg files) to
png files
"""

import cairo
import rsvg
import os

# get list of file names


def convert_image(svg_file):

    img = cairo.ImageSurface(cairo.FORMAT_ARGB32, 640, 480)

