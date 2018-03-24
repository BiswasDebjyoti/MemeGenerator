import dlib
from PIL import Image
import argparse

from imutils import face_utils
import numpy as numpy

import moviepy.editor as mpy

parser = argparse.ArgumentParser()
parser.add_argument("-image", required=True,help="path to input image")
args=parser.parse_args() 