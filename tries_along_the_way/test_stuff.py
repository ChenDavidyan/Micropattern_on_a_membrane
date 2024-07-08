import cv2
import nd2
from matplotlib import pyplot as plt
import numpy as np
from skimage import io, restoration, exposure, img_as_ubyte
import seaborn as sns
import pandas as pd
from nd2reader import ND2Reader
from pims import Frame



nd2_path = r"C:\Users\חן דודיאן\OneDrive\מסמכים\biology\MSc\Rotations\Eyal Karzbrun\membrane\Ex12\Exp12\Membrane_on_glass_r1_002.nd2"


imgs = []
with ND2Reader(nd2_path) as images:
    images.bundle_axes = 'zcyx'
    # images.iter_axes = 'v'
    print(images.sizes)
    if 'v' in images.sizes.keys(): # more than 1 image
        print("more than 1 image")
    else: # only 1 image
        print('only 1 image')
    all = (Frame(images))

