from cellpose import models, io
import io


import cv2
import nd2
from matplotlib import pyplot as plt
import numpy as np
from skimage import io, restoration, exposure, img_as_ubyte

image_files = io.load_images('<path_to_your_images>')

def split_channels (images):
    #split channels into np arrays of dtype uint16
    dapi = images[:,0,:,:]
    phalloidin = images[:,1,:,:]
    psmad = images[:,2,:,:]
    bf = images[:,3,:,:]
    
    return dapi, phalloidin, psmad, bf

