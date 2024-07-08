# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 12:43:07 2024

@author: חן דודיאן
"""

import cv2
import nd2
from matplotlib import pyplot as plt
import numpy as np
from skimage import io, restoration, exposure, img_as_ubyte

#########################################################
###read as nd2 and create np array for each channel
images_path = r"C:\Users\חן דודיאן\OneDrive\מסמכים\biology\MSc\Rotations\Eyal Karzbrun\membrane\Ex12\Exp12\Membrane_on_glass_r1_001.nd2"
images = nd2.imread(images_path)  # read to numpy array


#split channels into np arrays of dtype uint16
dapi = images[:,0,:,:]
phalloidin = images[:,1,:,:]
psmad = images[:,2,:,:]
bf = images[:,3,:,:]

def create_mip_in_range(image, z_start, z_end):

    mip = np.max(image[z_start:z_end], axis=0)
    return mip

###MIP psmad channel across all z stacks 
psmad_MIP = np.max(psmad, axis=0)
bf_MIP = np.max(bf, axis = 0)

plt.imshow(bf_MIP, cmap='gray')

###8bit convertion######
dapi_8bit = img_as_ubyte(exposure.rescale_intensity(dapi))
phalloidin_8bit = img_as_ubyte(exposure.rescale_intensity(phalloidin))
bf_8bit = img_as_ubyte(exposure.rescale_intensity(bf_MIP))
psmad_8bit = img_as_ubyte(exposure.rescale_intensity(psmad_MIP))

plt.imshow(bf_8bit, cmap='gray')


    

