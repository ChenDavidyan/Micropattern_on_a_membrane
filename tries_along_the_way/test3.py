import sys
import numpy as np
from skimage import exposure, img_as_ubyte
from cellpose import models
import pandas as pd
import cv2

images_directory = sys.argv[1]
cyto_idx = sys.argv[2]
nuclei_idx = sys.argv[3]
marker_idx = sys.argv[4]
hole_diameter = sys.argv[5]

def convert_to_8bit(cyx_image): #works with xy, need to make sure also works with cxy

    return img_as_ubyte(exposure.rescale_intensity(cyx_image))

def segregat_image(cyx_image, cyto_channel, nuclei_channel): #everytime I call the function I create a new cellpose model. this is not best, better way wiuld be to create one once and work with it

    # Initialize the Cellpose model
    model = models.Cellpose(gpu=False, model_type='cyto')

    # Perform cell segmentation on the DAPI image
    masks, flows, styles, diams = model.eval(cyx_image, diameter=30, channels=[cyto_channel,nuclei_channel])

    # Convert the masks to a format we can use for analysis
    masks = np.array(masks)

    return masks


def utso_labeling_marker(cyx_image,marker_channel):
    
    ret, thresh = cv2.threshold(cyx_image[marker_channel], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ret, thresh

def is_positive(thresh, masks):
    pass

def save_image_data_csv(total, positive, fraction):
    pass

