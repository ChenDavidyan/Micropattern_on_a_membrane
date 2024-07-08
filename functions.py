import sys
import time
import numpy as np
from skimage import exposure, img_as_ubyte
from cellpose import models
import tifffile
import pandas as pd
import cv2

def open_image(path):

    with tifffile.TiffFile(path) as tif:
    # Read the image data into a numpy array
        image_array = tif.asarray()
    
    #convert image format to 8bit because cv2 lib workes better with 8bit
    image_array = img_as_ubyte(exposure.rescale_intensity(image_array))
        
    return image_array

def segregat_image(model, cyx_image, cyto_channel, nuclei_channel): 

    # Perform cell segmentation on the DAPI image
    masks, flows, styles, diams = model.eval(cyx_image, diameter=30, channels=[cyto_channel,nuclei_channel])

    # Convert the masks to a format we can use for analysis
    masks = np.array(masks)

    return masks

def threshold_marker(cyx_image, marker_channel):

    yx_marker_img = cyx_image[marker_channel,:,:]
    
    #perform Otsu threshold
    ret, _ = cv2.threshold(yx_marker_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return ret #return threshold value based on utso labeling method 

def main():

    start_time = time.time()

    ### start the operation for 1 image 

    image_array = open_image(r'r1_002.nd2 (series 06).tif')

    # Initialize the Cellpose model
    model = models.Cellpose(gpu=False, model_type='cyto')

    mask = segregat_image(model, image_array, 1, 0)
    print(len(np.unique(mask)))
    
    # thresh = threshold_marker(image_array,2)
    # print(thresh)

    ### end of operation for 1 image

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")



if __name__ == "__main__":
    main()