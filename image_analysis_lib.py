import time
import numpy as np
from skimage import exposure, img_as_ubyte
import tifffile
import pandas as pd
import cv2
import os

def open_image(path):

    with tifffile.TiffFile(path) as tif:
    # Read the image data into a numpy array
        image_array = tif.asarray()
    
    #convert image format to 8bit because cv2 lib workes better with 8bit
    image_array = img_as_ubyte(exposure.rescale_intensity(image_array))
        
    return image_array

def segregat_image(model, cyx_image, cyto_channel, nuclei_channel): 

    # Perform cell segmentation on the DAPI image
    # masks, flows, styles, diams = model.eval(cyx_image, diameter=30, channels=[cyto_channel,nuclei_channel])
    masks, flows, styles, diams = model.eval(cyx_image, diameter=30, channels=[nuclei_channel])

    # Convert the masks to a format we can use for analysis
    masks = np.array(masks)

    return masks

def threshold_marker(cyx_image, marker_channel):

    yx_marker_img = cyx_image[marker_channel,:,:]
    
    #perform Otsu threshold
    ret, _ = cv2.threshold(yx_marker_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return ret #return threshold value based on utso labeling method 

def count_positive(mask, marker_img, thresh = 15):
    
    # Iterate through each segmented cell and analyze marker expression
    cell_ids = np.unique(mask)
    positive_cells = 0
    tot_num_cell = len(np.unique(mask)) - 1 # exclude background ID

    for cell_id in cell_ids:
        if cell_id == 0:  # skip background
            continue

        cell_mask = (mask == cell_id)

        cell_marker_intensity = marker_img[cell_mask > 0]
        if np.mean(cell_marker_intensity) > thresh:
            positive_cells += 1
    
    return positive_cells

def document_data(total,positive,hole_diameter,file_path):
    file_name = 'micropattern_analysis.csv'
    file_exists = os.path.isfile(file_name)
    with open(file_name, "a") as file:
        if not file_exists:
            file.write("total,positive,hole_diameter,file_path\n")
        file.write(f"{total}, {positive}, {hole_diameter}, {file_path}\n")

def analyse_files_in_dir(directory_path, model,cyto_idx,nuclei_idx,marker_idx,hole_diameter):

    i=1
    for filename in os.listdir(directory_path):
            
            file_path = os.path.join(directory_path, filename)

            ### start the operation for 1 image

            start_time = time.time()

            image_array = open_image(file_path)

            mask = segregat_image(model, image_array, cyto_idx,nuclei_idx)

            total = len(np.unique(mask))

            thresh = threshold_marker(image_array,marker_idx)

            # positive = count_positive(mask,image_array[marker_idx,:,:],thresh)
            positive = count_positive(mask,image_array[marker_idx,:,:])

            document_data(total,positive,hole_diameter, file_path)

            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f'operation {i} out of {len(os.listdir(directory_path))} is done! duration: {elapsed_time:.2f} seconds' )

            i+=1