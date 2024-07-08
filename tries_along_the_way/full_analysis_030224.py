# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 18:30:04 2024

@author: חן דודיאן
"""

import cv2
import nd2
from matplotlib import pyplot as plt
import numpy as np
from skimage import io, restoration, exposure, img_as_ubyte
import seaborn as sns
import pandas as pd
from nd2reader import ND2Reader

#########################################################
###read as nd2 and create np array for each channel
# images_path = r"C:\Users\חן דודיאן\OneDrive\מסמכים\biology\MSc\Rotations\Eyal Karzbrun\membrane\Ex12\Exp12\Membrane_on_glass_r1_001.nd2"
# images = nd2.imread(images_path)  # read to numpy array

r1_002_nd2_path = r"C:\Users\חן דודיאן\OneDrive\מסמכים\biology\MSc\Rotations\Eyal Karzbrun\membrane\Ex12\Exp12\Membrane_on_glass_r1_002.nd2"
# r1_002_1 =cv2.imread(r"C:\Users\חן דודיאן\OneDrive\מסמכים\biology\MSc\Rotations\Eyal Karzbrun\membrane\convert_to_tiff\CSU Dapi-50_z0_y0.tiff")

def read_nd2_file(file_path):
    imgs = []
    with ND2Reader(file_path) as images:
        images.bundle_axes = 'zcyx'
        images.iter_axes = 'v'
        for fov in images: # loop over all fields of view
            img = fov[:,:,:,:]
            imgs.append(img)
            print('img appended!')
    return imgs

def split_channels (images):
    #split channels into np arrays of dtype uint16
    dapi = images[:,0,:,:]
    phalloidin = images[:,1,:,:]
    psmad = images[:,2,:,:]
    bf = images[:,3,:,:]
    
    return dapi, phalloidin, psmad, bf

def convert_to_8bit(yx_image):
    
    return img_as_ubyte(exposure.rescale_intensity(yx_image))

def create_mip_in_range(image, z_start, z_end):

    mip = np.max(image[z_start:z_end], axis=0)
    return mip

def MIP (zyx_image):
    
    miped = np.max(zyx_image, axis=0)
    
    return miped

def get_fixel_micron_ratio(path): #the amount of microns per pixel
    with ND2Reader(path) as images:
        ratio = images.metadata['pixel_microns']
    return ratio
    
def adjust_contrast_and_brightness (image, a,b):
    
    new_image = np.zeros(image.shape, image.dtype)
    # alpha = a # Simple contrast control
    # beta = b    # Simple brightness control
    new_image = cv2.convertScaleAbs(image, a, b)
    return new_image

def psmad_area_detector(psmad_yx_img):
    
    #perform Otsu thresho
    ret, mask = cv2.threshold(psmad_yx_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f'Otsu threshold value: {ret}')

    # do connected components processing (segmentation)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, None, None, None, 8, cv2.CV_32S)

    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)
    
    
    #get rid of small background noise (keep only "big" objects)
    for i in range(0, nlabels - 1):
        if areas[i] >= 170:   #keep
            result[labels == i + 1] = 255
   
    
    #fill with white small balck holes inside psmad detected area 
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(result,kernel,iterations = 8)
    

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(dilation,kernel,iterations = 7)

    #find psmad area edges
    img = dilation - erosion

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 800, 800)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #return only edges
    return img

def fit_circle(bf_yx_img):
    #####################################################
    #canny edges detector, fit canny thresholds with sigma & median
    sigma = 0.3
    median = np.median(bf_yx_img)

    lower = int(max(0, (1.0-sigma)* median))
    upper = int(min(255,(1.0+sigma)*median))

    auto_canny = cv2.Canny(bf_yx_img,lower,upper)
    # plt.imshow(auto_canny, cmap='gray')

    ##################################################
    #perform Otsu thresho
    ret, mask = cv2.threshold(auto_canny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, None, None, None, 8, cv2.CV_32S)

    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= 170:   #keep
            result[labels == i + 1] = 255

    ##############################################
    #find and draw circle

    bf_copy = bf_yx_img.copy()
    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(
        result,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=50,
        minRadius=100,
        maxRadius=350
    )

    print(circles)
    print(len(circles))
    # If circles are found, draw them on the original image
    if circles is not None:
        if len(circles == 1):
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw the filled circle
                cv2.circle(bf_copy, (i[0], i[1]), i[2], (255, 0, 0), thickness=3)
                # Draw the center of the circle
                cv2.circle(bf_copy, (i[0], i[1]), 2, (255, 255, 255), 5)

        else:
            print(f'len(circles) = {len(circles)}')
    else:
        print("No circles found.")
    return circles[0]

def get_absolute_distances(locations, circle_coordinates):
    
    y_center = circle_coordinates[0,0]
    x_center = circle_coordinates[0,1]
    distances = np.sqrt((locations[:, 0] - y_center)**2 + (locations[:, 1] - x_center)**2)
    
    return distances

def round_to_closest(value):
    
    radiuses = [200, 175, 160, 140, 100, 85, 70]
    return min(radiuses, key=lambda x: abs(x - value))

def plot_distance_from_hole_edges(absolute_distances, radius, ratio):
   
    # absolute distance minus the given radius
    dis_from_edges = absolute_distances - radius
    sorted_distances = np.sort(dis_from_edges)
    
    sorted_distances = sorted_distances * ratio
    radius = round_to_closest(radius * ratio)
    
    df = pd.DataFrame(sorted_distances,columns=[f"signal distance from microhole edges, r = {radius} um"])
    sns.displot(df, x=f"signal distance from microhole edges, r = {radius} um", kind = 'kde',bw_adjust=0.6)




##########################################################
# ratio = get_fixel_micron_ratio(images_path) #the amount of microns per pixel

# dapi, phalloidin, psmad, bf = split_channels(images)

# psmad_mip = MIP(psmad)
# bf_mip = MIP(bf)

# psmad_8bit = convert_to_8bit(psmad_mip)
# bf_8bit = convert_to_8bit(bf_mip)

# # bf_adj = adjust_contrast_and_brightness(bf_8bit, 1, 2)
# # psmad_adj = adjust_contrast_and_brightness(psmad_8bit, 1, 20)

# psmad_edges = psmad_area_detector(psmad_8bit)
# locations = np.argwhere(psmad_edges == 255)

# circle_coordinates = fit_circle(bf_8bit)
# absolute_distances = get_absolute_distances(locations, circle_coordinates)

# plot_distance_from_hole_edges(absolute_distances, circle_coordinates[0,2],ratio)

##########################################################

all_r1_images = read_nd2_file(r1_002_nd2_path)
ratio = get_fixel_micron_ratio(r1_002_nd2_path) #the amount of microns per pixel

for image in all_r1_images:

    dapi, phalloidin, psmad, bf = split_channels(image)
    
    psmad_mip = MIP(psmad)
    bf_mip = MIP(bf)
    
    psmad_8bit = convert_to_8bit(psmad_mip)
    bf_8bit = convert_to_8bit(bf_mip)
    
    # bf_adj = adjust_contrast_and_brightness(bf_8bit, 1, 2)
    # psmad_adj = adjust_contrast_and_brightness(psmad_8bit, 1, 20)
    
    psmad_edges = psmad_area_detector(psmad_8bit)
    locations = np.argwhere(psmad_edges == 255)
    
    circle_coordinates = fit_circle(bf_8bit)
    absolute_distances = get_absolute_distances(locations, circle_coordinates)
    
    plot_distance_from_hole_edges(absolute_distances, circle_coordinates[0,2],ratio)
    