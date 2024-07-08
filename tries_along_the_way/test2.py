import cv2
import tifffile
import numpy as np
from cellpose import models, utils, io
from skimage import io, restoration, exposure, img_as_ubyte

file_path = r"C:\work\image_analysis_project\r1_002.nd2 (series 06).tif"

with tifffile.TiffFile(file_path) as tif:
# Read the image data into a numpy array
    image_array = tif.asarray()

image_array = img_as_ubyte(exposure.rescale_intensity(image_array))

# Separate the channels
dapi = image_array[0]
phalloidin = image_array[1]
psmad = image_array[2]
brightfield = image_array[3]

# Initialize the Cellpose model
model = models.Cellpose(gpu=False, model_type='cyto')

# Perform cell segmentation on the DAPI image
masks, flows, styles, diams = model.eval(image_array, diameter=30, channels=[1,0])

# Convert the masks to a format we can use for analysis
masks = np.array(masks)

def is_positive(cell_mask, marker_image, threshold=128):
    """
    Determine if a cell is positive for the marker.
    
    Args:
    - cell_mask (np.ndarray): The mask of the cell.
    - marker_image (np.ndarray): The marker image.
    - threshold (int): The intensity threshold to consider a cell as marker-positive.
    
    Returns:
    - bool: True if the cell is positive for the marker, False otherwise.
    """
    cell_marker_intensity = marker_image[cell_mask > 0]
    return np.mean(cell_marker_intensity) > threshold

# Iterate through each segmented cell and analyze marker expression
cell_ids = np.unique(masks)
positive_cells = 0
total_cells = len(cell_ids) - 1  # exclude background ID

for cell_id in cell_ids:
    if cell_id == 0:  # skip background
        continue
    
    cell_mask = (masks == cell_id)
    if is_positive(cell_mask, psmad):
        positive_cells += 1

# Calculate the proportion of marker-positive cells
proportion_positive = positive_cells / total_cells
print(f"Proportion of cells positive for the marker: {proportion_positive:.2f}")

print('positive: ', positive_cells)
print('total: ', total_cells)
