import cv2
import numpy as np
from cellpose import models, utils, io

# Load the DAPI-stained image and the marker-stained image
dapi_image_path = 'path/to/dapi_image.png'
marker_image_path = 'path/to/marker_image.png'

dapi_image = cv2.imread(dapi_image_path, cv2.IMREAD_GRAYSCALE)
marker_image = cv2.imread(marker_image_path, cv2.IMREAD_GRAYSCALE)

# Initialize the Cellpose model
model = models.Cellpose(gpu=False, model_type='nuclei')

# Perform cell segmentation on the DAPI image
masks, flows, styles, diams = model.eval(dapi_image, diameter=None, channels=[0,0])

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
    if is_positive(cell_mask, marker_image):
        positive_cells += 1

# Calculate the proportion of marker-positive cells
proportion_positive = positive_cells / total_cells
print(f"Proportion of cells positive for the marker: {proportion_positive:.2f}")
