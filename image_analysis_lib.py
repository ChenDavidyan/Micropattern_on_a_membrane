import time
import numpy as np
from skimage import exposure, img_as_ubyte
import tifffile
import os


def open_image(path):
    with tifffile.TiffFile(path) as tif:
        # Read the image data into a numpy array
        image_array = tif.asarray()

    # convert image format to 8bit for faster operation
    image_array = img_as_ubyte(exposure.rescale_intensity(image_array))

    return image_array


def segregate_image(
    model,
    nuclei_img,
):
    # Perform cell segmentation on the DAPI image
    masks, flows, styles, diams = model.eval(nuclei_img, diameter=None, channels=[0, 0])

    # Convert the masks to a format we can use for analysis
    masks = np.array(masks)

    return masks

def count_positive(
    mask, marker_img, thresh=15
):  # thresh = 15 works best based on analysis of background intensity in NC samples
    # Iterate through each segmented cell and analyze marker expression
    cell_ids = np.unique(mask)
    positive_cells = 0
    tot_num_cell = len(np.unique(mask)) - 1  # exclude background ID

    for cell_id in cell_ids:
        if cell_id == 0:  # skip background
            continue

        cell_mask = mask == cell_id

        cell_marker_intensity = marker_img[cell_mask > 0]
        if np.mean(cell_marker_intensity) > thresh:
            positive_cells += 1

    return positive_cells


def document_data(total, positive, hole_diameter, file_path, file_name="micropattern_analysis.csv"):
    # file_name = "micropattern_analysis.csv"
    file_exists = os.path.isfile(file_name)
    with open(file_name, "a") as file:
        if not file_exists:
            file.write("total,positive,hole_diameter,file_path\n")
        file.write(f"{total}, {positive}, {hole_diameter}, {file_path}\n")


def analyse_files_in_dir(directory_path, model, nuclei_idx, marker_idx, hole_diameter):
    i = 1
    print('Operation has started, it will take about 70-120 seconds for each image. Please be patient')
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        ### start the operation for 1 image

        start_time = time.time()

        image_array = open_image(file_path)

        mask = segregate_image(model, image_array[nuclei_idx, :, :])

        total = len(np.unique(mask))

        positive = count_positive(mask, image_array[marker_idx, :, :])

        document_data(total, positive, hole_diameter, file_path)

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(
            f"operation {i} out of {len(os.listdir(directory_path))} is done! duration: {elapsed_time:.2f} seconds"
        )

        i += 1
