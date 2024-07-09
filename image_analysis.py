import image_analysis_lib
import sys
from cellpose import models

def main():

    if len(sys.argv) != 6:
        print("Usage: python images_analyse.py DIRECTORY CYTO NUCLEI MARKER DIAMETER")
        sys.exit(1)

    images_directory = sys.argv[1]
    cyto_idx = int(sys.argv[2])
    nuclei_idx = int(sys.argv[3])
    marker_idx = int(sys.argv[4])
    hole_diameter = int(sys.argv[5])

    model = models.Cellpose(gpu=False, model_type='cyto')

    image_analysis_lib.analyse_files_in_dir(images_directory, model,cyto_idx,nuclei_idx,marker_idx,hole_diameter)

if __name__ == "__main__":
    main()