import image_analysis_lib
import sys
from cellpose import models


def main():
    if len(sys.argv) != 5:
        print("Usage: python images_analyse.py DIRECTORY NUCLEI MARKER DIAMETER")
        sys.exit(1)

    images_directory = sys.argv[1]
    nuclei_idx = int(sys.argv[2])
    marker_idx = int(sys.argv[3])
    hole_diameter = int(sys.argv[4])

    model = models.Cellpose(gpu=True, model_type="cyto3")

    image_analysis_lib.analyse_files_in_dir(
        images_directory, model, nuclei_idx, marker_idx, hole_diameter
    )


if __name__ == "__main__":
    main()
