# image_analysis_project

## Introduction

'Micropattern on a membrane' is a local signaling system for in-vitro study of symmetry-breaking events. The system consists of a polystyrene plate with a hole, to which a porous polyester membrane is attached, allowing media to pass from both sides. Another layer with predesigned microholes seals the membrane's pores except for these microholes. On top of each microhole, cells are seeded as micropatterns, ensuring the microhole area is contained within the micropattern area.

This setup allows for the introduction of two types of media to the cells: the upper media, available to all the cells, and the bottom media, available only to the cells on top of the microhole.

![alt text](micropattern_on_a_membrane_explaination.png "")


## Project goal

In this project, we aimed to study the effect of BMP4 on human pluripotent stem cells (hPSC) using the 'Micropattern on a membrane' system, focusing on the micropattern diameter. hPSCs were exposed to BMP4 in the bottom media for 24 hours, followed by fixation and antibody staining.

## Usage

to analyse your images, run the analysis script:

```
python images_analyse.py DIRECTORY CYTO NUCLEI MARKER DIAMETER
```

* `DIRECTORY` - directory to the folder of your images
* `CYTO` - index of the cyto channel in your images
* `NUCLEI` - index of the NUCLEI channel in your images 
* `MARKER` - index of the MARKER channel in your images 
* `DIAMETER` - the microhole diameter. all images in the same folder should be in the same microhole size.

If you wish to plot all the data from the images you've analyzed, run the plot script:

```
python plot_data.py
```

### input:

The input of the program is a directory to a folder of micropattern images in .tiff format. The images should be in the shape of [cyx] - `c` for channel, `y` for y-axis, and `x` for x-axis.

### output:
the output of the `images_analyse.py` script is a csv file consis of the following data:

| total  | positive | hole_diameter | file_path |
| ------ |:--------:| :------------:| :--------:|
| total number of cells | number of positive cells to the marker | diameter of the microhole |

* `total` - total number of cells
* `positive` - number of positive cells to the marker
* `hole diameter` - the diameter of the microhole

The output of the `plot_data.py` script is a `.png` file of a boxplot of the positive cells percentages in each micropattern, based on the microhole diameter.

## Dependencies

Use the following command to install the required dependencies:
```
pip install -r requirements.txt
```

## Test the program:
Test the program using pytest:
```
pytest
```

## Acknowledgment

This project was originally implemented as part of the [Python programming course](https://github.com/szabgab/wis-python-course-2024-04) at the [Weizmann Institute of Science](https://www.weizmann.ac.il/) taught by [Gabor Szabo](https://szabgab.com/).