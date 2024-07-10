import os
import pytest
import numpy as np
from subprocess import call
import tifffile
import pandas as pd
import image_analysis_lib

TEST_IMAGES_DIR = 'test_165um'  
NUCLEI_IDX = 0
MARKER_IDX = 2
HOLE_DIAMETER = 165 
TEST_CSV = 'micropattern_analysis.csv'  

@pytest.fixture
def setup_test_environment():
    """Setup test environment by creating test images directory."""
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    yield
    if os.path.exists(TEST_IMAGES_DIR):
        for file in os.listdir(TEST_IMAGES_DIR):
            file_path = os.path.join(TEST_IMAGES_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(TEST_IMAGES_DIR)
    if os.path.isfile(TEST_CSV):
        os.remove(TEST_CSV)

def generate_test_images(image_files):
    """Generate valid test images into the test directory."""
    for img_file in image_files:
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)  # Create a random 100x100x3 image
        tifffile.imwrite(img_path, data)

@pytest.mark.parametrize("image_files", [
    (['image1.tif', 'image2.tif'])
])
def test_image_analysis(setup_test_environment, image_files):
    """Test image analysis functionality."""
    generate_test_images(image_files)

    call(['python', 'image_analysis.py', TEST_IMAGES_DIR, str(NUCLEI_IDX), str(MARKER_IDX), str(HOLE_DIAMETER)])

    # Check if test_micropattern_analysis.csv was generated
    assert os.path.isfile(TEST_CSV), f"Expected {TEST_CSV} to be created, but it wasn't."

@pytest.mark.parametrize("data, expected_plot_exists", [
    ({'total': [100, 120], 'positive': [60, 80], 'hole_diameter': [165, 165]}, True)
])
def test_plot_data(setup_test_environment, data, expected_plot_exists):
    """Test plotting functionality."""
    # Prepare sample test_micropattern_analysis.csv for testing plotting
    df = pd.DataFrame(data)
    df.to_csv(TEST_CSV, index=False)

    call(['python', 'plot_data.py', TEST_CSV])

    # Check if positive_fraction_vs_hole_diameter.png was generated
    assert os.path.isfile('positive_fraction_vs_hole_diameter.png') == expected_plot_exists

if __name__ == "__main__":
    pytest.main()
