# Adaptive Gripper System

This repository provides a comprehensive adaptive detection system pipeline containing camera calibration, gripping point optimization, and real-time detection. Follow the steps outlined below to set up and use the system effectively.


## Table of Contents
- [Installation](#installation)
- [Step 1: Capture Calibration Image](#step-1-capture-images)
- [Step 2: Calibrate the Camera](#step-2-calibrate-the-camera)
- [Step 3: Optimize Gripping Points](#step-3-optimize-gripping-points)
- [Step 4: Object Detection](#step-4-object-detection)


## Installation

To get started, clone this repository and install the required dependencies:

```
bash
git clone https://github.com/rasmushyypia/adaptive-detection.git
cd adaptive-detection
conda create --name adaptive-detection python=3.8
conda activate adaptive-detection
pip install -r requirements.txt
```

## Step 1: Capture Images
This step involves capturing calibration and general images using round-cornered Radon checkerboards to ensure accurate camera calibration and coordinate frame definition.
This project utilized two different sized checkerboards, each serving a specific purpose in the calibration process. Both the **SVG** and **JPG** files for these checkerboards
are located in the `adaptive-detection/data/calibration_boards/` directory.

   **Medium Checkerboard** (`checkerboard_radon_medium.jpg`) is utilized for capturing the **20 calibration images** required for accurate camera calibration.
   
   **Large Checkerboard** (`checkerboard_radon_large.jpg`) is utilized for capturing the **coordinate_frame_image.jpg**, used in defining the coordinate frame on the table.

<img src="/data/calibration_boards/checkerboard_radon_medium.jpg" alt="Calibration Image" width="500">

#### 1. Verify that the checkerboard used for main camera calibration matches the grid size specified in the script (default is 10x7).

#### 2. run `capture_images.py`

#### 3. Follow on-screen instructions:
 - Press `SPACEBAR` to capture calibration images when the checkerboard is detected.
    - Capture a total of **20 calibration** images for accurate calibration.
 - Align checkerboard with the table and press `c` to capture a `coordinate_frame_image.jpg`
 - Press `x` to capture a `test_image.jpg` of a rectangular object positioned at a known location relative to the table.
    - This is optional, but can be used in the next step to verify the accuracy of the calibration.

#### 4. Additional Information:
 - Calibration images are stored in `data/calibration_images`
 - General images are stored in `data/`.


## Step 2: Calibrate the Camera
This step involves computing the camera's intrinsic parameters and distortion coefficients using the previosly captured calibration image. Proper calibration allows for undistorting images and accurately mapping image points to real-world coordinates.

#### 1. Configure calibration parameters: 
Open the `camera_calibration.py` script and locate the **Calibration Parameters** section within the `main()` function. Ensure the following parameters are correctly configured:
  - **Calibration parameters** (`calib_grid_size` and `calib_square_size`) define the number of internal corners per chessboard row and column, and size of each chessboard square in millimeters respectively.
  - **Mapping parameters** (`mapping_grid_size` and `mapping_square_size`) are similar to calibration parameters but used for defining the coordinate frame.
    - These might differ based on your setup.
  - **Offsets** (`offset_x` and `offset_y`) define the physical offset in millimeters from the chessboard's origin to the tables's corner.
    - Used to move the origin to the corner of table for ease of use.
    - To better understand how `offset_x` and `offset_y` affect the origin. The orange dashed lines in the image below indicate the offsets move the origin from the chessboard's origin to the table's edge.

<img src="/media/offset_image.jpg" alt="offset_image" width="800">

#### 2. Run `camera_calibration.py`
   - The script includes a `visualize` flag. When set to `True`, the script will display intermediate steps such as detected chessboard corners and annotated images to help verify the calibration process.
   - The script will display camera matrix, distortion coefficients, and the mean reprojection error, which can indicate the calibration accuracy.

#### 3. Additional Information:
   - The calibration data, including offset information is saved to `data/calibration_data.pkl` file. 


## Step 3: Optimize Gripping Points
This step involves selecting and refining the optimal points on object's contour where the grippers will make contact.

#### 1. Load and prepare contour data
  - The process begins by loading the contour data of the object to be gripped. This can be done either by:
    - Importing the object's outline from a **DXF file**.
    - Selecting from a set of **predefined shapes** for testing purposes.
     
#### 2. Calculate centroid and internal angles
  - Calculate the geometric center of the contour and center the shape around origin.
  - Computes the internal angles at each vertex of the contour to identify significant points for selecting initial gripping points.
  
#### 3. GUI for adjusting and optimizing the gripping points.
  - Displays the object's contour, initial gripping points, and safe quadrilateral areas.
  - Includes sliders and buttons to select step size and move gripping points clockwise or counterclockwise along the contour.
  - Utilizes **Sequential Least Squares Programming** for optimization.
     - Optimizes either `safe quadrilateral area` or `max offset` from outer contour.
  - Currently program has two versions `GrippingPointSymmetric.py` and `GrippingPointNonSymmetric.py` visualized in images below.

<img src="/media/symmetric_gui.png" alt="offset_image" width="600">
<img src="/media/non_symmetric_gui.png" alt="offset_image" width="600">

#### 4. Additional Information:     
 - The gripping points and polygon coordinates are saved to a **CSV file** (`data/gripping_data.csv`)


## Step 4: Object Detection
Object detection is the main program in this project that utilizes data generated in earlier steps to identify and localize objects within the camera's field of view. 

#### 1. Load calibration and gripping data
  - The object detection utilizes `calibration_data.pkl` and `gripping_data.csv` generated in earlier scripts.

#### 2. Configure detection parameters
  - `THRESHOLD`: similarity threshold for contour matching
  - `AREA_TOLERANCE`: tolerance for area difference between detected and reference contours.
  - `USE_STATIC_IMAGE`: Set to **True** to run detection on a static image, or **False** to use live webcam feed.
  - furthermore `optimize_angle' function contains parameters relevant to **Differential Evolution** used in optimizing the detected contour angle.

#### 3. Run `detection_modularized.py`

### How it works:
  - **Image Preprocessing**: Converts images to grayscale, applies Gaussian blur to reduce noise, and utilizes Canny edge detection to highlight object boundaries.
  - **Contour Detection and Matching**: Detects contours from the preprocessed image and compares them with the reference polygon using OpenCV's `matchShapes` method. Detected contours are filtered based on similarity score and area difference to the reference image.
  - **Centroid Calculation**: Calculates centroid of the detected contour to provide precise location (x, y) on the table.
  - **Angle Optimization**: Utilizes Differential Evolution algorithm to calculate rotation angle and small positional shifts by maximizing the Intersection over Union (IoU) between reference and detected contours.
  - **Data to Robot**: The calculated coordinates (x, y) and rotation angle (R) are provided to the robot to move it to the gripping position.
  - **Gripper Positioning**: The gripper jig, mounted on the 6-axis robot, utilizes the gripping points calculated in GrippingPointSymmetric.py. Servos within the jig adjust each individual gripper's position based on these coordinates.

#### 4. Additional information
  - The script calculates time that it took to optimize, which can be used to find good parameter values.
  
