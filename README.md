# Adaptive Gripper System

This repository provides a comprehensive adaptive detection system pipeline containing camera calibration, gripping point optimization, and real-time detection. Follow the steps outlined below to set up and use the system effectively.


## Table of Contents
- [Installation](#installation)
- [Step 1: Capture Calibration Image](#step-1-capture-images)
- [Step 2: Calibrate the Camera](#step-2-calibrate-the-camera)
- [Step 3: Capture ROI (Region of Interest) Image](#step-3-optimize-gripping-points)
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

## Step 1: capture images
This step involves capturing calibration and general images using round-cornered Radon checkerboards to ensure accurate camera calibration and coordinate frame definition.
This project utilized two different sized checkerboards, each serving a specific purpose in the calibration process. Both the **SVG** and **JPG** files for these checkerboards
are located in the `adaptive-detection/data/calibration_boards/` directory.

   **Medium Checkerboard** (`checkerboard_radon_medium.jpg`) is utilized for capturing the **20 calibration images** required for accurate camera calibration.
   
   **Large Checkerboard** (`checkerboard_radon_large.jpg`) is utilized for capturing the **coordinate_frame_image.jpg**, used in defining the coordinate frame on the table.

<img src="/data/calibration_boards/checkerboard_radon_medium.jpg" alt="Calibration Image" width="500">

**Instructions**
1. Verify that the checkerboard used for main camera calibration matches the grid size specified in the script (default is 10x7).
2. run `capture_images.py`
3. Follow on-screen instructions:
   - Press `SPACEBAR` to capture calibration images when the checkerboard is detected.
      - Capture a total of **20 calibration** images for accurate calibration.
   - Align checkerboard with the table and press `c` to capture a `coordinate_frame_image.jpg`
   - Press `x` to capture a `test_image.jpg` of a rectangular object positioned at a known location relative to the table.
      - This is optional, but can be used in the next step to verify the accuracy of the calibration. 
4. Additional Information:
   - Calibration images are stored in `data/calibration_images`
   - General images are stored in `data/`.


## Step 2: calibrate the camera
This step involves computing the camera's intrinsic parameters and distortion coefficients using the previosly captured calibration image. Proper calibration allows for undistorting images and accurately mapping image points to real-world coordinates.

1. **Configure calibration parameters**:
  Open the `camera_calibration.py` script and locate the **Calibration Parameters** section within the `main()` function. Ensure the following parameters are correctly configured:

    - **Calibration parameters** (`calib_grid_size` and `calib_square_size`) define the number of internal corners per chessboard row and column, and size of each chessboard square in millimeters respectively.
    - **Mapping parameters** (`mapping_grid_size` and `mapping_square_size`) are similar to calibration parameters but used for defining the coordinate frame.
      - These might differ based on your setup.
    - **Offsets** (`offset_x` and `offset_y`) define the physical offset in millimeters from the chessboard's origin to the tables's corner.
      - Used to move the origin to the corner of table for ease of use.
      - To better understand how `offset_x` and `offset_y` affect the origin. The orange dashed lines represent the offsets from the checkerboards origin

2. Run `camera_calibration.py`

   - The script includes a `visualize` flag. When set to `True`, the script will display intermediate steps such as detected chessboard corners and annotated images to help verify the calibration process.
   - The script will display camera matrix, distortion coefficients, and the mean reprojection error, which can indicate the calibration accuracy

3. Additional Information:

   - The calibration data, including offset information is saved to `data/calibration_data.pkl` file. 



## Step 3: optimize gripping points
Calculate optimal positions for the four grippers based on the object's shape.
The gripping point optimization is handled by two scripts:
   - `gripping_point_symmetric.py`: ensures symmetrical gripping points.
   - `gripping_point_non_symmetric.py`: Allows independent positioning of grippers.

1. Provide object shape:
   - the script reads objects shapes from a DXF file specified in the script. Place the DXF file in `data/dxf_shapes/`
   - 
2. Provide the script `gripper_radius` variable so it can account for the size of the gripper head.

3. Run either script. Interact with the GUI. GUI visualizes the gripping points on the object's contour
   - Click the optimize button to maximize the safe gripping area
   - Click the save button to create a `gripping_data.csv` in `data` folder.


## Step 4: object detection

1. Configure Parameters
   - Ensure calibration data and gripping data paths are correctly set in the script
   - select `CAMERA_INDEX` that matches with the camera you are using

2. Select Operation Mode:
   - Static image test: Set `USE_STATIC_IMAGE = True` to process a single test image.
   - Live Detection: Set `USE_STATIC_IMAGE = False` to enable real-time detection via the camera.
  
Features
1. Shape Detection
   - Image Preprocessing: Converts images to grayscale, applies Gaussian blur to reduce noise, and utilizes Canny edge detection to highlight object boundaries.
   - Contour Extraction: Detects contours from the edge-detected image and compares them with the reference polygon using OpenCV's `matchShapes` method. Detected contours are filtered based on similarity score and area-difference to the reference image.
   - Centroid Calculation: Calculates image moments of each detected contour to determine the centroid coordinates, providing the object's precise location within the frame.
   - Angle optimization: Utilizes the Differential Evolution algorithm to optimize the rotation angle and positional shifts of the reference polygon, aiming to maximize the IoU with detected contours.
   - Uses `Logging` module to record detailed debug information of the program. Includes performance tracking of the process by taking time.
