# LEGO Detection and Sorting Pipeline

This repository provides a comprehensive adaptive detection system pipeline containing camera calibration, gripping point optimization, and real-time detection. Follow the steps outlined below to set up and use the system effectively.


## Table of Contents
- [Installation](#installation)
- [Step 1: Capture Calibration Image](#step-1-capture-images)
- [Step 2: Calibrate the Camera](#step-2-calibrate-the-camera)
- [Step 3: Capture ROI (Region of Interest) Image](#step-3-optimize-gripping-points)
- [Step 4: Object Detection](#step-4-object-detection)


## Installation

To get started, clone this repository and install the required dependencies.

```bash
git clone https://github.com/rasmushyypia/adaptive-detection.git
cd adaptive-detection
(ensure this is right)
conda create --name adaptive-detection
pip install -r requirements.txt
```

## Step 1: capture images

1. run `capture_images.py`:

2. Follow on-screen instructions
   Calibration Images:
   - Press **SPACEBAR** to capture a calibration image when the chessboard is detected.
   - Capture a total of 20 calibration images for accurate calibration.
   General Images:
   - Press **c** to capture a coordinate frame image. This is a separate image of the calibration grid that is later used to place the coordinate frame
   - Press **x** to capture a test image. Take a picture of a rectangular test shape of which location you know to test the accuracy of calibration
     
3. Additional Information:
   - Calibration images are saved in **data/calibration_images**
   - General images are saved in **data**.
   - Make sure the chessboard **grid size** matches your physical calibration grid.

  
## Step 2: calibrate the camera

1. run `camera_calibration.py`:

2. Calibration Process
   - The script processes each calibration image, detects chessboard corners, and refines their positions.
   - Calculates the camera matrix and distortion coefficients
  
3. Output:
   - Calibration data is saved in **data/calibration_data.pkl**

4. Additional Information:
   - Ensure the **grid_size** and **square_size** match you physical calibration board.
   - The scrips utilizes variables **offset_x** and **offset_y**. These are physical distances from the calibration grid origin to the corner of the table used for           object detection.
   - If you have ractangular shapes placed on known positions on the table, you can use variable **visualize** to see how accurate you calibration was.
   - (add image here showing the measurement of offsets)


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
