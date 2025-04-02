# Adaptive Gripping and Detection System
This repository provides a comprehensive adaptive detection system pipeline containing camera calibration, gripping point optimization, and real-time detection. Follow the steps outlined below to set up and use the system effectively.


## Table of Contents
- [Installation](#installation)
- [Step 1: Camera Calibration](#step-1-camera-calibration)
- [Step 2: Calibrate the Camera](#step-2-calibrate-the-camera)
- [Step 3: Optimize Gripping Points](#step-3-optimize-gripping-points)
- [Step 4: Object Detection](#step-4-object-detection)


## Installation

To get started, clone the repository and set up the conda environment with all necessary dependencies using the `environment.yml` file:

```
git clone https://github.com/rasmushyypia/adaptive-detection.git
cd adaptive-detection/
conda env create -f environment.yml
cd src/
conda activate adaptive-detection
```

## Step 1: Camera Calibration
This step involves using `calibration_gui.py` to capture checkerboard images, calibrate the camera, and establish a coordinate frame on the table. By following the instructions, you will obtain both the camera's intrinsic parameters (e.g., focal length, distortion coefficients) and coordinate frame mapping for subsequent operations. 

### 1. Prepare the Checkerboards
Two **round-cornered Radon checkerboards are used:**
- **Small Checkerboard** (`checkerboard_radon_small.jpg`): Used to capture ~20 images from varying angles to compute the camera's intrinsics.
- **Large Checkerboard** (`checkerboard_radon_large.jpg`): Used to capture image for defining the coordinate frame on the table.

SVG files for the checkerboards used in this demo are located in the `adaptive-detection/media/calibration_boards`
<img src="/media/checkerboard_radon_small.png" alt="small calibration board" width="600">

### 2. Launch the Calibration GUI
Run the following command from the project's **source directory**, run:
```
python calibration_gui.py
```
A window will appear, showing a live camera feed on the right and parameter settings on the left.
<img src="/media/calibration_gui_image.png" alt="calibration gui" width="600">

### 3. Capture Calibration Images
**1. Start the Camera**: In the **Capture Image Parameters** section, set the camera index, capture resolution, and chessboard grid dimensions to match your setup, then click **Start Camera**.
**2. Verify Checkerboard Detection**: Check that the live feed in the right panel can detect the checkerboard corners. The checkerboard grid size settings must exactly match the checkerboard pattern being used.
**3. Capture Calibration Images**: Click **Capture Calibration Image** (or press **X**) to capture multiple images. Aim for **~20 images** from various angles to improve calibration accuracy.
- All calibration images (`calib_XX.jpg`) are saved automatically to `data/calibration_images`.
- You can open or delete these images using the **Calibration Image Folder** panel at the bottom left.

### 3. Capture Calibration Images

1. **Start the Camera**  
   In the **Capture Image Parameters** section, set the camera index, capture resolution, and chessboard grid dimensions to match your setup, then click **Start Camera**.

2. **Verify Checkerboard Detection**  
   Check that the live feed in the right panel can detect the checkerboard corners.  
  ⚠️ The checkerboard grid size settings must exactly match the checkerboard pattern being used.

3. **Capture Calibration Images**  
   Click **Capture Calibration Image** (or press **X**) to capture multiple images. Aim for **~20 images** from various angles to improve calibration accuracy.

   - All calibration images (`calib_XX.jpg`) are saved automatically to `data/calibration_images/`.
   - You can open or delete these images using the **Calibration Image Folder** panel at the bottom left.


### 4. Capture Coordinate Frame Image
Position the larger checkerboard where you want to place the table's (0,0) origin, including the directions of the x- and y-axes. Then press **C** to save the coordinate frame image (`coord_frame_XX.jpg`) to `data/calibration_images`.

### 5. Configure Calibration and Mapping
In the **Calibration & Mapping Parameters** section, set the following according to the checkerboards you're using:
- **Calibration Grid / Square Size (mm)**: Specifies the checkerboard dimensions and physical square size used for **intrinsic calibration**.
- **Mapping Grid / Square Size (mm)**: Specifies the checkerboard dimensions and physical square size used to establish the **table coordinate system**.
- **Flip Mapping Origin**: Changes the origin from the checkerboard's top-left corner (default) to the bottom-right. Useful if your coordinate axes appear flipped after calibration
- **Visualize Calibration**: When enabled, displays the undistorted coordinate frame image with the projected mapping grid.
- **Save Calibration Data**: When enabled, saves the calibration result in `data/` folder.

### 6. Run Single or Multi-Image Calibration

1. **Single-Image Calibration**
- Uses **only** the single coordinate frame for both intrinsic calibration and coordinate mapping.
- Calibration and mapping parameters must be **identical** (e.g., same grid size and square size).

2. **Multi-Image Calibration**
- Computes intrinsic parameters using **all** captured calibration images (e.g., `calib_00.jpg` to `calib_19.jpg`.
- Uses the coordinate frame image to define the table's origin and axis orientation.






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

<img src="/media/object_detection_result.png" alt="detection_result" width="1000">
