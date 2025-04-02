# Adaptive Gripping and Detection System
This repository provides a comprehensive adaptive detection system pipeline containing camera calibration, gripping point optimization, and real-time detection. Follow the steps outlined below to set up and use the system effectively.


## Table of Contents
- [Installation](#installation)
- [Step 1: Camera Calibration](#step-1-camera-calibration)
- [Step 2: Optimize Gripping Points](#step-2-optimize-gripping-points)
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

1. **Start the Camera**  
   In the **Capture Image Parameters** section, set the camera index, capture resolution, and chessboard grid dimensions to match your setup, then click **Start Camera**.

2. **Verify Checkerboard Detection**  
   Check that the live feed in the right panel can detect the checkerboard corners.  
  ⚠️ The checkerboard grid size settings must exactly match the checkerboard pattern being used.

3. **Capture Calibration Images**  
   Click **Capture Calibration Image** (or press **X**) to capture multiple images. Aim for **~20 images** from various angles to improve calibration accuracy.

   - All calibration images (180°calib_XX.jpg`) are saved automatically to `data/calibration_images/`.
   - You can open or delete these images using the **Calibration Image Folder** panel at the bottom left.

### 4. Capture Coordinate Frame Image
Position the larger checkerboard where you want to place the table's (0,0) origin, including the directions of the x- and y-axes. Then press **C** to save the coordinate frame image (`coord_frame_XX.jpg`) to `data/calibration_images`.

### 5. Configure Calibration and Mapping
In the **Calibration & Mapping Parameters** section, set the following according to the checkerboards you're using:
- **Calibration Grid / Square Size (mm)**: Specifies the checkerboard dimensions and physical square size used for **intrinsic calibration**.
- **Mapping Grid / Square Size (mm)**: Specifies the checkerboard dimensions and physical square size used to establish the **table coordinate system**.
- **Flip Mapping Origin**: Changes the origin from the checkerboard's top-left corner (default) to the bottom-right. Useful if your coordinate axes appear flipped after calibration
- **Visualize Calibration**: When enabled, displays the undistorted coordinate frame image with the projected mapping grid.
- **Save Calibration Data**: When enabled, saves the calibration result in `data/` folder as 180°gripping_data.pkl`.

### 6. Run Single or Multi-Image Calibration

1. **Single-Image Calibration**
- Uses **only** the single coordinate frame for both intrinsic calibration and coordinate mapping.
- Calibration and mapping parameters must be **identical** (e.g., same grid size and square size).

2. **Multi-Image Calibration**
- Computes intrinsic parameters using **all** captured calibration images (e.g., `calib_00.jpg` to `calib_19.jpg`).
- Uses the coordinate frame image to define the table's origin and axis orientation.

<img src="/media/calibration_gui_image2.png" alt="calibration gui2" width="600">

## Step 2: Optimize Gripping Points
This step `gripping_point_optimizer.py` to determine optimal two-finger gripper placement around an object defined in a DXF file. It automatically computes an inner (inset) boundary to avoid collisions with the object edges or holes, then allows manual fine-tuning or automatic optimization for gripper positions.

### 1. **Launch the gripping point GUI**  
From the project's **source directory**, run:
```
python gripping_point_optimizer.py
```

### 2. **Open a DXF File**
1. Click **Open DXF** and select a DXF containing a closed polygonal shape.
2. The script extracts:
   - **Outer Boundary** (shown in blue),
   - **Holes** (if present, shown in red)
   - **Inner Contour** (green dashed line), an inset boundary ensuring safe clearance for the gripper
   - 📁 Example DXF files used in this project can be found in the `data/dxf_shapes/` folder.

### 3. **Adjust the Grippers Manually**
1. **Step Size Slider**: Controls how far each gripper moves on a single click.

2. **G1/G2 CW or CCW**: Moves the respective gripper along the inner contour in a clockwise or counterclockwise direction.

3. **Reset**: Returns both grippers to their original, default positions.

### 4. Optimization
Click **Optimize Grippers** to run a **Differential Evolution** algorithm that searches for the most suitable symmetric gripper positions along the inner contour.
The optimization:
- Places one gripper point along the inner contour, and places the second gripper at the point **halfway around the contour** from the first.
- **Minimizes the maximum distance** between the gripper region and the object's outer boundary, encouraging a central and stable grip.
- Applies a **penalty** if either gripper overlaps with a hole, steering the solution away from unsafe regions.

### 5. Save Data
When you are satisfied with the gripping configuration:
1. Click **Save Data** to store the polygon geometry and gripper points in `data/gripping_data.pkl`.
2. This file includes:
   - **final_poly**: The complete Shapely polygon (outer shape + holes)
   - **gripping_points**: A NumPy array of the final (x,y) gripper coordinates.
   - **gripper_distance**: The distance between the two gripping points.
   - **gripper_line_angle**: The angle of the line connecting gripping points.



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
