# Adaptive Gripping and Detection System
This repository provides a complete pipeline for adaptive gripping and object detection. It combines three key stages: **camera calibration**, **gripping point optimization**, and **object detection**. The system is designed for a **custom variable gripper** and can be integrated with **Universal Robots (UR5)** using a dedicated URCap plugin (`/media/VariableGripper-2.0.urcap`). Additionally, a standalone detection version is available, compatible with any camera and laptop/PC capable of running Python and OpenCV.

<!-- Example of three images side-by-side --> <div style="display: flex; justify-content: space-around; align-items: center;"> <img src="/media/red.png" alt="Robot System" style="width: 30%; max-width: 200px;"> <img src="/media/green.PNG" alt="Custom Gripper" style="width: 30%; max-width: 200px;"> <img src="/media/blue.png" alt="Detection Window" style="width: 30%; max-width: 200px;"> </div>


## Table of Contents
- [Installation](#installation)
- [Camera Calibration](#camera-calibration)
- [Gripping Point Optimization](#gripping-point-optimization)
- [Object Detection](#object-detection)



## Installation 
### 1. Clone the Repository
```
git clone https://github.com/rasmushyypia/adaptive-detection.git
cd adaptive-detection/

```


### 2. Create and Activate the Conda Environment

```
conda env create -f environment.yml
cd src/
conda activate adaptive-detection

```



## Camera Calibration
This step uses `calibration_gui.py` to capture checkerboard images, calibrate the camera, and establish the table's coordinate frame. After calibration, you'll have the camera's intrinsic parameters (e.g., focal lenght, distortion coefficients) and mapping for later operations.
 

### 1. Prepare the Checkerboards

Two **round-cornered Radon checkerboards** are used:

- **Small Checkerboard** (`checkerboard_radon_small.jpg`): Used to capture ~20 images from varying angles to compute the camera's intrinsics.
- **Large Checkerboard** (`checkerboard_radon_large.jpg`): Used to capture an image to define table's coordinate frame.

SVG files for the checkerboards used in this demo are located in the `adaptive-detection/media/calibration_boards`

<div style="position: relative; display: inline-block;">
  <img src="/media/checkerboard_radon_small.png" alt="small calibration board" width="700">
  <div style="
      position: absolute;
      top: 20px;
      left: 20px;
      background: rgba(255,255,255,0.7);
      padding: 10px;">
  </div>
</div>


### 2. Launch the Calibration GUI
From the project's **source directory**, run:
```
python calibration_gui.py
```

<div style="position: relative; display: inline-block;">
  <img src="/media/calibration_gui_image.png" alt="small calibration board" width="700">
  <div style="
      position: absolute;
      top: 20px;
      left: 20px;
      background: rgba(255,255,255,0.7);
      padding: 10px;">
  </div>
</div>


### 3. Capture Calibration Images
1. **Start the Camera**  
   In the **Capture Image Parameters** section, set the camera index, capture resolution, and chessboard grid dimensions to match your setup, then click **Start Camera**.

2. **Verify Checkerboard Detection**  
   Ensure that the live feed visualizes checkerboard corners clearly.  
  ⚠️ _Important: The checkerboard grid size setting must exactly match the pattern on your checkerboard._

3. **Capture Images**  
   Click **Capture Calibration Image** (or press **X**) to take images. Aim for **~20 images** from various angles to improve calibration accuracy. All images (e.g., `calib_XX`) are by default saved to `data/calibration_images/`. Images can be viewed or deleted via the **Calibration Image Folder** panel in the GUI.


### 4. Capture Coordinate Frame Image
Place the larger checkerboard at the desired location to define the table's (0,0) origin, including the x- and y-axis directions. Once positioned correctly, press **C** to capture and save the coordinate frame image (`coord_frame_XX.jpg`) to `data/calibration_images`.

🤖 _Note: If you plan to use the vision system with a robot, now is a good time to define a corresponding frame the robot that aligns with the camera's frame._


### 5. Configure Calibration Settings
In the **Calibration & Mapping Parameters** section, set these options according to the checkerboards you're using:
- **Calibration Grid / Square Size (mm)**: Specifies the checkerboard dimensions used for **intrinsic calibration**.
- **Mapping Grid / Square Size (mm)**: Specifies the checkerboard dimensions used to establish the **table's coordinate system**.
- **Flip Mapping Origin**: Changes the origin from the checkerboard's top-left corner (default) to the bottom-right. Useful if your coordinate axes appear flipped after calibration
- **Visualize Calibration**: When enabled, displays the undistorted coordinate frame image with the projected coordinate frame axes in the camera feed window.
- **Save Calibration Data**: When enabled, saves the calibration result in `data/` folder as gripping_data.pkl`.


### 6. Run Single or Multi-Image Calibration
**Single-Image Calibration**
- Uses a single coordinate frame image for both intrinsic calibration and coordinate mapping.
- Requires that calibration and mapping parameters are **identical** (eg., same grid size and square size).

**Multi-Image Calibration**
- Computes intrinsic parameters using **all** captured calibration images (e.g., `calib_00.jpg` to `calib_19.jpg`).
- Uses the coordinate frame image to define the table's origin and axis orientation.

<div style="position: relative; display: inline-block;">
  <img src="/media/calibration_gui_image2.png" alt="calibration gui2" width="700">
  <div style="
      position: absolute;
      top: 20px;
      left: 20px;
      background: rgba(255,255,255,0.7);
      padding: 10px;">
  </div>
</div>



## Gripping Point Optimization
This step runs `gripping_point_optimizer.py` to find a safe two-finger gripper placement for an object from a DXF file. It calculates a safe inner boundary to avoid hitting the object's edges or holes and lets you adjust the gripper positions by hand or automatically.


### 1. **Launch the gripping point GUI**  
From the project's **source directory**, run:
```
python gripping_point_optimizer.py
```


### 2. **Open a DXF File**
1. Click **Open DXF** and select a DXF containing **a closed polygon**.
2. The script extracts:
   - **Outer Boundary** (shown in blue),
   - **Holes** (if present, shown in red)
   - **Inner Contour** (green dashed line) – an inset boundary ensuring safe clearance for the gripper
- 📁 _DXF files used in this project can be found in the `data/dxf_shapes/` folder._


### 3. **Adjust the Grippers Manually**
- **Step Size Slider**: Controls how much each gripper moves on a single click.

- **G1/G2 CW or CCW**: Moves the respective gripper along the inner contour in a clockwise or counterclockwise direction.

- **Reset**: Returns both gripping points to default positions.


### 4. Optimization
When you click **Optimize Grippers**, the program automatically finds the best two-finger grip positions using [Differential Evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html) algorithm. Here's how it works:

- **Symmetric Gripper Placement**: The optimization starts by treating the position of first gripper as a variable along the inner contour. The second gripper is always placed exactly halfway around the inner contour, ensuring grip is symmetrical.

- **Objective function**: Each gripper is modeled as a circle (representing the suction cup). The circles are combined into a convex hull, and the maximum distance from this hull to the object's outer boundary is calculated. The objective is to minimize this distance to achieve a centered and stable grip.

- **Penalty for Unsafe Areas**: If a suction cup (circle) overlaps any holes in the object, the function applies a penalty that increases with the degree of overlap. This steers the solution away from unsafe gripping positions.

- **Optimization Process:**: Differential Evolution adjusts the gripper position along the contour until combine objective (base measurement + penalties) is minimized.


### 5. Save Data
Once you are satisfied with the gripping configuration:
1. Click **Save Data** to store the polygon geometry and gripper points.
2. The data is saved in `data/gripping_data.pkl` and includes:
   - **final_poly**: A Shapely polygon representing the object's full outline, including holes.
   - **gripping_points**: An array of (x, y) coordinates for the optimized suction cup positions
   - **gripper_distance**: The straight-line distance between the two gripping points.
   - **gripper_line_angle**: The baseline orientation of line connecting the gripping points, needed to determine final alignment angle during real-time detection.

<div style="position: relative; display: inline-block;">
  <img src="/media/gripping_point_optimizer_image.png" alt="gripping_point_image" width="700">
  <div style="
      position: absolute;
      top: 20px;
      left: 20px;
      background: rgba(255,255,255,0.7);
      padding: 10px;">
  </div>
</div>



## Object Detection

In this stage, you combine the camera calibration with part geometry and gripper configuration to detect objects in real time. The detection pipeline involves edge extraction, contour analysis, shape matching, and IoU-based optimization to determine the object's position and orientation in the camera feed.

You can run object detection in two ways:

### 1. Server-Based Detection (`detection_server.py`)
This method is used when you need to integrate with a robot or external client. The script continuously processes the live camera feed and exposes a socket server on port `40411`. When a client (e.g., a robot controller) sends the text command `"get_vision_data"`, the server:
1. Grabs the most recent camera frame.
2. Runs the detection pipeline to determine the object's (x, y) coordinates and orientation.
3. Returns a formatted string such as `(distance_in_mm, x_coordinate, y_coordinate, r_angle)`.


#### Usage Instructions:
1. Ensure that both `calibration_data.pkl` and `gripping_data.pkl` (from Steps 1 and 2) are located in the `data/` directory.
2. From the `src/` folder, run:
```
python detection_server.py
```
3. The server launches two threads:
   - **Camera Thread**: Displays a live OpenCV window with any detection overlays.
   - **Socket Server Thread**: Listens for `"get_vision_data"` requests.
4. Integrate with your robot controller by sending TCP commands. This setup is designed to work with the URCap provided in this project. Ensure that the IP address used to run detection server matches the robot-side settings and that both devices are on the same network.

<div style="position: relative; display: inline-block;">
  <img src="/media/detection_result.png" alt="detection_result" width="700">
  <div style="
      position: absolute;
      top: 20px;
      left: 20px;
      background: rgba(255,255,255,0.7);
      padding: 10px;">
  </div>
</div>


### 2. Standalone Detection (`detection_standalone.py`)
This script lets you test the object detection process without needing an external client. You can configure it to use live camera feed or use previously taken test images by modifying the `USE_STATIC_IMAGE` flag within the script.




