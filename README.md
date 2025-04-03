# Adaptive Gripping and Detection System
This repository provides a comprehensive adaptive detection system pipeline containing camera calibration, gripping point optimization, and real-time detection. Follow the steps outlined below to set up and use the system effectively.


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
