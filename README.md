# LEGO Detection and Sorting Pipeline

This repository provides a comprehensive adaptive detection system pipeline containing camera calibration, gripping point optimization, and real-time detection. Follow the steps outlined below to set up and use the system effectively.


## Table of Contents
- [Installation](#installation)
- [Step 1: Capture Calibration Image](#step-1-capture-images)
- [Step 2: Calibrate the Camera](#step-2-calibrate-the-camera)
- [Step 3: Capture ROI (Region of Interest) Image](#step-3-capture-roi-region-of-interest-image)
- [Step 4: Create LEGO Dataset](#step-4-create-lego-dataset)
- [Step 5: Generate Label Names](#step-5-generate-label-names)
- [Step 6: Label the LEGO Pieces](#step-6-label-the-lego-pieces)
- [Step 7: Create Templates](#step-7-create-templates)
- [Step 8: Generate Synthetic Images](#step-8-generate-synthetic-images)
- [Step 9: Train the YOLO Model](#step-9-train-the-yolo-model)

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
   - Press **x** to capture a test image. Take a picture of rectangular test shape of which location you know to test the accuracy of calibration
     
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


## Step 3: capture roi (region of interest) image

run `camera_gui.py` again:

- **calibration data:** ensure `calibration_input_location` matches the calibration data `.npz` file created in the previous step.
- **calibration prompt:** when prompted, choose "yes" to use the calibration file.
- **adjust roi:**
  - adjust the exposure if necessary to get a clear view.
  - select the appropriate area under the backlight using the gui. example settings: `exposure: 120000; roi_size: 1550`.
  - click the "capture roi" button to save an image of the empty background along with roi and exposure information.
<img src="/media/roi_image.png" alt="Calibration Image" width="500">

## Step 4: create lego dataset

**Lego part id management**:

`TAULegoPartIDs.xlsx` file contains the ID information for all lego parts. Ensure that each part ID is unique and consistent with this file.  
When adding new parts, assign a new ID following the existing naming scheme and update the `TAULegoPartIDs.xlsx` file.

run `lego_dataset_creator.py` to open the graphical user interface.

- click **set new ID (n)** to assign a new ID to the lego part.
- use the **select color** dropdown to choose the color of the lego part.

**capture standard orientations**:

- position the lego part and click **capture image (c)** to capture images from different angles.
- the image will be saved as `<part_id>_<color>_<orientation>.jpg` in the `data/orig_images` directory.

**capture unusual orientations**:

- for challenging angles, click **capture weird orientation (u)** to take images that might confuse the model with other parts.
  
  <img src="/media/data_generator_image.png" alt="Calibration Image" width="500">

## Step 5: generate label names

run `generate_label_names.py`:

- **functionality:** this script gathers names of all the original images and writes them to a text file, `image_filenames.txt`.
- **purpose:** the generated text file can be used to add label names in label studio without needing to manually type them out.

## Step 6: Label the LEGO Pieces

### Install Label Studio:

- **Installation Guide:** Follow the Label Studio [installation guide](https://labelstud.io/guide/install.html) to set up Label Studio on your machine.

### Set Up Label Studio Project:

- **Project Creation:** Create a new project in Label Studio to manage your LEGO image labeling task.
- **Image Loading:** Configure the project to load images from the directory containing your LEGO images (`data/orig_images`).

### Define Labels:

- **Label Import:** Use the text file generated by `generate_label_names.py` to import label names directly into Label Studio.

### Label the Images:

- **Draw Bounding Boxes:** Open each image in Label Studio and draw a bounding box around each LEGO piece.
- **Assign Labels:** Assign the correct label to each bounding box.


## Step 7: create templates

export the labeled images:

- **export:** export the labeled images in YOLO format from label-studio.
- **unzip:** place the zip file in the `data/annotated_data` folder and unzip it.

run `create_templates.py`:

- **template creation:** this script processes the annotated images and their corresponding labels to create template images based on the bounding boxes.
- **output:** the templates are saved in the `data/templates` directory.

## Step 8: generate synthetic images

run `create_images.py`:

- **background and templates:** the script reads the background image and templates.
- **template placement:** it places multiple templates on the background image to create synthetic images.
- **non-overlapping:** ensures that the templates do not overlap significantly and stay within the margins.
- **augmentation:** applies random rotations and intensity changes to templates for augmentation.
- **image generation:** generates a specified number of training and validation images.
- **output:** saves the synthetic images and their corresponding labels in the appropriate directories.

## Step 9: train the yolo model

### prepare for training:

- **ensure data.yaml is configured correctly:** verify that `data.yaml` contains the correct paths to your training and validation datasets.
  

### run training script:
open `Yolo_trainer.ipynb`

use the following script to train your yolo model:

```python
from ultralytics import YOLO
import os

# load the pretrained yolov8 model
model = YOLO('yolov8n.pt')

# define the dataset path
current_path = os.getcwd()
dataset_path = os.path.join(current_path, 'src', 'data', 'data.yaml')

# train the model
results = model.train(data=dataset_path, epochs=50, imgsz=640, batch=32)
```
more info on the model training settings and hyperparameters on [ultralytics website](https://docs.ultralytics.com/modes/train/#train-settings)
