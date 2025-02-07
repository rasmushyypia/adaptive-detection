# camera_calibration.py

import cv2
import numpy as np
import glob
import os
import pickle


# Step 1: Camera Calibration

def calibrate_camera(calib_images_dir, grid_size, square_size, visualize=False):
    """
    Calibrates the camera using chessboard images.

    Parameters:
        calib_images_dir (str): Directory containing calibration images.
        grid_size (tuple): Number of internal corners per chessboard row and column (columns, rows).
        square_size (float): Size of a chessboard square in real-world units (e.g., millimeters).
        visualize (bool): Whether to display calibration images with detected corners.

    Returns:
        tuple: camera_matrix, dist_coeffs, new_camera_mtx
    """
    # Prepare object points without offset
    objp = np.zeros((grid_size[1]*grid_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale to real-world units

    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane
    img_size = None

    # Supported image extensions
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(calib_images_dir, f'calib_*.{ext}')))

    if not images:
        print("No calibration images found. Check the directory path and image formats.")
        return None, None, None

    # Define termination criteria for corner refinement
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"Image {fname} could not be loaded.")
            continue  # Skip if image not loaded
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCornersSB(gray, grid_size, flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
        
        if ret:
            # Refine corner locations
            corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            if img_size is None:
                img_size = gray.shape[::-1]
            # Optional: draw and display corners for verification
            if visualize:
                cv2.drawChessboardCorners(img, grid_size, corners_refined, ret)
                cv2.imshow('Calibration', img)
                cv2.waitKey(100)
        else:
            print(f"Chessboard corners not found in image {fname}.")
            # Save the problematic image for review
            problematic_dir = os.path.join(calib_images_dir, 'problematic')
            os.makedirs(problematic_dir, exist_ok=True)
            problematic_path = os.path.join(problematic_dir, os.path.basename(fname))
            cv2.imwrite(problematic_path, img)
            print(f"Saved problematic image to {problematic_path}")
    if visualize:
        cv2.destroyAllWindows()

    if not objpoints or not imgpoints:
        print("No corners were found in any image. Calibration failed.")
        return None, None, None

    # Camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)

    # Calculate total reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)
    print(f"Mean Reprojection Error: {mean_error}")

    # Compute new camera matrix for undistortion
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, img_size, 0, img_size)

    return camera_matrix, dist_coeffs, new_camera_mtx


# Step 2: Undistort Images

def undistort_image(img, camera_matrix, dist_coeffs, new_camera_mtx):
    """
    Undistorts the given image using the camera matrix and distortion coefficients.

    Parameters:
        img (numpy.ndarray): Input distorted image.
        camera_matrix (numpy.ndarray): Original camera matrix.
        dist_coeffs (numpy.ndarray): Distortion coefficients.
        new_camera_mtx (numpy.ndarray): New camera matrix from calibration.

    Returns:
        numpy.ndarray: Undistorted image.
    """
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_mtx)
    return undistorted_img


# Step 3: Compute Homography from Test Image
def compute_homography_from_test_image(objp, corners_refined, camera_matrix, dist_coeffs, new_camera_mtx):
    """
    Computes the homography matrix using the test image.

    Parameters:
        objp (numpy.ndarray): 3D object points of the chessboard (without offset).
        corners_refined (numpy.ndarray): Refined 2D image points from the test image.
        camera_matrix (numpy.ndarray): Original camera matrix.
        dist_coeffs (numpy.ndarray): Distortion coefficients.
        new_camera_mtx (numpy.ndarray): New camera matrix for undistortion.

    Returns:
        numpy.ndarray: Homography matrix.
    """
    # Undistort image points to pixel coordinates using new camera matrix
    imgp_undistorted = cv2.undistortPoints(corners_refined, camera_matrix, dist_coeffs, P=new_camera_mtx)
    imgp_undistorted = imgp_undistorted.reshape(-1, 2)

    # Use only the 2D world coordinates (without offset)
    objp_2d = objp[:, :2]

    # Compute homography using Direct Linear Transformation (DLT)
    H, status = cv2.findHomography(imgp_undistorted, objp_2d, method=cv2.RANSAC)
    if H is not None:
        print("Homography from test image successfully computed.")
    else:
        print("Homography computation from test image failed.")
    return H


# Step 4: Object Detection

import cv2
import numpy as np

def detect_objects(img, visualize=False, low_threshold=50, high_threshold=150, min_area=500, aspect_ratio_range=(0.8, 1.2)):
    """
    Detects rectangular objects in the image and returns their centroids with sub-pixel accuracy.

    Parameters:
        img (numpy.ndarray): Input undistorted image.
        visualize (bool): Whether to visualize the detection process.
        low_threshold (int): Lower threshold for Canny edge detection.
        high_threshold (int): Upper threshold for Canny edge detection.
        min_area (int): Minimum contour area to be considered a valid object.
        aspect_ratio_range (tuple): Range of acceptable aspect ratios.

    Returns:
        list: List of (x, y) centroids of detected rectangular objects with sub-pixel precision.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=3)

    # Apply Dilate and Erode to close gaps between edges (Morphological Operations)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Optional: Visualize the edge-detected image
    if visualize:
        cv2.imshow('Edges', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Find contours based on the edges detected
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Total contours found: {len(contours)}")

    object_centroids = []

    for cnt in contours:
        # Calculate contour area to filter out small noises
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # Approximate the contour to a polygon
        peri = cv2.arcLength(cnt, True)
        epsilon = 0.01 * peri  # Smaller epsilon for precise approximation
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check if the approximated contour has 4 vertices (rectangle)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Further check: Ensure the shape is roughly rectangular by checking aspect ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]:
                # Calculate centroid using image moments (floating-point precision)
                M = cv2.moments(approx)
                if M['m00'] == 0:
                    continue  # Avoid division by zero
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                object_centroids.append((cx, cy))

                if visualize:
                    # Draw the approximated rectangle on the image
                    cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)

                    # Draw the initial centroid (floating-point coordinates)
                    # For visualization, convert to integer
                    cv2.circle(img, (int(cx), int(cy)), 5, (0, 0, 255), -1)

                    # Define ROI around the centroid for sub-pixel refinement
                    roi_size = 10  # Define the size of the window (in pixels)
                    x_min = max(int(cx) - roi_size, 0)
                    y_min = max(int(cy) - roi_size, 0)
                    x_max = min(int(cx) + roi_size, img.shape[1] - 1)
                    y_max = min(int(cy) + roi_size, img.shape[0] - 1)
                    roi = gray[y_min:y_max, x_min:x_max]

                    # Threshold the ROI to find the centroid more precisely
                    _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # Calculate moments in the thresholded ROI
                    M_roi = cv2.moments(roi_thresh)
                    if M_roi['m00'] != 0:
                        # Calculate refined centroid within ROI
                        cx_roi = M_roi['m10'] / M_roi['m00']
                        cy_roi = M_roi['m01'] / M_roi['m00']

                        # Adjust the centroid coordinates relative to the whole image
                        refined_cx = x_min + cx_roi
                        refined_cy = y_min + cy_roi
                        object_centroids[-1] = (refined_cx, refined_cy)

                        # Draw the refined centroid
                        cv2.circle(img, (int(refined_cx), int(refined_cy)), 3, (255, 0, 0), -1)

                    # Label the rectangle with a unique identifier
                    cv2.putText(img, f"Rect {len(object_centroids)}",
                                (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 0), 2)  # Blue for ID

                    # Label the image coordinates with two decimal places
                    cv2.putText(img, f"Image: ({cx:.2f}, {cy:.2f})",
                                (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 255), 2)  # Yellow for image coordinates

    if visualize:
        # Display the image with detected rectangles and centroids
        cv2.imshow('Detected Rectangular Objects', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Detected {len(object_centroids)} rectangular objects.")
    return object_centroids


# Step 5: Map Image Points to Real-World Coordinates

def map_image_to_world(points, H):
    """
    Maps image points to table world coordinates using homography.

    Parameters:
        points (list): List of (u, v) image coordinates in undistorted pixel space.
        H (numpy.ndarray): Homography matrix.

    Returns:
        list: List of (x, y) table world coordinates.
    """
    if len(points) == 0:
        return []
    image_points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    # Use the homography to map image points to world coordinates
    world_points = cv2.perspectiveTransform(image_points, H)
    # Reshape to (N, 2)
    world_points = world_points.reshape(-1, 2)
    return [tuple(pt) for pt in world_points]


# Step 6: Helper Functions for Coordinate Axes Overlay and Debugging

def draw_axes(img, corners, rvecs, tvecs, camera_matrix, dist_coeffs, axis_length):
    """
    Draws the 3D axes on the image based on the chessboard pose.

    Parameters:
        img (numpy.ndarray): The image on which to draw.
        corners (numpy.ndarray): Detected chessboard corners.
        rvecs (numpy.ndarray): Rotation vectors from solvePnP.
        tvecs (numpy.ndarray): Translation vectors from solvePnP.
        camera_matrix (numpy.ndarray): Camera matrix.
        dist_coeffs (numpy.ndarray): Distortion coefficients.
        axis_length (float): Length of the axes in real-world units.
    """
    # Define 3D axes points in object space
    axis = np.float32([
        [0, 0, 0],                   # Origin
        [axis_length, 0, 0],         # X-axis
        [0, axis_length, 0],         # Y-axis
        [0, 0, -axis_length]         # Z-axis (negative direction)
    ]).reshape(-1, 3)

    # Project 3D points to image plane
    imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, dist_coeffs)

    # Draw axes
    corner = tuple(corners[0].ravel().astype(int))
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0, 0, 255), 3)  # X-axis in red
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0, 255, 0), 3)  # Y-axis in green
    img = cv2.line(img, corner, tuple(imgpts[3].ravel().astype(int)), (255, 0, 0), 3)  # Z-axis in blue


def draw_real_world_grid(img, H_inv, grid_size_mm=100, step=50):
    """
    Draws a real-world grid on the image for verification.

    Parameters:
        img (numpy.ndarray): The image on which to draw.
        H_inv (numpy.ndarray): Inverse homography matrix.
        grid_size_mm (int): Size of the grid in millimeters.
        step (int): Step size between grid lines in millimeters.
    """
    # Define grid lines in real-world coordinates
    for x in range(0, grid_size_mm + step, step):
        start_world = np.array([x, 0], dtype=np.float32)
        end_world = np.array([x, grid_size_mm], dtype=np.float32)
        start_img, end_img = map_world_to_image(
            np.array([start_world, end_world]), H_inv)
        start_img = tuple(map(int, start_img))
        end_img = tuple(map(int, end_img))
        cv2.line(img, start_img, end_img, (0, 255, 0), 1)  # Green grid lines

    for y in range(0, grid_size_mm + step, step):
        start_world = np.array([0, y], dtype=np.float32)
        end_world = np.array([grid_size_mm, y], dtype=np.float32)
        start_img, end_img = map_world_to_image(
            np.array([start_world, end_world]), H_inv)
        start_img = tuple(map(int, start_img))
        end_img = tuple(map(int, end_img))
        cv2.line(img, start_img, end_img, (0, 255, 0), 1)  # Green grid lines

    # Display the grid
    cv2.imshow('Real-World Grid Overlay', img)
    cv2.waitKey(500)  # Display for 500ms


def map_world_to_image(world_points, H_inv):
    """
    Maps world coordinates to image coordinates using the inverse homography matrix.

    Parameters:
        world_points (numpy.ndarray): Array of world points with shape (N, 2).
        H_inv (numpy.ndarray): Inverse homography matrix.

    Returns:
        list: List of (u, v) image coordinates.
    """
    if len(world_points) == 0:
        return []
    world_points_homog = np.hstack([world_points, np.ones((world_points.shape[0], 1))])  # Convert to homogeneous coordinates
    image_points_homog = H_inv @ world_points_homog.T  # Apply inverse homography
    image_points_homog /= image_points_homog[2, :]  # Normalize
    image_points = image_points_homog[:2, :].T  # Extract (u, v)
    return image_points.tolist()


def main():
    # Calibration parameters
    calib_images_dir = 'data/calibration_images'  # Directory containing calibration images
    calib_grid_size = (10, 7)
    calib_square_size = 30.0  # in millimeters

    # Homography (Mapping) parameters
    mapping_grid_size = (14, 7)
    mapping_square_size = 50.0  # in millimeters

    # Paths to images
    coordinate_frame_img_path = 'data/coordinate_frame_image.jpg'
    object_img_path = 'data/test_image.jpg'

    # Toggles
    save_calibration = False
    visualize = True            

    # Define the physical offset from table origin to grid origin
    offset_x = -71  # in millimeters (adjust as needed)
    offset_y = 6 * mapping_square_size + 70.6  # in millimeters (adjust as needed)

    #offset_x = -65  # in millimeters (adjust as needed)
    #offset_y = 6 * mapping_square_size + 67  # in millimeters (adjust as needed)

    # Calibrate the camera
    print("Calibrating camera...")
    calibration_results = calibrate_camera(
        calib_images_dir, calib_grid_size, calib_square_size, visualize=True)

    if calibration_results[0] is None:
        print("Camera calibration failed. Exiting.")
        return

    camera_matrix, dist_coeffs, new_camera_mtx = calibration_results

    print("\nCamera calibrated.")
    print("Original Camera Matrix:")
    print(camera_matrix)
    print("\nNew Camera Matrix:")
    print(new_camera_mtx)
    print("\nDistortion Coefficients:")
    print(dist_coeffs.ravel())

    # Load test image
    if not os.path.exists(coordinate_frame_img_path):
        print(f"\nTest image '{coordinate_frame_img_path}' not found. Exiting.")
        return
    coordinate_frame_img = cv2.imread(coordinate_frame_img_path)
    if coordinate_frame_img is None:
        print("Test image could not be loaded. Exiting.")
        return

    # Load object image
    if not os.path.exists(object_img_path):
        print(f"\nObject image '{object_img_path}' not found. Exiting.")
        return
    object_img = cv2.imread(object_img_path)
    if object_img is None:
        print("Object image could not be loaded. Exiting.")
        return

    # Undistort test image using new camera matrix without cropping
    undistorted_coordinate_frame_img = undistort_image(coordinate_frame_img, camera_matrix, dist_coeffs, new_camera_mtx)
    undistorted_obj_img = undistort_image(object_img, camera_matrix, dist_coeffs, new_camera_mtx)

    # Detect chessboard corners in the test image using mapping grid
    gray = cv2.cvtColor(undistorted_coordinate_frame_img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCornersSB(
        gray, mapping_grid_size, flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)

    if ret:
        # Refine corner locations
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Prepare object points without offset
        objp = np.zeros((mapping_grid_size[1] * mapping_grid_size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:mapping_grid_size[0], 0:mapping_grid_size[1]].T.reshape(-1, 2)
        objp *= mapping_square_size  # Correct scaling

        # Compute rotation and translation vectors
        ret_pnp, rvecs, tvecs = cv2.solvePnP(objp, corners_refined, camera_matrix, dist_coeffs)
        if not ret_pnp:
            print("Pose estimation failed. Exiting.")
            return

        # Compute homography from test image
        print("\nComputing homography from test image...")
        H = compute_homography_from_test_image(objp, corners_refined, camera_matrix, dist_coeffs, new_camera_mtx)
        if H is not None:
            print("\nHomography matrix from test image computed:")
            print(H)
        else:
            print("Homography computation from test image failed. Exiting.")
            return

        # Compute inverse homography for mapping world to image (if needed)
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            print("Homography matrix is singular and cannot be inverted. Exiting.")
            return

        # Draw axes on the undistorted test image
        axis_length = mapping_square_size * 3  # Adjust as needed
        draw_axes(undistorted_coordinate_frame_img, corners_refined, rvecs, tvecs, camera_matrix, dist_coeffs, axis_length)

        # Optionally, visualize the calibration origin on the test image
        if visualize:
            cv2.imshow('Test Image with Axes', undistorted_coordinate_frame_img)
            cv2.waitKey(500)  # Display for 500ms
    else:
        print("Chessboard corners not found in test image. Cannot compute homography or draw axes.")
        return

    # Detect objects in the object image
    print("\nDetecting objects...")
    object_points = detect_objects(
        undistorted_obj_img.copy(),  # Use a copy to avoid drawing over the axes
        visualize=True,              # Enable visualization
        low_threshold=30, 
        high_threshold=100, 
        min_area=300, 
        aspect_ratio_range=(0.5, 1.5)
    )
    print(f"Detected {len(object_points)} rectangular objects.")

    # Map image points to chessboard (homography) coordinates
    chessboard_coordinates = map_image_to_world(object_points, H)

    # Apply offset to get world coordinates
    table_coordinates = [(x - offset_x, y - offset_y) for (x, y) in chessboard_coordinates]

    # Annotate the final image with rectangle numbers and table coordinates
    for idx, ((u, v), (x, y)) in enumerate(zip(object_points, table_coordinates)):
        # Convert floating-point coordinates to integers for drawing
        u_int, v_int = int(u), int(v)

        # Draw a circle at the centroid
        cv2.circle(undistorted_obj_img, (u_int, v_int), 5, (0, 0, 255), -1)  # Red dot for centroid

        # Draw rectangle number (Rect 1, Rect 2, etc.)
        cv2.putText(undistorted_obj_img, f"Rect {idx+1}",
                    (u_int + 10, v_int + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)  # Blue text for rectangle number

        # Draw table coordinates
        cv2.putText(undistorted_obj_img, f"Table: ({x:.1f}, {y:.1f}) mm",
                    (u_int + 10, v_int + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 165, 255), 2)  # Orange text for table coordinates

    # Visualization of Shifted Origin (Optional)
    if visualize:
        # Define the shifted origin in world coordinates
        shifted_origin_world = np.array([[offset_x, offset_y]], dtype=np.float32)

        # Map the shifted origin back to image coordinates using inverse homography
        shifted_origin_image = cv2.perspectiveTransform(np.array([shifted_origin_world]), H_inv)
        shifted_origin_image = tuple(shifted_origin_image.ravel().astype(int))

        # Draw the shifted origin on the object image
        cv2.circle(undistorted_obj_img, shifted_origin_image, 5, (0, 0, 255), -1)  # Red dot for shifted origin

        # Label the shifted origin
        cv2.putText(undistorted_obj_img, "Shifted Origin",
                    (shifted_origin_image[0] + 10, shifted_origin_image[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red text for shifted origin

        # Display the final annotated image with shifted origin
        cv2.imshow('Final Result with Table Coordinates and Shifted Origin', undistorted_obj_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # SAVE the annotated image
    results_dir = 'data/results'
    base_filename = 'calibration_test_result.jpg'
    save_unique_image(undistorted_obj_img, results_dir, base_filename)


    # Save the homography matrix into calibration_data.pkl
    if save_calibration:
        calibration_data = {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'new_camera_mtx': new_camera_mtx,
            'homography': H,
            'offset_data': (offset_x, offset_y)
        }
        with open('data/calibration_data.pkl', 'wb') as f:
            pickle.dump(calibration_data, f)
        print("\nCalibration data  saved to 'data/calibration_data.pkl'.")
        
def save_unique_image(image, directory, base_filename):
    """
    Saves the image to the specified directory with a unique filename.
    If a file with the base_filename exists, appends a number suffix.

    Parameters:
        image (numpy.ndarray): The image to save.
        directory (str): The directory where the image will be saved.
        base_filename (str): The base name for the image file.

    Returns:
        str: The path to the saved image file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    name, ext = os.path.splitext(base_filename)
    save_path = os.path.join(directory, base_filename)
    counter = 1
    while os.path.exists(save_path):
        save_path = os.path.join(directory, f"{name}_{counter}{ext}")
        counter += 1
    cv2.imwrite(save_path, image)
    print(f"Annotated image saved as '{save_path}'.")
    return save_path



if __name__ == "__main__":
    main()
