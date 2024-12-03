import os
import logging
import cv2
import numpy as np
from scipy.optimize import differential_evolution
from utils import Mapping as mp
from utils import DataLoading as ld
from functools import partial
import time

# FILE PATHS
CALIBRATION_FILE_PATH = 'data/calibration_data.pkl'
GRIPPER_DATA_PATH = 'data/gripping_data.csv'
TEST_IMAGE_PATH = 'data/image_original3.jpg'

# Configuration constants
THRESHOLD = 0.1                
AREA_TOLERANCE = 0.1
LOGGING_LEVEL = logging.DEBUG   # Change to logging.INFO for less verbose output

USE_STATIC_IMAGE = False
CAMERA_INDEX = 1
DESIRED_WIDTH = 1920
DESIRED_HEIGHT = 1080
DESIRED_FPS = 30
BACKEND = cv2.CAP_DSHOW  # For Windows systems


def calculate_iou(contour1, contour2, frame_shape):
    """
    Calculates the Intersection over Union (IoU) between two contours.

    Parameters:
        contour1 (np.ndarray): First contour.
        contour2 (np.ndarray): Second contour.
        frame_shape (tuple): Shape of the image frame (height, width).

    Returns:
        float: IoU value between 0 and 1.
    """
    mask1 = np.zeros(frame_shape[:2], dtype=np.uint8)
    mask2 = np.zeros(frame_shape[:2], dtype=np.uint8)

    cv2.drawContours(mask1, [contour1], -1, 255, -1)  # Fill the contour
    cv2.drawContours(mask2, [contour2], -1, 255, -1)  # Fill the contour

    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)

    intersection_area = np.sum(intersection == 255)
    union_area = np.sum(union == 255)

    if union_area == 0:
        return 0.0  # Avoid division by zero

    iou = intersection_area / union_area
    return iou


def rotate_points(points, angle_deg):
    """
    Rotates points by a given angle in degrees.

    Parameters:
        points (np.ndarray): Array of points of shape (n, 2).
        angle_deg (float): Rotation angle in degrees.

    Returns:
        np.ndarray: Rotated points.
    """
    if points.size == 0:
        return points
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad),  np.cos(angle_rad)]])
    rotated_points = np.dot(points, rotation_matrix.T)
    return rotated_points


def detect_matching_shapes(frame, polygon_contour, method=cv2.CONTOURS_MATCH_I1, threshold=0.03, area_tolerance=0.2):
    """
    Detects contours in the frame that match the reference polygon shape.
    
    Parameters:
        frame (np.ndarray): The input image frame.
        polygon_contour (np.ndarray): Reference polygon contour in image coordinates.
        method (int): Contour comparison method.
        threshold (float): Similarity threshold.
        area_tolerance (float): Tolerance for area difference.

    Returns:
        list of tuples: List containing matched contours and their similarity scores.
    """
    matched_contours = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ref_area = cv2.contourArea(polygon_contour)
    logging.debug(f"Total contours found: {len(contours)}")

    for cnt in contours:
        similarity = cv2.matchShapes(polygon_contour, cnt, method, 0.0)
        if similarity < threshold:
            area = cv2.contourArea(cnt)
            if ref_area == 0:
                continue  # Avoid division by zero
            area_diff = abs(area - ref_area) / ref_area
            if area_diff < area_tolerance:
                matched_contours.append((cnt, similarity))
                logging.debug(f"Contour accepted with area difference: {area_diff:.2%}")
            else:
                logging.debug(f"Contour rejected due to area difference: {area_diff:.2%}")

    logging.debug(f"Detected {len(matched_contours)} matching contours with threshold {threshold} and area tolerance {area_tolerance}.")

    if matched_contours:
        logging.info(f"Found {len(matched_contours)} matching contour(s).")
    else:
        logging.info("No matching contours found.")

    return matched_contours


def negative_iou(params, polygon_coords_world, x_world, y_world, homography_inv, offset_x, offset_y, frame_shape, cnt):
    angle_deg, x_shift, y_shift = params
    rotated_polygon = rotate_points(polygon_coords_world, angle_deg)
    # Apply translation
    polygon_coords_absolute_world = rotated_polygon + np.array([x_world + x_shift, y_world + y_shift])
    # Map to image coordinates
    polygon_coords_image = mp.map_world_to_image(
        polygon_coords_absolute_world, homography_inv, offset_x, offset_y
    )
    polygon_contour_image = np.array(polygon_coords_image, dtype=int).reshape((-1, 1, 2))
    iou = calculate_iou(cnt, polygon_contour_image, frame_shape)
    return -iou  # Negative because we are minimizing


def optimize_angle(
    cnt, x_world, y_world, polygon_coords_world, homography_inv, offset_x, offset_y, frame_shape
):
    """
    Optimizes the rotation angle using Differential Evolution to maximize IoU.
    """
    # Create a partial function with fixed parameters
    objective = partial(
        negative_iou,
        polygon_coords_world=polygon_coords_world,
        x_world=x_world,
        y_world=y_world,
        homography_inv=homography_inv,
        offset_x=offset_x,
        offset_y=offset_y,
        frame_shape=frame_shape,
        cnt=cnt
    )

    # Define optimized bounds based on prior knowledge
    angle_bounds = (0, 360)
    shift_bounds = (-3, 3)  # Adjust as necessary
    bounds = [angle_bounds, shift_bounds, shift_bounds]

    # Differential Evolution parameters
    try:
        result = differential_evolution(
            objective,
            bounds,
            strategy='best1bin',
            maxiter=1000,          # Reduced number of generations
            popsize=9,          # Reduced population size
            tol=0.01,            # Increased tolerance for faster convergence
            mutation=(0.5, 1),
            recombination=0.7,
            disp=False,
            updating='immediate',
            workers=1,          
            seed=None              # For reproducibility
        )
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        return None, None, None, 0.0

    optimized_angle, x_shift, y_shift = result.x
    max_iou = -result.fun

    # Log only essential information to minimize overhead
    logging.debug(
        f"DE Optimization result - Angle: {optimized_angle:.2f}°, "
        f"x_shift: {x_shift:.2f}, y_shift: {y_shift:.2f}, IoU: {max_iou:.4f}"
    )

    return optimized_angle, x_shift, y_shift, max_iou


def visualize_detections(
    frame, matched_contours, world_coords, polygon_coords_world,
    gripping_points_world, homography_inv, offset_x, offset_y
):
    """
    Visualizes detections by drawing contours, centroids, orientation lines, and gripping points.

    Parameters:
        frame (np.ndarray): The image frame to draw on.
        matched_contours (list of tuples): List of matched contours and their similarity scores.
        world_coords (list of tuples): List of object centers in world coordinates.
        polygon_coords_world (np.ndarray): Polygon coordinates in world coordinates.
        gripping_points_world (np.ndarray): Gripping points in world coordinates.
        homography_inv (np.ndarray): Inverse homography matrix.
        offset_x (float): X-axis offset.
        offset_y (float): Y-axis offset.
    """

    def draw_orientation_line(img, cnt, angle_deg):
        """
        Draws a line representing the orientation of a contour on the image.

        Parameters:
            img (np.ndarray): The image on which to draw.
            cnt (np.ndarray): Contour points.
            angle_deg (float): Orientation angle in degrees.
        """
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            return
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        length = 100  # Length of the orientation line
        angle_rad = np.radians(angle_deg)
        x2 = int(cx + length * np.cos(angle_rad))
        y2 = int(cy + length * np.sin(angle_rad))
        cv2.line(img, (cx, cy), (x2, y2), (255, 0, 0), 2)  # Blue line

    for i, ((cnt, similarity), (x_world, y_world)) in enumerate(zip(matched_contours, world_coords), 1):
        # Draw detected contour in green
        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)  # Green contour
        logging.debug(f"Detected contour {i} drawn with similarity {similarity:.4f}.")

        # Compute centroid in image coordinates
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cX_img = int(M['m10'] / M['m00'])
            cY_img = int(M['m01'] / M['m00'])
        else:
            x_min, y_min, w, h = cv2.boundingRect(cnt)
            cX_img, cY_img = x_min + w // 2, y_min + h // 2

        # Draw centroid as red dot
        cv2.circle(frame, (cX_img, cY_img), 5, (0, 0, 255), -1)  # Red dot

        # Annotate centroid with world coordinates
        cv2.putText(
            frame, f"Center: ({x_world:.1f}, {y_world:.1f}) mm",
            (cX_img + 10, cY_img + 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2)

        # Optimize the angle using IoU
        optimized_angle, x_shift, y_shift, max_iou = optimize_angle(
            cnt, x_world, y_world, polygon_coords_world,
            homography_inv, offset_x, offset_y, frame.shape)

        logging.info(f"Optimized angle for object {i}: {optimized_angle:.2f} degrees with IoU: {max_iou:.4f}")

        # Draw orientation line
        draw_orientation_line(frame, cnt, optimized_angle)

        # Rotate the polygon and gripping points with optimized angle
        rotated_polygon = rotate_points(polygon_coords_world, optimized_angle)
        rotated_gripping_points = rotate_points(gripping_points_world, optimized_angle)

        # Translate to the object center (with shifts)
        polygon_coords_absolute_world = rotated_polygon + np.array([x_world + x_shift, y_world + y_shift])
        gripping_points_absolute_world = rotated_gripping_points + np.array([x_world + x_shift, y_world + y_shift])

        # Map to image coordinates
        polygon_coords_image = mp.map_world_to_image(
            polygon_coords_absolute_world, homography_inv, offset_x, offset_y)
        
        polygon_contour_image = np.array(polygon_coords_image, dtype=int).reshape((-1, 1, 2))

        # Draw the rotated polygon contour in yellow
        cv2.drawContours(frame, [polygon_contour_image], -1, (0, 255, 255), 2)  # Yellow contour

        # Annotate IoU on the image near the centroid
        cv2.putText(
            frame, f"IoU: {max_iou:.2f}",
            (cX_img + 10, cY_img + 50), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2)

        # Map and draw gripping points in blue
        gripping_points_image = mp.map_world_to_image(
            gripping_points_absolute_world, homography_inv, offset_x, offset_y)

        for idx, pt in enumerate(gripping_points_image, 1):
            x_pt, y_pt = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x_pt, y_pt), 5, (255, 0, 0), -1)  # Blue dots
            cv2.putText(
                frame, f"G{idx}",
                (x_pt + 5, y_pt - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 2)

        # Annotate the optimized orientation angle in red text
        cv2.putText(
            frame, f"Angle: {optimized_angle:.1f} deg",
            (cX_img + 10, cY_img + 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2)

    cv2.imshow('Live Object Detection', frame)


def run_static_image_test(
    test_image_path, calibration_data, offset_x, offset_y,
    polygon_coords_world, gripping_points_world):
    
    try:
        camera_matrix = calibration_data['camera_matrix']
        dist_coeffs = calibration_data['dist_coeffs']
        new_camera_mtx = calibration_data['new_camera_mtx']
        homography = calibration_data['homography']
        homography_inv = np.linalg.inv(homography)
    except KeyError as e:
        logging.error(f"Missing calibration key: {e}")
        return

    if not os.path.exists(test_image_path):
        logging.error(f"Test image '{test_image_path}' not found.")
        return

    frame = cv2.imread(test_image_path)
    if frame is None:
        logging.error(f"Failed to load test image '{test_image_path}'.")
        return

    # Undistort the image
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_mtx)

    polygon_coords_image = mp.map_world_to_image(polygon_coords_world, homography_inv, offset_x, offset_y)
    polygon_contour_image = np.array(polygon_coords_image, dtype=np.int32).reshape((-1, 1, 2))

    matched_contours = detect_matching_shapes(
        undistorted, polygon_contour_image, threshold=THRESHOLD, area_tolerance=AREA_TOLERANCE
    )

    if matched_contours:
        centroids = []
        for cnt, _ in matched_contours:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                centroids.append((cx, cy))

        world_coords = mp.map_image_to_world(centroids, homography, offset_x=offset_x, offset_y=offset_y)
        
        # Start the timer before optimization
        start_time = time.perf_counter()

        visualize_detections(
            undistorted, matched_contours, world_coords,
            polygon_coords_world, gripping_points_world,
            homography_inv, offset_x, offset_y)

        # End the timer after optimization
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Log the elapsed time
        logging.info(f"Optimization and visualization completed in {elapsed_time:.4f} seconds.")
    else:
        cv2.imshow('Live Object Detection', undistorted)
        logging.info("No matching objects detected.")

    print("Test image processed. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_live_detection(
    calibration_data, offset_x, offset_y,
    polygon_coords_world, gripping_points_world):
    try:
        camera_matrix = calibration_data['camera_matrix']
        dist_coeffs = calibration_data['dist_coeffs']
        new_camera_mtx = calibration_data['new_camera_mtx']
        homography = calibration_data['homography']
        homography_inv = np.linalg.inv(homography)
    except KeyError as e:
        logging.error(f"Missing calibration key: {e}")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX, BACKEND)
    if not cap.isOpened():
        logging.error(f"Cannot open webcam with index {CAMERA_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, DESIRED_FPS)

    # Verify and log actual settings
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    logging.info(f"Requested Resolution: {DESIRED_WIDTH}x{DESIRED_HEIGHT} at {DESIRED_FPS} FPS")
    logging.info(f"Actual Resolution: {int(actual_width)}x{int(actual_height)} at {int(actual_fps)} FPS")

    logging.info("Starting live object detection. Press 'q' to quit or 'spacebar' to capture a new image.")

    while True:
        # Display a live feed with instructions
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to grab frame.")
            break

        # Resize the frame to match the display size
        display_frame = cv2.resize(frame, (int(actual_width), int(actual_height)))

        # Add instructions on the frame
        cv2.putText(display_frame, "Press 'spacebar' to capture image, 'q' to quit.",
                    (80, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Live Object Detection', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logging.info("Exiting live object detection.")
            break
        elif key == ord(' '):  # Spacebar pressed
            # Capture and process the frame
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_mtx)
            polygon_coords_image = mp.map_world_to_image(polygon_coords_world, homography_inv, offset_x, offset_y)
            polygon_contour_image = np.array(polygon_coords_image, dtype=np.int32).reshape((-1, 1, 2))

            matched_contours = detect_matching_shapes(
                undistorted, polygon_contour_image, threshold=THRESHOLD, area_tolerance=AREA_TOLERANCE
            )

            if matched_contours:
                centroids = []
                for cnt, _ in matched_contours:
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        centroids.append((cx, cy))

                world_coords = mp.map_image_to_world(centroids, homography, offset_x=offset_x, offset_y=offset_y)
                
                # Start the timer before optimization
                start_time = time.perf_counter()

                visualize_detections(
                    undistorted, matched_contours, world_coords,
                    polygon_coords_world, gripping_points_world,
                    homography_inv, offset_x, offset_y)

                # End the timer after optimization
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

                # Log the elapsed time
                logging.info(f"Optimization and visualization completed in {elapsed_time:.4f} seconds.")
            else:
                cv2.imshow('Live Object Detection', undistorted)
                logging.info("No matching objects detected.")

            # Wait until the user presses 'spacebar' or 'q' again
            logging.info("Processing complete. Press 'spacebar' to capture again or 'q' to quit.")
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logging.info("Exiting live object detection.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord(' '):
                    break  # Exit the inner loop to capture a new image

    cap.release()
    cv2.destroyAllWindows()


def main():
    # Configure logging
    logging.basicConfig(
        level=LOGGING_LEVEL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("data/live_object_detection.log"),
            logging.StreamHandler()])

    # Load calibration data
    try:
        calibration_data, offset_x, offset_y = ld.load_calibration_data(CALIBRATION_FILE_PATH)
    except (FileNotFoundError, KeyError, ValueError) as e:
        logging.error(e)
        return

    # Load gripping data
    try:
        polygon_coords_world, gripping_points_world = ld.load_gripping_data(GRIPPER_DATA_PATH)
    except (FileNotFoundError, ValueError) as e:
        logging.error(e)
        return

    # Run the appropriate detection mode
    if USE_STATIC_IMAGE:
        run_static_image_test(TEST_IMAGE_PATH, calibration_data, offset_x, offset_y, polygon_coords_world, gripping_points_world)
    else:
        run_live_detection(calibration_data, offset_x, offset_y, polygon_coords_world, gripping_points_world)

if __name__ == "__main__":
    main()
