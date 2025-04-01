import os
import cv2
import pickle
import numpy as np
import logging

def load_calibration_data(calibration_file='data/calibration_data.pkl'):
    """
    Loads camera calibration data from a pickle file.

    Returns:
        (dict, (offset_x, offset_y)):
        - Dictionary containing at least:
            camera_matrix, dist_coeffs, new_camera_mtx, homography
          plus optional flip flags or offset data
        - Tuple of (offset_x, offset_y)
    """
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"Calibration file '{calibration_file}' not found.")

    with open(calibration_file, 'rb') as f:
        calibration_data = pickle.load(f)

    # We require these 4 keys to exist
    required_keys = ['camera_matrix', 'dist_coeffs', 'new_camera_mtx', 'homography']
    for key in required_keys:
        if key not in calibration_data:
            raise KeyError(f"'{key}' not found in calibration data.")

    offset_x, offset_y = calibration_data.get('offset_data', (0, 0))

    # Retrieve flip flag if present
    flip_mapping_origin = calibration_data.get('flip_mapping_origin', False)
    logging.info(f"Flip Mapping Origin: {flip_mapping_origin}")

    return calibration_data, offset_x, offset_y


def load_gripping_data(gripping_file='data/gripping_data.pkl'):
    """
    Loads gripping data from a pickle file.

    Parameters:
        gripping_file (str): Path to the gripping data pickle file.

    Returns:
        tuple: (final_poly, gripping_points, gripper_distance, gripper_line_angle)
    """
    if not os.path.exists(gripping_file):
        raise FileNotFoundError(f"Gripping data file '{gripping_file}' not found.")

    with open(gripping_file, 'rb') as f:
        data = pickle.load(f)

    # Validate expected keys
    required_keys = ["final_poly", "gripping_points", "gripper_distance", "gripper_line_angle"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Key '{key}' missing in the gripping data file.")

    final_poly = data["final_poly"]
    gripping_points = np.array(data["gripping_points"])
    gripper_distance = data["gripper_distance"]
    gripper_line_angle = data["gripper_line_angle"]

    logging.info("Gripping data loaded successfully.")
    return final_poly, gripping_points, gripper_distance, gripper_line_angle



def map_world_to_image(points, homography_inv, offset_x=0, offset_y=0):
    """
    Maps world coordinates to image coordinates using inverse homography.

    Parameters:
        points (list or np.ndarray): List or array of (x, y) points in world coordinates.
        homography_inv (np.ndarray): Inverse homography matrix.
        offset_x (float): X-axis offset.
        offset_y (float): Y-axis offset.

    Returns:
        list of tuples: Transformed points in image coordinates.
    """
    if len(points) == 0:
        return []
    points = np.array(points, dtype=np.float32).reshape(-1, 2)
    # Apply offsets
    points[:, 0] += offset_x
    points[:, 1] += offset_y
    points = points.reshape(-1, 1, 2)
    image_points = cv2.perspectiveTransform(points, homography_inv)
    image_points = image_points.reshape(-1, 2)
    return [tuple(pt) for pt in image_points]


def map_image_to_world(points, homography, offset_x=0, offset_y=0):
    """
    Maps image coordinates to world coordinates using homography.

    Parameters:
        points (list of tuples): List of (x, y) points in image coordinates.
        homography (np.ndarray): Homography matrix.
        offset_x (float): X-axis offset.
        offset_y (float): Y-axis offset.

    Returns:
        list of tuples: Transformed points in world coordinates.
    """
    if len(points) == 0:
        return []
    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    world_points = cv2.perspectiveTransform(points, homography)
    world_points = world_points.reshape(-1, 2)
    # Apply offsets
    world_points[:, 0] -= offset_x
    world_points[:, 1] -= offset_y
    return [tuple(pt) for pt in world_points]