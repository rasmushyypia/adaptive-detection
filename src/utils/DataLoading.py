import os
import pickle
import csv
import numpy as np
import logging

def load_calibration_data(calibration_file='data/calibration_data.pkl'):
    """
    Loads camera calibration data from a pickle file.

    Parameters:
        calibration_file (str): Path to the calibration pickle file.

    Returns:
        dict: Dictionary containing calibration parameters.
        tuple: (offset_x, offset_y)
    """
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"Calibration file '{calibration_file}' not found.")

    with open(calibration_file, 'rb') as f:
        calibration_data = pickle.load(f)

    required_keys = ['camera_matrix', 'dist_coeffs', 'new_camera_mtx', 'homography']
    for key in required_keys:
        if key not in calibration_data:
            raise KeyError(f"'{key}' not found in calibration data.")

    offset_x, offset_y = calibration_data.get('offset_data', (0, 0))

    return calibration_data, offset_x, offset_y


def load_gripping_data(gripping_file='data/gripping_data.csv'):
    """
    Loads polygon coordinates and gripping points from a CSV file.

    Parameters:
        gripping_file (str): Path to the gripping data CSV file.

    Returns:
        tuple: (polygon_coords_world, gripping_points_world)
    """
    polygon_coords = []
    gripping_points = []

    if not os.path.exists(gripping_file):
        raise FileNotFoundError(f"Gripping data file '{gripping_file}' not found.")

    with open(gripping_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        section = None
        for row in reader:
            if not row:
                continue  # Skip empty lines
            first_cell = row[0].strip()
            if first_cell == 'Polygon Coordinates':
                section = 'polygon'
                continue
            elif first_cell == 'Gripping Points':
                section = 'gripping'
                continue
            elif section == 'gripping' and 'G' in first_cell:
                try:
                    next_row = next(reader)
                    x, y = map(float, next_row)
                    gripping_points.append([x, y])
                except StopIteration:
                    logging.error(f"Unexpected end of file when reading {first_cell} coordinates.")
            elif section == 'polygon':
                try:
                    x, y = map(float, row)
                    polygon_coords.append([x, y])
                except ValueError:
                    logging.warning(f"Invalid polygon coordinate row: {row}")

    polygon_coords = np.array(polygon_coords)
    gripping_points = np.array(gripping_points)

    if polygon_coords.size == 0:
        raise ValueError("No polygon coordinates found in gripping data.")
    if gripping_points.shape[0] == 0:
        raise ValueError("No gripping points found in gripping data.")
    if polygon_coords.shape[0] < 3:
        raise ValueError("Polygon must have at least 3 points.")

    # Flip y-coordinates to match OpenCV's coordinate system
    polygon_coords[:, 1] = -polygon_coords[:, 1]
    gripping_points[:, 1] = -gripping_points[:, 1]

    logging.info("Gripping data loaded successfully.")
    return polygon_coords, gripping_points