import numpy as np
import cv2

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