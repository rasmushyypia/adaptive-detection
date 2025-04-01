#!/usr/bin/env python3
"""
detection_test.py – Object Detection with Static Image or Live Camera Feed

This script performs object detection and template matching using contours and IoU optimization.
It can run on a static test image or use a live camera feed, capturing detections on a spacebar press.
"""

import os
import logging
import cv2
import numpy as np
from scipy.optimize import differential_evolution
from functools import partial
import time

# Shapely for geometric transformations (rotation, translation, scaling)
from shapely.affinity import rotate, translate, scale

# Local utilities for mapping and data loading
import utils.detection_utils as du

# ---------------------------
# Global Configuration & Paths
# ---------------------------
CALIBRATION_FILE_PATH = 'data/calibration_data.pkl'
GRIPPER_DATA_PATH = 'data/gripping_data.pkl'
TEST_IMAGE_PATH = 'data/test_images/test_custom_2.jpg'

# Configuration constants
THRESHOLD = 0.08          # Template matching similarity threshold
AREA_TOLERANCE = 0.15     # Tolerance for area difference (15%)
LOGGING_LEVEL = logging.DEBUG

DEBUG_EDGES = True
SHOW_HIERARCHY_DEBUG = True  # Show hierarchy debug view

# Camera configuration
USE_STATIC_IMAGE = True # Set True to use the static image; False for live feed
CAMERA_INDEX = 0
DESIRED_WIDTH = 1920
DESIRED_HEIGHT = 1080
DESIRED_FPS = 30
BACKEND = cv2.CAP_DSHOW

# ---------------------------
# Utility Functions
# ---------------------------

def get_edges(image, canny_thresh1=5, canny_thresh2=40):
    """
    Extract strong edges using Canny edge detection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.rectangle(mask, (26, 0), (1865, 1042), 255, thickness=-1)
    cropped_gray = cv2.bitwise_and(gray, gray, mask=mask)
    blurred = cv2.medianBlur(cropped_gray, 7)
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2, L2gradient=True)
    kernel = np.ones((3, 3), np.uint8)
    stronger_edges = cv2.dilate(edges, kernel, iterations=1)
    return stronger_edges

def filter_composite(comp, ratio_threshold=0.9, min_area=1000):
    """
    Filter a composite contour to retain valid children based on area.
    """
    parent = comp[0]
    parent_area = cv2.contourArea(parent)
    valid_children = []
    for child in comp[1:]:
        child_area = cv2.contourArea(child)
        if child_area >= min_area and (child_area / parent_area < ratio_threshold):
            valid_children.append(child)
    return [parent] + valid_children

def rotate_points(points, angle_deg):
    """
    Rotate a set of points by a specified angle (in degrees).
    """
    if points.size == 0:
        return points
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad),  np.cos(angle_rad)]])
    return np.dot(points, rotation_matrix.T)

# ---------------------------
# Contour Extraction & Hierarchy Processing
# ---------------------------

def get_valid_children(parent_idx, hierarchy, contour_areas, ratio_threshold=0.9):
    """
    Recursively determine valid child contours for a given parent.
    """
    valid_children = []
    parent_area = contour_areas[parent_idx]
    child_index = hierarchy[parent_idx][2]  # first child index
    while child_index != -1:
        child_area = contour_areas[child_index]
        if child_area >= 1000:
            if child_area / parent_area >= ratio_threshold:
                grandchild_idx = hierarchy[child_index][2]
                while grandchild_idx != -1:
                    if contour_areas[grandchild_idx] >= 1000 and (contour_areas[grandchild_idx] / parent_area < ratio_threshold):
                        valid_children.append(grandchild_idx)
                    grandchild_idx = hierarchy[grandchild_idx][0]
            else:
                valid_children.append(child_index)
        child_index = hierarchy[child_index][0]
    return valid_children

def get_composite_contours(edges, ratio_threshold=0.9, min_area=1000):
    """
    Build composite contours using RETR_TREE. Each composite is a parent (area >= 10,000) and its valid children.
    """
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    composite_list = []
    if hierarchy is not None and len(contours) > 0:
        hierarchy = hierarchy[0]  # shape (N,4): [Next, Previous, First_Child, Parent]
        contour_areas = [cv2.contourArea(c) for c in contours]
        for i, h in enumerate(hierarchy):
            if h[3] == -1 and contour_areas[i] >= 10000:
                valid_child_indices = get_valid_children(i, hierarchy, contour_areas, ratio_threshold=ratio_threshold)
                comp = [contours[i]] + [contours[idx] for idx in valid_child_indices]
                composite_list.append(comp)
    return composite_list

def process_contours(image, edges):
    """
    Debug function to overlay contour hierarchy on the image.
    """
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or len(contours) == 0:
        print("No contours found.")
        return image

    hierarchy = hierarchy[0]
    contour_areas = [cv2.contourArea(c) for c in contours]
    print(f"Largest contour area: {max(contour_areas):.2f}")
    top_n = sorted(contour_areas, reverse=True)[:5]
    print("Top 5 contour areas:", [f"{a:.2f}" for a in top_n])

    parent_info = {}
    parent_count = 0
    for i, h in enumerate(hierarchy):
        if h[3] == -1 and contour_areas[i] >= 10000:
            parent_count += 1
            p_label = f"Parent{parent_count}"
            valid_children = get_valid_children(i, hierarchy, contour_areas, ratio_threshold=0.9)
            child_list = []
            child_count = 0
            for c_idx in valid_children:
                child_count += 1
                c_label = f"Child{parent_count}.{child_count}"
                child_list.append((c_idx, c_label, contour_areas[c_idx]))
            parent_info[i] = {"label": p_label, "area": contour_areas[i], "children": child_list}

    logging.debug(f"\nParent and Child Areas:")
    for parent_idx, info in parent_info.items():
        children_str = ", ".join(f"{lab} (Area: {a:.2f})" for (_, lab, a) in info["children"]) if info["children"] else "None"
        logging.debug(f"{info['label']} (Area: {info['area']:.2f}) : Children: {children_str}")

    vis_image = image.copy()
    for parent_idx, info in parent_info.items():
        cv2.drawContours(vis_image, [contours[parent_idx]], -1, (0, 255, 0), 2)
        M = cv2.moments(contours[parent_idx])
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(vis_image, info["label"], (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for (child_idx, child_label, _) in info["children"]:
            cv2.drawContours(vis_image, [contours[child_idx]], -1, (0, 0, 255), 2)
            M_child = cv2.moments(contours[child_idx])
            if M_child['m00'] != 0:
                cx_child = int(M_child['m10'] / M_child['m00'])
                cy_child = int(M_child['m01'] / M_child['m00'])
                cv2.putText(vis_image, child_label, (cx_child, cy_child), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return vis_image

# ---------------------------
# Shapely Polygon Conversions & IoU Calculation
# ---------------------------

def shapely_to_contours(poly, homography_inv, offset_x, offset_y):
    """
    Convert a Shapely polygon (with holes) into image contours.
    """
    exterior = np.array(poly.exterior.coords)
    exterior_img = du.map_world_to_image(exterior, homography_inv, offset_x, offset_y)
    contours = [np.array(exterior_img, dtype=int).reshape((-1, 1, 2))]
    for interior in poly.interiors:
        interior_coords = np.array(interior.coords)
        interior_img = du.map_world_to_image(interior_coords, homography_inv, offset_x, offset_y)
        contours.append(np.array(interior_img, dtype=int).reshape((-1, 1, 2)))
    return contours

def mask_from_contours(contours_list, frame_shape):
    """
    Create a binary mask from contours (with holes subtracted).
    """
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contours_list[0]], -1, 255, -1)
    for cnt in contours_list[1:]:
        cv2.drawContours(mask, [cnt], -1, 0, -1)
    return mask

def calculate_mask_iou(mask1, mask2):
    """
    Calculate IoU between two binary masks.
    """
    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)
    inter_area = np.sum(intersection == 255)
    union_area = np.sum(union == 255)
    return inter_area / union_area if union_area != 0 else 0.0

# ---------------------------
# Optimization Functions
# ---------------------------

def negative_iou(params, polygon, x_world, y_world, homography_inv, offset_x, offset_y, frame_shape, target_composite):
    """
    Objective function for optimization: returns negative IoU.
    """
    angle_deg, x_shift, y_shift = params
    rotated_poly = rotate(polygon, angle_deg, origin='centroid', use_radians=False)
    translated_poly = translate(rotated_poly, xoff=x_world + x_shift, yoff=y_world + y_shift)
    pred_contours = shapely_to_contours(translated_poly, homography_inv, offset_x, offset_y)
    pred_mask = mask_from_contours(pred_contours, frame_shape)
    target_mask = mask_from_contours(target_composite, frame_shape)
    iou = calculate_mask_iou(pred_mask, target_mask)
    return -iou

def optimize_angle(target_composite, x_world, y_world, polygon, homography_inv, offset_x, offset_y, frame_shape):
    """
    Optimize the alignment (rotation and translation) of the template polygon to maximize IoU.
    """
    objective = partial(
        negative_iou,
        polygon=polygon,
        x_world=x_world,
        y_world=y_world,
        homography_inv=homography_inv,
        offset_x=offset_x,
        offset_y=offset_y,
        frame_shape=frame_shape,
        target_composite=target_composite
    )
    bounds = [(-90, 90), (-3, 3), (-3, 3)]
    try:
        result = differential_evolution(
            objective,
            bounds=bounds,
            strategy='best1bin',
            maxiter=100,
            popsize=9,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            disp=False,
            updating='immediate',
            workers=1,
            seed=None
        )
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        return None, None, None, 0.0

    optimized_angle, x_shift, y_shift = result.x
    max_iou = -result.fun
    logging.debug(f"Optimized: angle={optimized_angle:.2f}°, x_shift={x_shift:.2f}, y_shift={y_shift:.2f}, IoU={max_iou:.4f}")
    return optimized_angle, x_shift, y_shift, max_iou

# ---------------------------
# Visualization Functions
# ---------------------------

def visualize_detections(frame, matched_composites, world_coords, template_polygon, gripping_points_world, homography_inv, offset_x, offset_y, gripper_line_angle):
    """
    Visualize detection results by drawing contours, centroids, optimized template alignment,
    and gripping points. The final command angle (optimized_angle + gripper_line_angle) is used
    for drawing the orientation line and displaying the angle, while the gripping points remain in
    their original positions (transformed only by the optimized_angle).
    """
    def draw_orientation_line(img, cnt, angle_deg):
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            return
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        length = 100
        angle_rad = np.radians(angle_deg)
        x2 = int(cx + length * np.cos(angle_rad))
        y2 = int(cy + length * np.sin(angle_rad))
        cv2.line(img, (cx, cy), (x2, y2), (255, 0, 0), 2)

    for i, ((comp, similarity), (x_world, y_world)) in enumerate(zip(matched_composites, world_coords), start=1):
        outer = comp[0]
        cv2.drawContours(frame, [outer], -1, (0, 255, 0), 2)
        for hole in comp[1:]:
            cv2.drawContours(frame, [hole], -1, (0, 0, 255), 2)

        M = cv2.moments(outer)
        if M['m00'] != 0:
            cX_img = int(M['m10'] / M['m00'])
            cY_img = int(M['m01'] / M['m00'])
        else:
            x_min, y_min, w, h = cv2.boundingRect(outer)
            cX_img, cY_img = x_min + w // 2, y_min + h // 2
        cv2.circle(frame, (cX_img, cY_img), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"Center: ({x_world:.1f}, {y_world:.1f}) mm",
                    (cX_img + 10, cY_img + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        target_for_optimization = filter_composite(comp)
        optimized_angle, x_shift, y_shift, max_iou = optimize_angle(
            target_for_optimization,
            x_world,
            y_world,
            template_polygon,
            homography_inv,
            offset_x,
            offset_y,
            frame.shape
        )

        # Compute the final command angle that will be sent to the robot.
        final_command_angle = optimized_angle + gripper_line_angle

        # Draw the orientation line (blue) using the final command angle.
        draw_orientation_line(frame, outer, final_command_angle)
        logging.info(f"Object {i}: X_coord={x_world:.2f}, Y_coord={y_world:.2f} Angle={final_command_angle:.2f}°, IoU={max_iou:.4f}")

        rotated_poly = rotate(template_polygon, optimized_angle, origin='centroid', use_radians=False)
        translated_poly = translate(rotated_poly, xoff=x_world + x_shift, yoff=y_world + y_shift)
        template_contours = shapely_to_contours(translated_poly, homography_inv, offset_x, offset_y)
        for cnt_poly in template_contours:
            cv2.drawContours(frame, [cnt_poly], -1, (0, 255, 255), 2)
        cv2.putText(frame, f"Angle: {final_command_angle:.1f} deg", (cX_img + 10, cY_img + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # For the gripping points, use the transformation computed from optimized_angle only.
        rotated_points = rotate_points(gripping_points_world, optimized_angle)
        shifted_points = rotated_points + np.array([x_world + x_shift, y_world + y_shift])
        gripping_points_image = du.map_world_to_image(shifted_points, homography_inv, offset_x, offset_y)
        for idx, pt in enumerate(gripping_points_image, start=1):
            x_pt, y_pt = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x_pt, y_pt), 5, (255, 0, 0), -1)
            cv2.putText(frame, f"G{idx}", (x_pt + 5, y_pt - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow('Object Detection', frame)


# ---------------------------
# Main Processing Functions
# ---------------------------

def run_static_image_test(test_image_path, calibration_data, offset_x, offset_y, template_polygon, gripping_points_world, gripper_line_angle):
    """
    Process a static test image.
    """
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

    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_mtx)
    edges = get_edges(undistorted, canny_thresh1=5, canny_thresh2=40)
    if DEBUG_EDGES:
        cv2.imshow("Edges", edges)
    if SHOW_HIERARCHY_DEBUG:
        hierarchy_vis = process_contours(undistorted, edges)
        cv2.imshow("Hierarchy Debug", hierarchy_vis)

    composites = get_composite_contours(edges)
    logging.info(f"Found {len(composites)} composite contour(s).")

    # Map the template polygon to image space
    template_exterior = np.array(template_polygon.exterior.coords)
    template_exterior_img = du.map_world_to_image(template_exterior, homography_inv, offset_x, offset_y)
    template_contour = np.array(template_exterior_img, dtype=int).reshape((-1, 1, 2))
    template_area = cv2.contourArea(template_contour)
    logging.debug(f"Template area: {template_area:.2f}")

    matched_composites = []
    for idx, comp in enumerate(composites):
        outer = comp[0]
        similarity = cv2.matchShapes(template_contour, outer, cv2.CONTOURS_MATCH_I1, 0.0)
        parent_area = cv2.contourArea(outer)
        area_diff = abs(parent_area - template_area) / template_area if template_area > 0 else 1.0
        logging.debug(f"Composite {idx}: similarity={similarity:.4f}, area_diff={area_diff:.2%}")
        if similarity < THRESHOLD and area_diff < AREA_TOLERANCE:
            matched_composites.append((comp, similarity))
            logging.debug(f"Composite {idx} matched.")
        else:
            logging.debug(f"Composite {idx} rejected.")

    logging.info(f"Found {len(matched_composites)} matching composite contour(s).")

    centroids_img = []
    for comp, _ in matched_composites:
        M = cv2.moments(comp[0])
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            centroids_img.append((cx, cy))
    world_coords = du.map_image_to_world(centroids_img, homography, offset_x, offset_y)

    start_time = time.perf_counter()
    visualize_detections(undistorted, matched_composites, world_coords, template_polygon, gripping_points_world, homography_inv, offset_x, offset_y, gripper_line_angle)
    end_time = time.perf_counter()
    logging.info(f"Optimization and visualization completed in {end_time - start_time:.4f} seconds.")
    print("Test image processed. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_live_detection(calibration_data, offset_x, offset_y, template_polygon, gripping_points_world, gripper_line_angle):
    """
    Run live detection using a camera feed. Press 'spacebar' to capture and process a frame.
    The live feed and processed output are combined into one window ("Live Detection").
    """
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

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f"Requested Resolution: {DESIRED_WIDTH}x{DESIRED_HEIGHT} at {DESIRED_FPS} FPS")
    logging.info(f"Actual Resolution: {int(actual_width)}x{int(actual_height)} at {int(actual_fps)} FPS")
    logging.info("Live detection started. Press 'q' to quit, 'spacebar' to capture image.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to grab frame.")
            break

        # Resize and add instructions to the live feed image
        display_frame = cv2.resize(frame, (int(actual_width), int(actual_height)))
        cv2.putText(display_frame, "Press 'spacebar' to capture, 'q' to quit.",
                    (80, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Object Detection', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logging.info("Exiting live detection.")
            break
        elif key == ord(' '):
            # Process the captured frame
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_mtx)
            edges = get_edges(undistorted, canny_thresh1=5, canny_thresh2=40)
            if DEBUG_EDGES:
                cv2.imshow("Edges", edges)  # Optional: debug window for edges
            if SHOW_HIERARCHY_DEBUG:
                hierarchy_vis = process_contours(undistorted, edges)
                cv2.imshow("Hierarchy Debug", hierarchy_vis)  # Optional: debug window for hierarchy

            composites = get_composite_contours(edges)
            logging.info(f"Found {len(composites)} composite contour(s).")
            
            # Map the template polygon to image space
            template_exterior = np.array(template_polygon.exterior.coords)
            template_exterior_img = du.map_world_to_image(template_exterior, homography_inv, offset_x, offset_y)
            template_contour = np.array(template_exterior_img, dtype=int).reshape((-1, 1, 2))
            template_area = cv2.contourArea(template_contour)
            
            matched_composites = []
            for idx, comp in enumerate(composites):
                outer = comp[0]
                similarity = cv2.matchShapes(template_contour, outer, cv2.CONTOURS_MATCH_I1, 0.0)
                parent_area = cv2.contourArea(outer)
                area_diff = abs(parent_area - template_area) / template_area if template_area > 0 else 1.0
                logging.debug(f"Composite {idx}: similarity={similarity:.4f}, area_diff={area_diff:.2%}")
                if similarity < THRESHOLD and area_diff < AREA_TOLERANCE:
                    matched_composites.append((comp, similarity))
                    logging.debug(f"Composite {idx} matched.")
                else:
                    logging.debug(f"Composite {idx} rejected.")
            logging.info(f"Found {len(matched_composites)} matching composite contour(s).")

            centroids_img = []
            for comp, _ in matched_composites:
                M = cv2.moments(comp[0])
                if M['m00'] != 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    centroids_img.append((cx, cy))
            world_coords = du.map_image_to_world(centroids_img, homography, offset_x, offset_y)
            
            start_time = time.perf_counter()
            # Overlay the detection results on the undistorted frame
            visualize_detections(undistorted, matched_composites, world_coords,
                                 template_polygon, gripping_points_world, homography_inv, offset_x, offset_y, gripper_line_angle)
            end_time = time.perf_counter()
            logging.info(f"Optimization and visualization completed in {end_time - start_time:.4f} seconds.")
            
            print("Processing complete. Press 'spacebar' to capture again or 'q' to quit.")
            while True:
                key_inner = cv2.waitKey(1) & 0xFF
                if key_inner == ord('q'):
                    logging.info("Exiting live detection.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key_inner == ord(' '):
                    break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------
# Main Entry Point
# ---------------------------

def main():
    """
    Load calibration and gripping data, scale the template polygon,
    and run detection in static or live mode based on the flag.
    """
    logging.basicConfig(
        level=LOGGING_LEVEL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler("data/live_object_detection.log"),
                  logging.StreamHandler()]
    )
    try:
        calibration_data, offset_x, offset_y = du.load_calibration_data(CALIBRATION_FILE_PATH)
    except Exception as e:
        logging.error("Calibration data load failed: " + str(e))
        return

    try:
        template_polygon, gripping_points, gripper_distance, gripper_line_angle = du.load_gripping_data(GRIPPER_DATA_PATH)
        scaled_polygon = scale(template_polygon, xfact=1.025, yfact=1.025, origin='centroid')
    except Exception as e:
        logging.error("Gripping data load failed: " + str(e))
        return


    if USE_STATIC_IMAGE:
        run_static_image_test(TEST_IMAGE_PATH, calibration_data, offset_x, offset_y, scaled_polygon, gripping_points, gripper_line_angle)
    else:
        run_live_detection(calibration_data, offset_x, offset_y, scaled_polygon, gripping_points, gripper_line_angle)

if __name__ == "__main__":
    main()
