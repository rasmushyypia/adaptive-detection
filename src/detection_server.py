#!/usr/bin/env python3
"""
detection_server.py – Live Object Detection + Socket Server

1) Launches a live camera feed in one thread, continuously displaying frames in an OpenCV window.
2) Runs a multithreaded socket server on another thread. When a client sends the command:
   "get_vision_data"
   the server:
     a) Takes the most recent camera frame,
     b) Runs the detection pipeline,
     c) Updates the display with detection overlays,
     d) Returns the detection result as a string in the format:
        "( distance_in_mm, x_coordinate, y_coordinate, r_angle )"
"""

import os
import cv2
import numpy as np
import logging
import time
import threading
from functools import partial
from shapely.affinity import rotate, translate, scale
import socket
from scipy.optimize import differential_evolution

# Local utilities for mapping and data loading
import utils.detection_utils as du



# ---------------------------
# Global Configuration & Paths
# ---------------------------
CALIBRATION_FILE_PATH = 'data/calibration_data.pkl'
GRIPPER_DATA_PATH      = 'data/gripping_data.pkl'

CAMERA_INDEX   = 0
DESIRED_WIDTH  = 1920
DESIRED_HEIGHT = 1080
DESIRED_FPS    = 30
BACKEND        = cv2.CAP_DSHOW

THRESHOLD       = 0.08   # Template matching threshold
AREA_TOLERANCE  = 0.15   # 15% area tolerance

# ---------------------------
# Globals for Multithreading
# ---------------------------
most_recent_frame = None       # The latest frame from the camera
detection_overlay = None       # The latest detection overlay (if any)
detection_overlay_timestamp = None  # Timestamp to track how long the detection is shown
frame_lock        = threading.Lock()
detection_lock    = threading.Lock()

# Placeholders to store calibration data
calibration_data  = None
offset_x, offset_y = 0, 0

# Template polygon, gripping points, distance, and extra gripper_line_angle
scaled_polygon    = None
gripping_points   = None
gripper_distance  = 0.0
gripper_line_angle = 0.0

# ---------------------------
# Utility Functions
# ---------------------------
def normalize_angle_old(angle):
    """
    Normalize an angle (in degrees) to be within [-90, 90].
    """
    while angle > 90:
        angle -= 180
    while angle < -90:
        angle += 180
    return angle

def normalize_angle(angle, lower_bound=-120, upper_bound=60):
    """
    Normalize an angle (in degrees) to be within [lower_bound, upper_bound),
    where the range spans 180 degrees. Default is [-135, 45].
    """
    range_width = upper_bound - lower_bound  # This should be 180
    normalized = (angle - lower_bound) % range_width  # Maps to [0, 180)
    return normalized + lower_bound

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

def rotate_points(points, angle_deg):
    """Rotate a set of 2D points by angle_deg around the origin (0,0)."""
    if points.size == 0:
        return points
    angle_rad = np.deg2rad(angle_deg)
    rot_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                        [np.sin(angle_rad),  np.cos(angle_rad)]])
    return np.dot(points, rot_mat.T)

# Shapely/IoU helpers
def shapely_to_contours(poly, homography_inv, ox, oy, frame_shape):
    """
    Convert a Shapely polygon (with holes) into image contours.
    Returns a list of contours (outer + holes).
    """
    exterior = np.array(poly.exterior.coords)
    exterior_img = du.map_world_to_image(exterior, homography_inv, ox, oy)
    exterior_img = np.clip(exterior_img, [0, 0], [frame_shape[1]-1, frame_shape[0]-1])
    contours_list = [np.array(exterior_img, dtype=int).reshape((-1, 1, 2))]

    for interior in poly.interiors:
        interior_coords = np.array(interior.coords)
        interior_img = du.map_world_to_image(interior_coords, homography_inv, ox, oy)
        interior_img = np.clip(interior_img, [0, 0], [frame_shape[1]-1, frame_shape[0]-1])
        contours_list.append(np.array(interior_img, dtype=int).reshape((-1, 1, 2)))
    return contours_list

def mask_from_contours(contours_list, frame_shape):
    """Create a binary mask from the list of (outer + holes) contours."""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contours_list[0]], -1, 255, -1)
    for hole in contours_list[1:]:
        cv2.drawContours(mask, [hole], -1, 0, -1)
    return mask

def calculate_mask_iou(mask1, mask2):
    """Compute Intersection-over-Union for two binary masks."""
    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)
    inter_area = np.sum(intersection == 255)
    union_area = np.sum(union == 255)
    return float(inter_area) / float(union_area) if union_area > 0 else 0.0

def negative_iou(params, polygon, x_world, y_world, homography_inv, ox, oy, frame_shape, target_composite):
    """
    Objective function: negative IoU of the template vs. target contour composite.
    """
    angle_deg, x_shift, y_shift = params
    rotated_poly = rotate(polygon, angle_deg, origin='centroid', use_radians=False)
    translated_poly = translate(rotated_poly, xoff=x_world + x_shift, yoff=y_world + y_shift)
    pred_contours = shapely_to_contours(translated_poly, homography_inv, ox, oy, frame_shape)
    pred_mask = mask_from_contours(pred_contours, frame_shape)
    target_mask = mask_from_contours(target_composite, frame_shape)
    iou_value = calculate_mask_iou(pred_mask, target_mask)
    return -iou_value

def optimize_angle(target_composite, x_world, y_world, polygon, homography_inv, ox, oy, frame_shape):
    """Optimize rotation and small translation to maximize IoU with a target composite contour."""
    objective = partial(
        negative_iou,
        polygon=polygon,
        x_world=x_world,
        y_world=y_world,
        homography_inv=homography_inv,
        ox=ox,
        oy=oy,
        frame_shape=frame_shape,
        target_composite=target_composite
    )
    bounds = [(0, 360), (-3, 3), (-3, 3)]
    try:
        result = differential_evolution(
            objective, bounds=bounds,
            strategy='best1bin',
            maxiter=120,
            popsize=8,
            tol=0.02,
            mutation=(0.5, 1),
            recombination=0.7,
            disp=False
        )
        angle_deg, x_shift, y_shift = result.x
        best_iou = -result.fun
        return angle_deg, x_shift, y_shift, best_iou
    except Exception as e:
        logging.error("Optimization failed: {}".format(e))
        return 0.0, 0.0, 0.0, 0.0

def perform_detection(frame):
    """
    Core detection routine:
    1) Undistort the frame.
    2) Detect edges and extract composite contours.
    3) Use shape matching to find candidates.
    4) For each matched composite, compute the area difference from the template.
    5) Select the candidate with the smallest area difference and optimize its alignment.
    6) Outline all candidates and return the detection result for the best candidate.
    Returns [distance_in_mm, x_coord, y_coord, final_command_angle] and an overlay image.
    """
    global calibration_data, offset_x, offset_y, flip_mapping_origin
    global scaled_polygon, gripping_points, gripper_distance, gripper_line_angle

    try:
        camera_matrix  = calibration_data['camera_matrix']
        dist_coeffs    = calibration_data['dist_coeffs']
        new_camera_mtx = calibration_data['new_camera_mtx']
        homography     = calibration_data['homography']
        homography_inv = np.linalg.inv(homography)
    except KeyError as e:
        logging.error("Missing calibration key: {}".format(e))
        return [0, 0, 0, 0], frame

    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_mtx)
    edges = get_edges(undistorted)
    composites = get_composite_contours(edges)
    
    # Prepare template data for matching.
    template_exterior = np.array(scaled_polygon.exterior.coords)
    template_exterior_img = du.map_world_to_image(template_exterior, homography_inv, offset_x, offset_y)
    template_contour = np.array(template_exterior_img, dtype=int).reshape((-1, 1, 2))
    template_area = cv2.contourArea(template_contour)
    
    matched_composites = []
    candidate_info = []
    for comp in composites:
        outer = comp[0]
        similarity = cv2.matchShapes(template_contour, outer, cv2.CONTOURS_MATCH_I1, 0.0)
        parent_area = cv2.contourArea(outer)
        area_diff = abs(parent_area - template_area) / template_area if template_area > 0 else 1.0
        if similarity < THRESHOLD and area_diff < AREA_TOLERANCE:
            matched_composites.append(comp)
            # Compute the centroid using image moments.
            M = cv2.moments(outer)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                x_min, y_min, w, h = cv2.boundingRect(outer)
                cx, cy = x_min + w/2, y_min + h/2

            world_coord = du.map_image_to_world([(cx, cy)], homography, offset_x, offset_y)
            x_world, y_world = world_coord[0][0], world_coord[0][1]

            candidate_info.append({
                'comp': comp,
                'cx': cx,
                'cy': cy,
                'x_world': x_world,
                'y_world': y_world,
                'area_diff': area_diff
            })

    if not candidate_info:
        return [0, 0, 0, 0], undistorted

    # Select the best candidate based on the smallest area_diff.
    best_candidate = min(candidate_info, key=lambda x: x['area_diff'])

    # Run optimization only for the best candidate.
    best_comp = best_candidate['comp']
    x_world = best_candidate['x_world']
    y_world = best_candidate['y_world']
    angle_deg, x_shift, y_shift, best_iou = optimize_angle(
        best_comp, x_world, y_world, scaled_polygon, homography_inv,
        offset_x, offset_y, undistorted.shape
    )
    # Update best candidate info with optimization results.
    best_candidate.update({
        'angle_deg': angle_deg,
        'x_shift': x_shift,
        'y_shift': y_shift,
        'best_iou': best_iou
    })

    # Prepare the overlay for visualization.
    overlay = undistorted.copy()
    # Draw all candidate contours in green.
    for cand in candidate_info:
        outer = cand['comp'][0]
        cv2.drawContours(overlay, [outer], -1, (0, 255, 0), 2)
    
    # Highlight the best candidate with extra markers.
    best = best_candidate
    cX_img, cY_img = int(best['cx']), int(best['cy'])
    cv2.circle(overlay, (cX_img, cY_img), 5, (0, 0, 255), -1)
    cv2.putText(overlay, "Center: ({:.1f}, {:.1f})mm".format(best['x_world'], best['y_world']),
                (cX_img+10, cY_img+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.putText(overlay, "IoU: {:.2f}".format(best['best_iou']),
                (cX_img+10, cY_img+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    final_command_angle = normalize_angle(best['angle_deg'] + gripper_line_angle)
    length = 100
    angle_rad = np.deg2rad(final_command_angle)
    x2 = int(cX_img + length * np.cos(angle_rad))
    y2 = int(cY_img + length * np.sin(angle_rad))
    cv2.line(overlay, (cX_img, cY_img), (x2, y2), (255, 0, 0), 2)
    cv2.putText(overlay, "Angle: {:.1f} deg".format(final_command_angle),
                (cX_img+10, cY_img+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    
    # Draw the template contours based on the optimized best candidate.
    rotated_poly = rotate(scaled_polygon, best['angle_deg'], origin='centroid', use_radians=False)
    translated_poly = translate(rotated_poly, xoff=best['x_world'] + best['x_shift'], yoff=best['y_world'] + best['y_shift'])
    template_cnts = shapely_to_contours(translated_poly, homography_inv, offset_x, offset_y, overlay.shape)
    for cnt in template_cnts:
        cv2.drawContours(overlay, [cnt], -1, (0, 255, 255), 2)
    
    # Compute and draw gripping points.
    rotated_gpoints = rotate_points(gripping_points, best['angle_deg'])
    shifted_gpoints = rotated_gpoints + np.array([best['x_world'] + best['x_shift'], best['y_world'] + best['y_shift']])
    gp_img_coords = mp.map_world_to_image(shifted_gpoints, homography_inv, offset_x, offset_y)
    for i, pt in enumerate(gp_img_coords, start=1):
        px, py = int(pt[0]), int(pt[1])
        cv2.circle(overlay, (px, py), 5, (255, 0, 0), -1)
        cv2.putText(overlay, "G{}".format(i), (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    result_data = [float(gripper_distance), float(best['x_world']), float(best['y_world']), float(final_command_angle)]
    return result_data, overlay

# ---------------------------
# Socket-Exposed Function
# ---------------------------
def socket_get_vision_data(conn):
    """
    When a client sends "get_vision_data" over the socket,
    run the detection pipeline on the most recent frame,
    update the display overlay, and return the detection result
    as a formatted string.
    """
    global most_recent_frame, detection_overlay, detection_overlay_timestamp
    with frame_lock:
        if most_recent_frame is None:
            detection_result = [0.0, 0.0, 0.0, 0.0]
            frame_copy = None
        else:
            frame_copy = most_recent_frame.copy()

    if frame_copy is not None:
        detection_result, new_overlay = perform_detection(frame_copy)
        with detection_lock:
            detection_overlay = new_overlay
            detection_overlay_timestamp = time.time()
        detection_result = [round(val, 2) for val in detection_result]
    else:
        detection_result = [0.0, 0.0, 0.0, 0.0]

    response = "( {:.2f}, {:.2f}, {:.2f}, {:.2f} )".format(*detection_result)
    logging.info("Sending message: %s", response)
    try:
        conn.sendall(response.encode("utf-8"))
    except Exception as e:
        logging.error("Error sending response: {}".format(e))

# ---------------------------
# Socket Server Functions
# ---------------------------
def handle_client(conn, addr):
    logging.info("Socket client connected: {}".format(addr))
    try:
        data = conn.recv(1024).decode("utf-8").strip()
        logging.info("Received request: {}".format(data))
        if data == "get_vision_data":
            socket_get_vision_data(conn)
        else:
            conn.sendall("Unknown command".encode("utf-8"))
    except Exception as e:
        logging.error("Socket server error: {}".format(e))
    finally:
        conn.close()

def run_socket_server(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        logging.info("Vision socket server listening on {}:{}".format(host, port))
        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

# ---------------------------
# Camera Loop
# ---------------------------
def camera_loop():
    global most_recent_frame, detection_overlay, detection_overlay_timestamp
    cap = cv2.VideoCapture(CAMERA_INDEX, BACKEND)
    if not cap.isOpened():
        logging.error("Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, DESIRED_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to read frame from camera.")
            break

        with frame_lock:
            most_recent_frame = frame

        with detection_lock:
            if detection_overlay is not None and detection_overlay_timestamp is not None:
                if time.time() - detection_overlay_timestamp < 10:
                    display_img = detection_overlay
                else:
                    detection_overlay = None
                    detection_overlay_timestamp = None
                    display_img = frame
            else:
                display_img = frame

        cv2.imshow("Live Object Detection", display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# Main Entry Point
# ---------------------------
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    global calibration_data, offset_x, offset_y, flip_mapping_origin
    global scaled_polygon, gripping_points, gripper_distance, gripper_line_angle

    try:
        # -- Load calibration, including flip info --
        calibration_data, offset_x, offset_y = du.load_calibration_data(CALIBRATION_FILE_PATH)
        flip_mapping_origin = calibration_data.get('flip_mapping_origin', False)
        logging.info(f"flip_mapping_origin from calibration: {flip_mapping_origin}")
    except Exception as e:
        logging.error("Calibration data load failed: {}".format(e))
        return

    try:
        template_polygon, gripping_points, gripper_distance, gripper_line_angle = du.load_gripping_data(GRIPPER_DATA_PATH)
        gripper_line_angle = normalize_angle(gripper_line_angle)
        scaled_polygon = scale(template_polygon, xfact=1.025, yfact=1.025, origin='centroid')
    except Exception as e:
        logging.error("Gripping data load failed: {}".format(e))
        return

    # Start the socket server in a separate thread.
    server_ip = "0.0.0.0"
    server_port = 40411
    socket_thread = threading.Thread(target=run_socket_server, args=(server_ip, server_port), daemon=True)
    socket_thread.start()
    logging.info("Starting Vision Socket Server on {}:{}".format(server_ip, server_port))

    logging.info("Starting live camera feed. Press 'q' to quit.")
    camera_loop()

    logging.info("Exiting.")

if __name__ == "__main__":
    main()
