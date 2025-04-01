#!/usr/bin/env python3
"""
multi_shape_detection.py – Combined Script with min_max_distance Optimization and IoU Threshold

1) Load/Optimize Many DXFs:
   - Reads each .dxf from a folder (e.g. 'data/dxf_shapes/').
   - For each DXF, computes a Shapely polygon (with holes), insets an "inner contour,"
     places two grippers, and then optimizes those gripper points using a
     min_max_distance approach (without a GUI).
   - Stores results (polygon, final gripping points, line angle, etc.) in memory.

2) Multi-Shape Detection:
   - Uses the EXACT same logic from the old detection_test.py for:
       - get_edges, get_composite_contours, filter_composite, process_contours
       - shapely_to_contours, negative_iou, optimize_angle, etc.
   - But extends it to detect and visualize ANY of the loaded shapes in either
     a static test image (USE_STATIC_IMAGE = True) or a live camera feed.
   - Press space to capture (in live mode) or simply run once on a static image.

NEW FEATURE:
   - We enforce a minimum IoU threshold (0.90). If a shape matches the outer boundary
     but the holes differ (or the alignment is poor), we discard it after IoU optimization.

Adjust file paths, folder names, and references to 'utils' as needed.
"""

import os
import glob
import logging
import cv2
import numpy as np
import time
import pickle
from functools import partial
from scipy.optimize import differential_evolution

# Shapely for geometry
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from shapely.affinity import rotate, translate, scale

# Local utilities (adjust if needed)
import utils.dxf_reading as dxf_reading  # For reading DXFs into Shapely polygons
import utils.detection_utils as du

# ---------------------------
# Global Configuration & Paths
# ---------------------------
CALIBRATION_FILE_PATH = 'data/calibration_data.pkl'
DXF_FOLDER = 'data/dxf_shapes'     # Folder with multiple .dxf files
TEST_IMAGE_PATH = 'data/test_general.jpg'

# For detection pipeline
THRESHOLD = 0.08           # Template matching similarity threshold
AREA_TOLERANCE = 0.15      # Tolerance for area difference (15%)
LOGGING_LEVEL = logging.DEBUG

DEBUG_EDGES = True
SHOW_HIERARCHY_DEBUG = True

# Camera config
USE_STATIC_IMAGE = True
CAMERA_INDEX = 0
DESIRED_WIDTH = 1920
DESIRED_HEIGHT = 1080
DESIRED_FPS = 30
BACKEND = cv2.CAP_DSHOW

# Gripper geometry for auto-optimization
GRIPPER_RADIUS = 10
INNER_CONTOUR_BUFFER = 3
HOLE_SAFETY_BUFFER = 5

# IoU Threshold (to reject mismatches)
IOU_THRESHOLD = 0.93

# ---------------------------
# PART A: Optimize DXFs Without GUI
# ---------------------------

def get_holes_union_with_buffer(final_poly, hole_safety_buffer=0.0):
    """
    Return a single geometry representing the union of all holes
    (optionally buffered outward by hole_safety_buffer).
    """
    if not final_poly.interiors:
        return None
    hole_polygons = []
    for hole in final_poly.interiors:
        hole_poly = Polygon(hole.coords)
        if hole_safety_buffer > 0.0:
            hole_poly = hole_poly.buffer(hole_safety_buffer)
        hole_polygons.append(hole_poly)
    return unary_union(hole_polygons)

def build_inner_contour_and_default_grippers(final_poly):
    """
    Build the 'inner' offset contour from the outer boundary (inset).
    Place two gripper points on opposite ends of that inner contour.
    """
    outer_polygon = Polygon(final_poly.exterior.coords)
    inset_dist = GRIPPER_RADIUS + INNER_CONTOUR_BUFFER
    inner_polygon = outer_polygon.buffer(-inset_dist)

    if inner_polygon.is_empty:
        return None, None
    if inner_polygon.geom_type == 'MultiPolygon':
        # Pick the largest piece if multiple
        inner_polygon = max(inner_polygon.geoms, key=lambda p: p.area)

    in_coords = np.array(inner_polygon.exterior.coords)
    inner_contour = LineString(in_coords)

    c_len = inner_contour.length
    p1 = inner_contour.interpolate(0.0)
    p2 = inner_contour.interpolate(c_len / 2.0)
    gripping_points = np.array([[p1.x, p1.y], [p2.x, p2.y]])
    return inner_contour, gripping_points

def compute_gripper_line_angle(gripping_points):
    """
    Angle of line connecting G1->G2, normalized to [-90..90].
    """
    if gripping_points is None or len(gripping_points) < 2:
        return 0.0
    p1, p2 = gripping_points
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_deg = np.degrees(np.arctan2(dy, dx))
    while angle_deg > 90:
        angle_deg -= 180
    while angle_deg < -90:
        angle_deg += 180
    return angle_deg

def optimize_two_grippers(inner_contour, holes_union, outer_poly):
    """
    Differential evolution to place two grippers along 'inner_contour'
    using a "minimize maximum distance from boundary" approach.
    - We measure the union of the two gripper circles, compute its convex hull,
      then measure the maximum distance from that hull to the outer boundary points.
    - Minimizing that maximum distance effectively forces the grippers to remain
      as close as possible to the boundary, while also applying a heavy penalty
      if any circle intersects a hole.

    Args:
        inner_contour: shapely LineString for the inset boundary.
        holes_union:   shapely geometry union of holes (buffered) or None.
        outer_poly:    shapely Polygon representing the outer shape.

    Returns:
        A 2x2 array of final [x, y] points for the two grippers.
    """
    # Prepare the boundary points for distance measurement
    boundary_pts = [Point(bp) for bp in outer_poly.exterior.coords]
    c_len = inner_contour.length

    def get_xy(s):
        s %= c_len
        pt = inner_contour.interpolate(s)
        return np.array([pt.x, pt.y])

    def base_objective(s):
        # s = [s1, s2]
        p1 = get_xy(s[0])
        p2 = get_xy(s[1])
        union_of_grips = unary_union([
            Point(p1).buffer(GRIPPER_RADIUS),
            Point(p2).buffer(GRIPPER_RADIUS)
        ])
        hull = union_of_grips.convex_hull

        # measure the maximum distance from hull to any boundary point
        max_dist = max(hull.exterior.distance(bpt) for bpt in boundary_pts)
        return max_dist

    def hole_penalty(s):
        if holes_union is None or holes_union.is_empty:
            return 0.0
        penalty_val = 0.0
        for i in range(2):
            p = get_xy(s[i])
            circle = Point(p).buffer(GRIPPER_RADIUS)
            if circle.intersects(holes_union):
                penalty_val += 1e6
        return penalty_val

    def combined_obj(s):
        return base_objective(s) + hole_penalty(s)

    bounds = [(0, c_len)]*2
    result = differential_evolution(
        combined_obj, bounds=bounds, maxiter=50, popsize=12,
        mutation=(0.5, 1.0), recombination=0.7, strategy='best1bin',
        disp=False, tol=1e-3, polish=True
    )
    s1, s2 = result.x
    p1, p2 = get_xy(s1), get_xy(s2)
    return np.array([p1, p2])

def optimize_dxf_shape(dxf_path):
    """
    Reads a single DXF, computes final_poly, builds an inner contour,
    and runs differential evolution to 'minimize maximum distance from boundary'.
    Returns a dictionary with final polygon & gripper info.
    """
    final_poly = dxf_reading.extract_polygons_from_dxf(dxf_path, circle_approx_points=80)
    if final_poly is None:
        logging.warning(f"No valid polygon found in {dxf_path}")
        return None

    # Outer polygon for boundary measurement
    outer_poly = Polygon(final_poly.exterior.coords)
    holes_union = get_holes_union_with_buffer(final_poly, HOLE_SAFETY_BUFFER)
    inner_contour, default_grips = build_inner_contour_and_default_grippers(final_poly)
    if inner_contour is None:
        logging.warning(f"Inner contour empty or invalid for {dxf_path}")
        return None

    # Optimize with min_max_distance approach
    optimized_points = optimize_two_grippers(inner_contour, holes_union, outer_poly)
    dist = np.linalg.norm(optimized_points[0] - optimized_points[1])
    angle_deg = compute_gripper_line_angle(optimized_points)

    return {
        "final_poly": final_poly,
        "gripping_points": optimized_points,
        "gripper_distance": dist,
        "gripper_line_angle": angle_deg
    }

def optimize_all_dxfs_in_folder(folder):
    """
    Reads all *.dxf in `folder`, calls `optimize_dxf_shape` for each,
    returns a dict: shape_name -> data
    """
    data_dict = {}
    dxf_files = glob.glob(os.path.join(folder, "*.dxf"))
    if not dxf_files:
        logging.warning(f"No .dxf found in {folder}")
        return data_dict

    for path in dxf_files:
        shape_name = os.path.splitext(os.path.basename(path))[0]
        logging.info(f"Optimizing shape '{shape_name}' from {path}")
        result = optimize_dxf_shape(path)
        if result is not None:
            data_dict[shape_name] = result
    return data_dict


# ---------------------------
# PART B: Original Detection Code (Unchanged Logic)
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
    Build composite contours using RETR_TREE. Each composite is a parent (area >= 10,000) + valid children.
    """
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    composite_list = []
    if hierarchy is not None and len(contours) > 0:
        hierarchy = hierarchy[0]
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
        if info["children"]:
            kids_str = ", ".join(f"{lab} (Area: {a:.2f})" for (_, lab, a) in info["children"])
        else:
            kids_str = "None"
        logging.debug(f"{info['label']} (Area: {info['area']:.2f}) : Children: {kids_str}")

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

# Shapely conversions & IoU from detection_test.py

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

def negative_iou(params, polygon, x_world, y_world, homography_inv, offset_x, offset_y, frame_shape, target_composite):
    angle_deg, x_shift, y_shift = params
    rotated_poly = rotate(polygon, angle_deg, origin='centroid', use_radians=False)
    translated_poly = translate(rotated_poly, xoff=x_world + x_shift, yoff=y_world + y_shift)
    pred_contours = shapely_to_contours(translated_poly, homography_inv, offset_x, offset_y)
    pred_mask = mask_from_contours(pred_contours, frame_shape)
    target_mask = mask_from_contours(target_composite, frame_shape)
    iou = calculate_mask_iou(pred_mask, target_mask)
    return -iou

def optimize_angle(target_composite, x_world, y_world, polygon, homography_inv, offset_x, offset_y, frame_shape):
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
            disp=False
        )
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        return None, None, None, 0.0

    angle_opt, x_shift, y_shift = result.x
    max_iou = -result.fun
    logging.debug(f"Optimized: angle={angle_opt:.2f}°, shift=({x_shift:.2f},{y_shift:.2f}), IoU={max_iou:.4f}")
    return angle_opt, x_shift, y_shift, max_iou

def visualize_detections(frame, matched_composites, world_coords,
                         template_polygon, gripping_points_world,
                         homography_inv, offset_x, offset_y,
                         gripper_line_angle):
    """
    The final command angle = (optimized_angle + gripper_line_angle).
    We now enforce IoU_THRESHOLD=0.90. If max_iou < that threshold,
    we reject the detection (don't draw final orientation or grips).
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

    for i, ((comp, similarity), (x_w, y_w)) in enumerate(zip(matched_composites, world_coords), start=1):
        outer = comp[0]
        # Draw the matched composite boundary (in green and red for holes)
        cv2.drawContours(frame, [outer], -1, (0, 255, 0), 2)
        for hole in comp[1:]:
            cv2.drawContours(frame, [hole], -1, (0, 0, 255), 2)

        # Mark the centroid
        M = cv2.moments(outer)
        if M['m00'] != 0:
            cX_img = int(M['m10'] / M['m00'])
            cY_img = int(M['m01'] / M['m00'])
        else:
            x_min, y_min, w, h = cv2.boundingRect(outer)
            cX_img, cY_img = x_min + w // 2, y_min + h // 2

        cv2.circle(frame, (cX_img, cY_img), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"Center: ({x_w:.1f}, {y_w:.1f}) mm",
                    (cX_img + 10, cY_img + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Composite + holes => IoU-based orientation refinement
        target_for_opt = filter_composite(comp)
        opt_angle, x_shift, y_shift, max_iou = optimize_angle(
            target_for_opt, x_w, y_w,
            template_polygon, homography_inv,
            offset_x, offset_y,
            frame.shape
        )

        final_command_angle = (opt_angle + gripper_line_angle)
        logging.info(f"Detected Object: X={x_w:.2f}, Y={y_w:.2f}, Angle={final_command_angle:.2f}°, IoU={max_iou:.4f}")

        # -- IoU THRESHOLD CHECK (Reject if below 0.90) --
        if max_iou < IOU_THRESHOLD:
            logging.info(f"Rejected object: IoU={max_iou:.4f} < {IOU_THRESHOLD}")
            # We do NOT draw orientation line or grips, skip to next object
            continue

        # If IoU >= threshold, finalize the detection:
        draw_orientation_line(frame, outer, final_command_angle)

        # Draw the (rotated+shifted) template polygon
        rotated_poly = rotate(template_polygon, opt_angle, origin='centroid', use_radians=False)
        translated_poly = translate(rotated_poly, xoff=x_w + x_shift, yoff=y_w + y_shift)
        template_cnts = shapely_to_contours(translated_poly, homography_inv, offset_x, offset_y)
        for cnt_poly in template_cnts:
            cv2.drawContours(frame, [cnt_poly], -1, (0, 255, 255), 2)

        cv2.putText(frame, f"Angle: {final_command_angle:.1f} deg",
                    (cX_img + 10, cY_img + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # For gripper points, rotate by opt_angle only
        rotated_points = rotate_points(gripping_points_world, opt_angle)
        shifted_points = rotated_points + np.array([x_w + x_shift, y_w + y_shift])
        grips_img = du.map_world_to_image(shifted_points, homography_inv, offset_x, offset_y)
        for g_idx, pt in enumerate(grips_img, start=1):
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(frame, (px, py), 5, (255, 0, 0), -1)
            cv2.putText(frame, f"G{g_idx}", (px+5, py-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Object Detection", frame)


# ---------------------------
# PART C: Multi-Shape Extension of the Original Routines
# ---------------------------

def run_static_image_multi(test_image_path, calibration_data, offset_x, offset_y, multi_shape_data):
    """
    Similar to run_static_image_test, but loops over each shape in 'multi_shape_data'.
    We do shape matching and IoU-based refinement, then apply the 0.90 IoU threshold
    before finalizing the detection.
    """
    try:
        camera_matrix = calibration_data['camera_matrix']
        dist_coeffs   = calibration_data['dist_coeffs']
        new_camera_mtx= calibration_data['new_camera_mtx']
        homography    = calibration_data['homography']
        homography_inv= np.linalg.inv(homography)
    except KeyError as e:
        logging.error(f"Missing calibration key: {e}")
        return

    if not os.path.exists(test_image_path):
        logging.error(f"Test image not found: {test_image_path}")
        return

    frame = cv2.imread(test_image_path)
    if frame is None:
        logging.error("Failed to load test image.")
        return

    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_mtx)
    edges = get_edges(undistorted, 5, 40)
    if DEBUG_EDGES:
        cv2.imshow("Edges", edges)
    if SHOW_HIERARCHY_DEBUG:
        hierarchy_vis = process_contours(undistorted, edges)
        cv2.imshow("Hierarchy Debug", hierarchy_vis)

    composites = get_composite_contours(edges)
    logging.info(f"Found {len(composites)} composite contour(s).")

    # We'll modify the undistorted image in-place for each shape's detection
    for shape_name, shape_info in multi_shape_data.items():
        polygon = shape_info["final_poly"]
        gripping_pts = shape_info["gripping_points"]
        line_angle = shape_info["gripper_line_angle"]

        # Convert the polygon's exterior to an OpenCV contour
        ext_coords = np.array(polygon.exterior.coords)
        ext_img = du.map_world_to_image(ext_coords, homography_inv, offset_x, offset_y)
        shape_cnt = np.array(ext_img, dtype=int).reshape((-1, 1, 2))
        shape_area = cv2.contourArea(shape_cnt)

        # Attempt matching
        matched_comps = []
        for idx, comp in enumerate(composites):
            outer = comp[0]
            sim = cv2.matchShapes(shape_cnt, outer, cv2.CONTOURS_MATCH_I1, 0.0)
            parent_area = cv2.contourArea(outer)
            area_diff = abs(parent_area - shape_area)/shape_area if shape_area>0 else 1.0
            if sim < THRESHOLD and area_diff < AREA_TOLERANCE:
                matched_comps.append((comp, sim))

        logging.info(f"Shape '{shape_name}' => matched {len(matched_comps)} composite(s).")

        # Centroids
        centroids_img = []
        for comp, _ in matched_comps:
            M = cv2.moments(comp[0])
            if M['m00'] != 0:
                cx = M['m10']/M['m00']
                cy = M['m01']/M['m00']
                centroids_img.append((cx, cy))
        world_coords = du.map_image_to_world(centroids_img, homography, offset_x, offset_y)

        # Visualize
        start_t = time.perf_counter()
        visualize_detections(undistorted, matched_comps, world_coords,
                             polygon, gripping_pts,
                             homography_inv, offset_x, offset_y,
                             line_angle)
        end_t = time.perf_counter()
        logging.info(f"'{shape_name}' visualization took {end_t - start_t:.4f}s.")

    print("Static image processed with all shapes. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_live_detection_multi(calibration_data, offset_x, offset_y, multi_shape_data):
    """
    Like run_live_detection, but for multiple shapes. When user presses space,
    we do edges->composites once, then test each shape in 'multi_shape_data'.
    We use the 0.90 IoU threshold to reject partial matches.
    """
    try:
        camera_matrix = calibration_data['camera_matrix']
        dist_coeffs   = calibration_data['dist_coeffs']
        new_camera_mtx= calibration_data['new_camera_mtx']
        homography    = calibration_data['homography']
        homography_inv= np.linalg.inv(homography)
    except KeyError as e:
        logging.error(f"Missing calibration key: {e}")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX, BACKEND)
    if not cap.isOpened():
        logging.error(f"Cannot open webcam index {CAMERA_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,         DESIRED_FPS)

    aw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ah = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    afps = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f"Requested: {DESIRED_WIDTH}x{DESIRED_HEIGHT}@{DESIRED_FPS}")
    logging.info(f"Actual:    {int(aw)}x{int(ah)}@{int(afps)}")

    logging.info("Live detection started. 'q' to quit, 'space' to capture.")
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to grab frame.")
            break

        display_frame = cv2.resize(frame, (int(aw), int(ah)))
        cv2.putText(display_frame, "Press SPACE to capture, Q to quit",
                    (80, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Object Detection", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logging.info("Exiting live detection.")
            break
        elif key == ord(' '):
            # Process the captured frame
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_mtx)
            edges = get_edges(undistorted, 5, 40)
            if DEBUG_EDGES:
                cv2.imshow("Edges", edges)
            if SHOW_HIERARCHY_DEBUG:
                debug_vis = process_contours(undistorted, edges)
                cv2.imshow("Hierarchy Debug", debug_vis)

            composites = get_composite_contours(edges)
            logging.info(f"Found {len(composites)} composite(s).")

            # We'll overlay all shapes on this undistorted frame
            for shape_name, shape_info in multi_shape_data.items():
                polygon = shape_info["final_poly"]
                grips   = shape_info["gripping_points"]
                line_ang= shape_info["gripper_line_angle"]

                # Convert polygon to image contour
                ext = np.array(polygon.exterior.coords)
                ext_img = du.map_world_to_image(ext, homography_inv, offset_x, offset_y)
                shape_cnt = np.array(ext_img, dtype=int).reshape((-1, 1, 2))
                shape_area= cv2.contourArea(shape_cnt)

                # Check for matches
                matched_comps = []
                for idx, comp in enumerate(composites):
                    outer = comp[0]
                    sim = cv2.matchShapes(shape_cnt, outer, cv2.CONTOURS_MATCH_I1, 0.0)
                    parent_area = cv2.contourArea(outer)
                    area_diff = abs(parent_area - shape_area)/shape_area if shape_area>0 else 1.0
                    if sim < THRESHOLD and area_diff < AREA_TOLERANCE:
                        matched_comps.append((comp, sim))

                logging.info(f"Shape '{shape_name}' => {len(matched_comps)} matches.")

                # Get centroids
                centroids_img = []
                for comp, _ in matched_comps:
                    M = cv2.moments(comp[0])
                    if M['m00'] != 0:
                        cx = M['m10']/M['m00']
                        cy = M['m01']/M['m00']
                        centroids_img.append((cx, cy))
                world_coords = du.map_image_to_world(centroids_img, homography, offset_x, offset_y)

                # Visualize
                visualize_detections(undistorted, matched_comps, world_coords,
                                     polygon, grips,
                                     homography_inv, offset_x, offset_y,
                                     line_ang)

            logging.info("All shapes processed. Press space or q...")
            while True:
                k2 = cv2.waitKey(1) & 0xFF
                if k2 == ord('q'):
                    logging.info("Exiting live detection.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif k2 == ord(' '):
                    break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------
# MAIN
# ---------------------------
def main():
    logging.basicConfig(
        level=LOGGING_LEVEL,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 1) Load camera calibration
    try:
        calibration_data, offset_x, offset_y = du.load_calibration_data(CALIBRATION_FILE_PATH)
    except Exception as e:
        logging.error("Calibration data load failed: " + str(e))
        return

    # 2) Optimize multiple shapes from a folder of DXFs using min_max_distance
    multi_shape_data = optimize_all_dxfs_in_folder(DXF_FOLDER)
    if not multi_shape_data:
        logging.error(f"No shapes loaded/optimized from {DXF_FOLDER}. Exiting.")
        return

    # Optional scaling for each shape's polygon
    for shape_name in multi_shape_data:
        poly = multi_shape_data[shape_name]["final_poly"]
        multi_shape_data[shape_name]["final_poly"] = scale(poly, xfact=1.025, yfact=1.025, origin='centroid')

    # 3) Decide static vs. live
    if USE_STATIC_IMAGE:
        run_static_image_multi(TEST_IMAGE_PATH, calibration_data, offset_x, offset_y, multi_shape_data)
    else:
        run_live_detection_multi(calibration_data, offset_x, offset_y, multi_shape_data)


if __name__ == "__main__":
    main()
