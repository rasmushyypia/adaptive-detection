import cv2
import numpy as np
import os
import glob

def get_frame_with_grid(cap, grid_size):
    """
    Capture a frame from the camera (cap), detect the chessboard grid, draw it on the frame,
    and return the processed frame along with a boolean flag indicating whether the grid was detected.
    """
    ret, frame = cap.read()
    if not ret:
        return None, False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, grid_size)
    display_frame = frame.copy()
    
    if found:
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.1)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(display_frame, grid_size, corners, found)
        return display_frame, True
    else:
        cv2.putText(display_frame, "Chessboard not detected. Adjust board for calibration.",
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.4, (0, 0, 255), 2, cv2.LINE_AA)
        return display_frame, False

def generate_object_points(grid_size, square_size, flip_origin=False):
    """
    Generates 3D object points for a chessboard.
    By default, (0,0,0) corresponds to the top-left corner.
    If flip_origin is True, the bottom-right becomes (0,0,0).
    """
    cols, rows = grid_size
    objp = np.zeros((cols * rows, 3), np.float32)
    for r in range(rows):
        for c in range(cols):
            objp[r * cols + c, 0] = c * square_size
            objp[r * cols + c, 1] = r * square_size

    if flip_origin:
        max_x = (cols - 1) * square_size
        max_y = (rows - 1) * square_size
        objp[:, 0] = max_x - objp[:, 0]
        objp[:, 1] = max_y - objp[:, 1]
    return objp

def calibrate_camera_single_image(image_path, grid_size, square_size, flip_origin=False, visualize=False):
    """
    Calibrates camera intrinsics using a single chessboard image.
    Note: Using one image is not ideal for robust calibration, but it serves as a quick test.
    
    Parameters:
      image_path: Path to the calibration image.
      grid_size: Chessboard grid dimensions (columns, rows) for detection.
      square_size: Size of each square (in millimeters).
      flip_origin: Boolean flag; if True, the origin is flipped.
      visualize: Boolean flag to display the calibration process.
    
    Returns:
      cam_mtx, dist_coeffs, new_cam_mtx (or None if calibration failed).
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load the image:", image_path)
        return None, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCornersSB(gray, grid_size, flags=cv2.CALIB_CB_EXHAUSTIVE+cv2.CALIB_CB_ACCURACY)
    if not ret:
        print("Chessboard corners not found in", image_path)
        return None, None, None

    criteria = (cv2.TermCriteria_EPS+cv2.TermCriteria_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Use the passed flip_origin parameter here.
    objp = generate_object_points(grid_size, square_size, flip_origin=flip_origin)
    objpoints = [objp]
    imgpoints = [corners_refined]
    img_size = gray.shape[::-1]

    ret, cam_mtx, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    if not ret:
        print("Calibration failed for the image.")
        return None, None, None

    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cam_mtx, dist_coeffs)
        total_error += cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    print("Mean Reprojection Error: {:.6f}".format(total_error / len(objpoints)))

    new_cam_mtx, _ = cv2.getOptimalNewCameraMatrix(cam_mtx, dist_coeffs, img_size, 0, img_size)
    return cam_mtx, dist_coeffs, new_cam_mtx

def calibrate_camera(images_dir, grid_size, square_size, flip_origin=False, visualize=False):
    """
    Calibrates camera intrinsics using multiple chessboard images.
    Note: Flipping is not applied for calibration images unless specified.
    
    Parameters:
      images_dir: Directory containing calibration images.
      grid_size: Chessboard grid dimensions (columns, rows) for detection.
      square_size: Size of each square (in millimeters).
      flip_origin: Boolean flag; if True, the origin is flipped.
      visualize: Boolean flag; if True, print progress messages (but do not display each image).
    
    Returns:
      cam_mtx, dist_coeffs, new_cam_mtx (or None if calibration failed).
    """
    objp_template = generate_object_points(grid_size, square_size, flip_origin=flip_origin)
    objpoints, imgpoints, img_size = [], [], None
    exts = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(images_dir, f'calib_*.{ext}')))
    if not images:
        print("No calibration images found.")
        return None, None, None

    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print("Cannot load", fname)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCornersSB(
            gray, grid_size, flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp_template)
            imgpoints.append(corners_refined)
            if img_size is None:
                img_size = gray.shape[::-1]
            if visualize:
                print(f"Processed {fname}: Chessboard detected.")
        else:
            print("Chessboard not found in", fname)
    if visualize:
        print("Finished processing calibration images.")
    if not objpoints:
        print("Calibration failed: No valid corners found.")
        return None, None, None

    ret, cam_mtx, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cam_mtx, dist_coeffs)
        total_error += cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    print("Mean Reprojection Error: {:.6f}".format(total_error / len(objpoints)))
    new_cam_mtx, _ = cv2.getOptimalNewCameraMatrix(cam_mtx, dist_coeffs, img_size, 0, img_size)
    return cam_mtx, dist_coeffs, new_cam_mtx


def draw_axes(img, rvec, tvec, cam_mtx, dist_coeffs, axis_length):
    """
    Overlays axes on the image: origin, x-axis (red) and y-axis (green).
    """
    axis_pts = np.float32([[0, 0, 0],
                           [axis_length, 0, 0],
                           [0, axis_length, 0]])
    imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, cam_mtx, dist_coeffs)
    origin = tuple(imgpts[0].ravel().astype(int))
    x_axis = tuple(imgpts[1].ravel().astype(int))
    y_axis = tuple(imgpts[2].ravel().astype(int))
    cv2.line(img, origin, x_axis, (0, 0, 255), 3)
    cv2.line(img, origin, y_axis, (0, 255, 0), 3)


def get_calibrated_image(coord_frame_img_path, cam_mtx, dist_coeffs, new_cam_mtx, 
                         mapping_grid_size, mapping_square_size, flip_mapping_origin):
    """
    Loads the coordinate frame image, undistorts it, detects chessboard corners using the mapping grid,
    computes pose (via solvePnP), draws the axes on the undistorted image, and returns the final image.
    """
    img = cv2.imread(coord_frame_img_path)
    if img is None:
        print("Failed to load coordinate frame image.")
        return None
    undistorted_img = cv2.undistort(img, cam_mtx, dist_coeffs, None, new_cam_mtx)
    gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCornersSB(
        gray, mapping_grid_size, flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
    if not ret:
        print("Chessboard corners not found in coordinate frame image.")
        return undistorted_img
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    ret_pnp, rvec, tvec = cv2.solvePnP(
        generate_object_points(mapping_grid_size, mapping_square_size, flip_origin=flip_mapping_origin),
        corners_refined, cam_mtx, dist_coeffs)
    if not ret_pnp:
        print("Pose estimation failed.")
        return undistorted_img
    axis_length = mapping_square_size * 6
    draw_axes(undistorted_img, rvec, tvec, cam_mtx, dist_coeffs, axis_length)
    return undistorted_img