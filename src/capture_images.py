import cv2
import os

def capture_images(
    calibration_save_dir='data/calibration_images',
    general_save_dir='data',
    num_calib_images=20,
    grid_size=(10,7),
    camera_index=0,
    desired_width=1920,
    desired_height=1080,
    desired_fps=30
):
    """
    Captures calibration images and general images using the specified camera.

    Parameters:
        calibration_save_dir (str): Directory to save calibration images.
        general_save_dir (str): Directory to save general images.
        num_calib_images (int): Number of calibration images to capture.
        grid_size (tuple): Chessboard grid size (columns, rows).
        camera_index (int): Index of the camera to use.
        desired_width (int): Desired frame width in pixels.
        desired_height (int): Desired frame height in pixels.
        desired_fps (int): Desired frames per second.

    Returns:
        None
    """
    if not os.path.exists(calibration_save_dir):
        os.makedirs(calibration_save_dir)
        print(f"Created directory '{calibration_save_dir}' for saving calibration images.")
    
    if not os.path.exists(general_save_dir):
        os.makedirs(general_save_dir)
        print(f"Created directory '{general_save_dir}' for saving general images.")

    # Initialize camera with a specified backend
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Cannot open camera with index {camera_index}.")
        return

    # Set desired resolution and frame rate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    cap.set(cv2.CAP_PROP_FPS, desired_fps)

    # Retrieve actual resolution and FPS
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Requested Resolution: {desired_width}x{desired_height} at {desired_fps} FPS")
    print(f"Actual Resolution: {int(actual_width)}x{int(actual_height)} at {int(actual_fps)} FPS")
    print("Instructions:")
    print(" - Press SPACEBAR to capture a calibration image when the chessboard is detected.")
    print(" - Press 'c' to capture a coordinate frame image.")
    print(" - Press 'x' to capture a test image.")
    print(" - Press 'q' to quit the image capture process at any time.")

    captured_calib = 0
    while captured_calib < num_calib_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCornersSB(
            gray, grid_size, 
            flags=None
        )

        display_frame = frame.copy()

        if ret_corners:
            # Refine corner locations for better accuracy
            criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.1)
            corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            ret_corners_refined, corners_refined = cv2.findChessboardCornersSB(gray, grid_size, flags=None)

            if ret_corners_refined:
                # Draw chessboard corners on the display frame
                cv2.drawChessboardCorners(display_frame, grid_size, corners_refined, ret_corners_refined)
                # Overlay capture information
                cv2.putText(display_frame, f"Calibration Image {captured_calib}/{num_calib_images}", 
                            (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        else:
            # Overlay warning text on the display frame
            cv2.putText(display_frame, "Chessboard not detected. Adjust the board for calibration.", 
                        (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

        cv2.imshow('Image Capture', display_frame)

        key = cv2.waitKey(1)
        
        if key != -1:
            # Convert the key code to its corresponding character and make it lowercase
            try:
                key_char = chr(key & 0xFF).lower()
            except:
                key_char = ''

            if key_char == ' ':
                if ret_corners:
                    calib_image_number = captured_calib
                    img_path = os.path.join(calibration_save_dir, f'calib_{calib_image_number:02d}.jpg')
                    cv2.imwrite(img_path, frame)  # Save the original frame without overlays
                    print(f"Saved calibration image: {img_path}")
                    captured_calib += 1
                else:
                    print("Chessboard not detected. Calibration image not saved.")
            
            elif key_char == 'c':
                coord_img_path = os.path.join(general_save_dir, 'coordinate_frame_image.jpg')
                cv2.imwrite(coord_img_path, frame)
                print(f"Saved coordinate frame image: {coord_img_path}")
            
            elif key_char == 'x':
                test_img_path = os.path.join(general_save_dir, 'test_image_x.jpg')
                cv2.imwrite(test_img_path, frame)
                print(f"Saved test image: {test_img_path}")
            
            elif key_char == 'q':
                print("Exiting image capture.")
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {captured_calib} out of {num_calib_images} calibration images.")

if __name__ == '__main__':
    # Set your desired parameters here
    SAVE_DIR = 'data/calibration_images'
    GENERAL_SAVE_DIR = "data"
    NUM_IMAGES = 20
    GRID_SIZE = (10, 7)          # (columns, rows)
    CAMERA_INDEX = 1             # Change to 0 if using the default camera
    DESIRED_WIDTH = 1920
    DESIRED_HEIGHT = 1080
    DESIRED_FPS = 30

    capture_images(
        calibration_save_dir=SAVE_DIR,
        general_save_dir=GENERAL_SAVE_DIR,
        num_calib_images=NUM_IMAGES, 
        grid_size=GRID_SIZE, 
        camera_index=CAMERA_INDEX,
        desired_width=DESIRED_WIDTH,
        desired_height=DESIRED_HEIGHT,
        desired_fps=DESIRED_FPS
    )

