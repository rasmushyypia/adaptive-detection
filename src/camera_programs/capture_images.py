import cv2
import os

def capture_images(
    calibration_save_dir: str = 'data/calibration_images',
    general_save_dir: str = 'data',
    num_calib_images: int = 20,
    grid_size: tuple = (13, 7),
    camera_index: int = 0,
    desired_width: int = 1920,
    desired_height: int = 1080,
    desired_fps: int = 30) -> None:
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
    """
    # Create necessary directories if they don't exist
    os.makedirs(calibration_save_dir, exist_ok=True)
    os.makedirs(general_save_dir, exist_ok=True)

    # Initialize the camera with a specified backend
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Cannot open camera with index {camera_index}.")
        return

    # Set desired resolution and frame rate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    cap.set(cv2.CAP_PROP_FPS, desired_fps)

    # Retrieve and display actual resolution and FPS
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Requested: {desired_width}x{desired_height} at {desired_fps} FPS")
    print(f"Actual: {int(actual_width)}x{int(actual_height)} at {int(actual_fps)} FPS")
    print("Instructions:")
    print(" - Press SPACEBAR to capture a calibration image (when chessboard is detected).")
    print(" - Press 'c' to capture a coordinate frame image.")
    print(" - Press 'x' to capture a test image.")
    print(" - Press 'q' to quit.")

    captured_calib = 0
    while captured_calib < num_calib_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(gray, grid_size)
        display_frame = frame.copy()

        if found:
            # Refine and draw chessboard corners for better accuracy
            criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.1)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display_frame, grid_size, corners, found)
            cv2.putText(
                display_frame, 
                f"Calibration Image {captured_calib}/{num_calib_images}", 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
            )
        else:
            cv2.putText(
                display_frame, 
                "Chessboard not detected. Adjust board for calibration.", 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA
            )

        cv2.imshow('Image Capture', display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if found:
                img_path = os.path.join(calibration_save_dir, f'calib_{captured_calib:02d}.jpg')
                cv2.imwrite(img_path, frame)  # Save the original frame without overlays
                print(f"Saved calibration image: {img_path}")
                captured_calib += 1
            else:
                print("Chessboard not detected. Calibration image not saved.")
        elif key == ord('c'):
            coord_img_path = os.path.join(general_save_dir, 'coordinate_frame_image2.jpg')
            cv2.imwrite(coord_img_path, frame)
            print(f"Saved coordinate frame image: {coord_img_path}")
        elif key == ord('x'):
            test_img_path = os.path.join(general_save_dir, 'test_image_x.jpg')
            cv2.imwrite(test_img_path, frame)
            print(f"Saved test image: {test_img_path}")
        elif key == ord('q'):
            print("Exiting image capture.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {captured_calib} out of {num_calib_images} calibration images.")

if __name__ == '__main__':
    SAVE_DIR = 'data/calibration_images'
    GENERAL_SAVE_DIR = 'data'
    NUM_IMAGES = 25
    GRID_SIZE = (13, 7)
    CAMERA_INDEX = 0
    DESIRED_WIDTH = 1920
    DESIRED_HEIGHT = 1080
    DESIRED_FPS = 20

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
