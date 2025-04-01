import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, Image as PILImage
import cv2
import os
import glob
import pickle
from utils.camera_utils import (
    get_frame_with_grid,
    calibrate_camera_single_image,
    calibrate_camera,
    get_calibrated_image,
    generate_object_points,  # Make sure generate_object_points is accessible
)

class CalibrationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Camera Calibration GUI with Homography")
        self.geometry("1240x600")
        self.resizable(False, False)

        # --- Default Parameters ---
        self.camera_index = 0
        self.desired_width = 1920
        self.desired_height = 1080

        # Display area inside GUI
        self.display_width = 900
        self.display_height = 506

        # Default chessboard grid for real-time detection
        self.grid_cols = 10
        self.grid_rows = 7

        # Default calibration parameters
        self.default_calib_grid = (10, 7)   # For single/multi calibrations
        self.default_calib_square = 20.0
        self.default_mapping_grid = (13, 7)
        self.default_mapping_square = 40.0

        # Feed states
        self.feed_running = False
        self.cap = None

        # Directories
        self.calibration_dir = "data/calibration_images"
        os.makedirs(self.calibration_dir, exist_ok=True)
        self.coord_frame_path = os.path.join(self.calibration_dir, "coord_frame_00.jpg")
        self.test_images_dir = "data/test_images"
        os.makedirs(self.test_images_dir, exist_ok=True)

        self.create_widgets()
        self.update_feed()

        # Bind hotkeys for capturing images
        self.bind("<x>", self._on_key_capture_calib)
        self.bind("<X>", self._on_key_capture_calib)
        self.bind("<c>", self._on_key_capture_coord)
        self.bind("<C>", self._on_key_capture_coord)
        self.bind("<t>", self._on_key_capture_test)
        self.bind("<T>", self._on_key_capture_test)

    # ------------------------------------------------------------
    # GUI Layout
    # ------------------------------------------------------------
    def create_widgets(self):
        """Create and arrange all widgets in the main window."""
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        style = ttk.Style(self)
        style.configure("BoldLabelframe.TLabelframe.Label", font=("Helvetica", 10, "bold"))

        # ==========================
        # Left Panel
        # ==========================
        left_panel = ttk.Frame(self)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # ------------------------------------------------------------
        # 1) Capture Image Parameters
        # ------------------------------------------------------------
        cam_frame = ttk.LabelFrame(
            left_panel, text="Capture Image Parameters", style="BoldLabelframe.TLabelframe"
        )
        cam_frame.pack(fill="x", pady=(0, 6))

        # -- Camera Index --
        row_cam_idx = ttk.Frame(cam_frame)
        row_cam_idx.pack(anchor="w", pady=3)
        ttk.Label(row_cam_idx, text="Camera Index:").pack(side="left", padx=2)
        self.camera_index_var = tk.IntVar(value=self.camera_index)
        ttk.Entry(row_cam_idx, textvariable=self.camera_index_var, width=3).pack(side="left", padx=2)

        # -- Capture Resolution --
        row_res = ttk.Frame(cam_frame)
        row_res.pack(anchor="w", pady=3)
        ttk.Label(row_res, text="Capture Resolution:").pack(side="left", padx=2)
        self.width_var = tk.IntVar(value=self.desired_width)
        ttk.Entry(row_res, textvariable=self.width_var, width=5).pack(side="left", padx=2)
        ttk.Label(row_res, text="x").pack(side="left", padx=2)
        self.height_var = tk.IntVar(value=self.desired_height)
        ttk.Entry(row_res, textvariable=self.height_var, width=5).pack(side="left", padx=2)

        # -- Chessboard Grid (Preview) --
        row_chess = ttk.Frame(cam_frame)
        row_chess.pack(anchor="w", pady=3)
        ttk.Label(row_chess, text="Chessboard Grid:").pack(side="left", padx=2)
        self.chessboard_cols_var = tk.IntVar(value=self.grid_cols)
        ttk.Entry(row_chess, textvariable=self.chessboard_cols_var, width=3).pack(side="left", padx=2)
        ttk.Label(row_chess, text="x").pack(side="left", padx=2)
        self.chessboard_rows_var = tk.IntVar(value=self.grid_rows)
        ttk.Entry(row_chess, textvariable=self.chessboard_rows_var, width=3).pack(side="left", padx=2)

        # -- Start Camera Button --
        row_cam_btn = ttk.Frame(cam_frame)
        row_cam_btn.pack(anchor="w", padx=3, pady=5)
        self.init_cam_btn = ttk.Button(row_cam_btn, text="Start Camera", command=self.start_camera)
        self.init_cam_btn.pack()

        # ------------------------------------------------------------
        # 2) Calibration & Mapping Parameters
        # ------------------------------------------------------------
        calib_frame = ttk.LabelFrame(
            left_panel, text="Calibration & Mapping Parameters", style="BoldLabelframe.TLabelframe"
        )
        calib_frame.pack(fill="x", pady=(0, 6))

        # -- Calibration Grid --
        row_calib_grid = ttk.Frame(calib_frame)
        row_calib_grid.pack(anchor="w", pady=3)
        ttk.Label(row_calib_grid, text="Calibration Grid:").pack(side="left", padx=2)
        self.calib_grid_cols_var = tk.IntVar(value=self.default_calib_grid[0])
        ttk.Entry(row_calib_grid, textvariable=self.calib_grid_cols_var, width=3).pack(side="left", padx=2)
        ttk.Label(row_calib_grid, text="x").pack(side="left", padx=2)
        self.calib_grid_rows_var = tk.IntVar(value=self.default_calib_grid[1])
        ttk.Entry(row_calib_grid, textvariable=self.calib_grid_rows_var, width=3).pack(side="left", padx=2)

        # -- Calibration Square Size --
        row_calib_sq = ttk.Frame(calib_frame)
        row_calib_sq.pack(anchor="w", pady=3)
        ttk.Label(row_calib_sq, text="Calib Square Size (mm):").pack(side="left", padx=2)
        self.calib_square_size_var = tk.DoubleVar(value=self.default_calib_square)
        ttk.Entry(row_calib_sq, textvariable=self.calib_square_size_var, width=5).pack(side="left", padx=2)

        # -- Mapping Grid --
        row_map_grid = ttk.Frame(calib_frame)
        row_map_grid.pack(anchor="w", pady=3)
        ttk.Label(row_map_grid, text="Mapping Grid:").pack(side="left", padx=2)
        self.mapping_grid_cols_var = tk.IntVar(value=self.default_mapping_grid[0])
        ttk.Entry(row_map_grid, textvariable=self.mapping_grid_cols_var, width=3).pack(side="left", padx=2)
        ttk.Label(row_map_grid, text="x").pack(side="left", padx=2)
        self.mapping_grid_rows_var = tk.IntVar(value=self.default_mapping_grid[1])
        ttk.Entry(row_map_grid, textvariable=self.mapping_grid_rows_var, width=3).pack(side="left", padx=2)

        # -- Mapping Square Size --
        row_map_sq = ttk.Frame(calib_frame)
        row_map_sq.pack(anchor="w", pady=3)
        ttk.Label(row_map_sq, text="Mapping Square Size (mm):").pack(side="left", padx=2)
        self.mapping_square_size_var = tk.DoubleVar(value=self.default_mapping_square)
        ttk.Entry(row_map_sq, textvariable=self.mapping_square_size_var, width=5).pack(side="left", padx=2)

        # -- Checkboxes --
        row_checks = ttk.Frame(calib_frame)
        row_checks.pack(anchor="w", padx=3, pady=6)
        self.flip_mapping_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row_checks, text="Flip Mapping Origin", variable=self.flip_mapping_var).pack(anchor="w")
        self.save_calib_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row_checks, text="Save Calibration Data", variable=self.save_calib_var).pack(anchor="w")
        self.visualize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_checks, text="Visualize Calibration", variable=self.visualize_var).pack(anchor="w")

        # -- Buttons for Single & Multi Calibration --
        row_btn_single = ttk.Frame(calib_frame)
        row_btn_single.pack(anchor="w", padx=3, pady=0)
        self.single_calib_btn = ttk.Button(
            row_btn_single, text="Single-Image Calibration", command=self.run_single_image_calibration
        )
        self.single_calib_btn.pack()

        row_btn_multi = ttk.Frame(calib_frame)
        row_btn_multi.pack(anchor="w", padx=3, pady=3)
        self.multi_calib_btn = ttk.Button(
            row_btn_multi, text="Multi-Image Calibration ", command=self.run_multi_image_calibration
        )
        self.multi_calib_btn.pack()

        # ------------------------------------------------------------
        # 3) Folder Visualization
        # ------------------------------------------------------------
        folder_frame = ttk.LabelFrame(
            left_panel, text="Calibration Image Folder", style="BoldLabelframe.TLabelframe"
        )
        folder_frame.pack(fill="x", pady=(0, 0))

        list_frame = ttk.Frame(folder_frame)
        list_frame.pack(anchor="w", pady=5)

        self.image_listbox = tk.Listbox(list_frame, width=22, height=8)
        self.image_listbox.pack(side="left", padx=(5, 0), pady=5)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.image_listbox.yview)
        scrollbar.pack(side="left", fill="y", padx=(0, 5), pady=5)
        self.image_listbox.config(yscrollcommand=scrollbar.set)
        self.image_listbox.bind("<Double-Button-1>", lambda event: self.open_selected_image())

        btn_frame = ttk.Frame(list_frame)
        btn_frame.pack(side="left", padx=10, pady=5)

        self.refresh_btn = ttk.Button(btn_frame, text="Refresh Folder", command=self.refresh_image_list)
        self.refresh_btn.pack(pady=4, fill="x")
        self.open_btn = ttk.Button(btn_frame, text="Open Image", command=self.open_selected_image)
        self.open_btn.pack(pady=4, fill="x")
        self.delete_btn = ttk.Button(btn_frame, text="Delete Image", command=self.delete_selected_image)
        self.delete_btn.pack(pady=4, fill="x")
        self.make_main_btn = ttk.Button(btn_frame, text="Set as Reference Frame", command=self.make_main_coord_image)
        self.make_main_btn.pack(pady=4, fill="x")

        # ==========================
        # Right Panel: Live Feed
        # ==========================
        right_panel = ttk.LabelFrame(
            self, text="Live Camera Feed & Capture", style="BoldLabelframe.TLabelframe"
        )
        right_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=(10, 10))

        # Placeholder
        placeholder = PILImage.new("RGB", (self.display_width, self.display_height), color=(128, 128, 128))
        self.placeholder_imgtk = ImageTk.PhotoImage(placeholder)
        self.video_label = ttk.Label(right_panel, image=self.placeholder_imgtk)
        self.video_label.pack(padx=5, pady=5)

        cap_button_frame = ttk.Frame(right_panel)
        cap_button_frame.pack(pady=5)
        self.calib_img_btn = ttk.Button(
            cap_button_frame, text="Capture Calibration Image (X)", command=self.capture_calibration_image
        )
        self.calib_img_btn.pack(side="left", padx=5)
        self.coord_img_btn = ttk.Button(
            cap_button_frame, text="Capture Coordinate Frame (C)", command=self.capture_coordinate_image
        )
        self.coord_img_btn.pack(side="left", padx=5)
        self.test_img_btn = ttk.Button(
            cap_button_frame, text="Capture Test Image (T)", command=self.capture_test_image
        )
        self.test_img_btn.pack(side="left", padx=5)

        # Initial listing of images
        self.refresh_image_list()

    # ------------------------------------------------------------
    # Hotkey Handlers
    # ------------------------------------------------------------
    def _on_key_capture_calib(self, event):
        self.capture_calibration_image()

    def _on_key_capture_coord(self, event):
        self.capture_coordinate_image()

    def _on_key_capture_test(self, event):
        self.capture_test_image()

    # ------------------------------------------------------------
    # Camera / Feed
    # ------------------------------------------------------------
    def start_camera(self):
        """Initialize or re-initialize the camera and update parameters."""
        self.feed_running = False
        if self.cap is not None:
            self.cap.release()

        self.camera_index = self.camera_index_var.get()
        self.desired_width = self.width_var.get()
        self.desired_height = self.height_var.get()

        self.current_grid_cols = self.chessboard_cols_var.get()
        self.current_grid_rows = self.chessboard_rows_var.get()

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Failed to open camera index {self.camera_index}")
            return

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_height)

        self.feed_running = True
        self.init_cam_btn.config(text="Re-initialize Camera")
        self.video_label.config(text="Camera feed started.")

    def update_feed(self):
        """Continuously update the live video feed."""
        if self.feed_running and self.cap is not None and self.cap.isOpened():
            grid_size = (self.current_grid_cols, self.current_grid_rows)
            frame, grid_found = get_frame_with_grid(self.cap, grid_size)
            if frame is not None:
                display_frame = self._aspect_ratio_resize(frame, self.display_width, self.display_height)
                display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(display_frame_rgb)
                imgtk = ImageTk.PhotoImage(image=pil_img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                self.current_grid_found = grid_found

        self.after(30, self.update_feed)

    def _aspect_ratio_resize(self, frame, max_width, max_height):
        h, w = frame.shape[:2]
        scale = min(max_width / w, max_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------
    def capture_calibration_image(self):
        if not self.feed_running or self.cap is None:
            messagebox.showwarning("Warning", "Camera not started.")
            return

        grid_size = (self.current_grid_cols, self.current_grid_rows)
        frame_with_grid, found = get_frame_with_grid(self.cap, grid_size)
        if frame_with_grid is None:
            messagebox.showwarning("Warning", "Failed to grab frame from the camera.")
            return
        if not found:
            messagebox.showwarning(
                "Warning",
                "No chessboard detected in this frame. Adjust the board and try again."
            )
            return

        # Capture a clean frame for saving
        ret, clean_frame = self.cap.read()
        if not ret:
            messagebox.showwarning("Warning", "Failed to grab a clean frame from the camera.")
            return

        existing_files = glob.glob(os.path.join(self.calibration_dir, "calib_*.jpg"))
        next_index = len(existing_files)
        save_path = os.path.join(self.calibration_dir, f"calib_{next_index:02d}.jpg")
        cv2.imwrite(save_path, clean_frame)
        print("Image Captured", f"Saved {save_path}.")
        self.refresh_image_list()

    def capture_coordinate_image(self):
        if not self.feed_running or self.cap is None:
            messagebox.showwarning("Warning", "Camera not started.")
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showwarning("Warning", "Failed to grab frame from the camera.")
            return

        existing_files = glob.glob(os.path.join(self.calibration_dir, "coord_frame_*.jpg"))
        next_index = len(existing_files)
        save_path = os.path.join(self.calibration_dir, f"coord_frame_{next_index:02d}.jpg")
        cv2.imwrite(save_path, frame)
        print("Coordinate Frame Captured", f"Saved {save_path}.")
        self.refresh_image_list()

    def capture_test_image(self):
        if not self.feed_running or self.cap is None:
            messagebox.showwarning("Warning", "Camera not started.")
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showwarning("Warning", "Failed to grab frame from the camera.")
            return

        os.makedirs(self.test_images_dir, exist_ok=True)
        existing_files = glob.glob(os.path.join(self.test_images_dir, "test_image_*.jpg"))
        next_index = len(existing_files)
        save_path = os.path.join(self.test_images_dir, f"test_image_{next_index:02d}.jpg")
        cv2.imwrite(save_path, frame)
        print("Test Image Captured", f"Saved {save_path}.")

    # ------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------
    def run_single_image_calibration(self):
        """
        Single-image calibration + compute homography (using the same coordinate frame image).
        """
        default_coord_path = os.path.join(self.calibration_dir, "coord_frame_00.jpg")
        if not os.path.exists(default_coord_path):
            messagebox.showerror("Error", "No coordinate frame image found. Please capture one first.")
            return

        grid_size = (self.calib_grid_cols_var.get(), self.calib_grid_rows_var.get())
        square_size = self.calib_square_size_var.get()
        flip = self.flip_mapping_var.get()
        visualize = self.visualize_var.get()

        # Intrinsic calibration from single image
        cam_mtx, dist_coeffs, new_cam_mtx = calibrate_camera_single_image(
            self.coord_frame_path, grid_size, square_size, flip_origin=flip, visualize=False
        )
        if cam_mtx is None:
            messagebox.showerror(
                "Calibration Failed",
                "Chessboard corners not detected or calibration failed. Check console."
            )
            return

        # --- Optionally compute homography with the "mapping" grid ---
        H = None
        if self.save_calib_var.get():
            # 1. Load & undistort the coordinate frame
            coord_img = cv2.imread(self.coord_frame_path)
            if coord_img is not None:
                undistorted_img = cv2.undistort(coord_img, cam_mtx, dist_coeffs, None, new_cam_mtx)

                # 2. Detect corners with the *mapping* grid
                map_grid = (self.mapping_grid_cols_var.get(), self.mapping_grid_rows_var.get())
                gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCornersSB(
                    gray, map_grid, flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
                )
                if ret:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                    # 3. Generate object points in 2D for homography
                    objp_3D = generate_object_points(
                        map_grid, self.mapping_square_size_var.get(), flip_origin=flip
                    )
                    objp_2D = objp_3D[:, :2].astype("float32")
                    imgpts_2D = corners_refined.reshape(-1, 2).astype("float32")

                    # 4. Compute homography
                    H, _ = cv2.findHomography(imgpts_2D, objp_2D, 0)
                    if H is None:
                        print("Homography computation failed in single-image calibration.")
                else:
                    print("Mapping chessboard not found; homography not computed.")

            # --- Save everything including homography ---
            cal_data = {
                'camera_matrix': cam_mtx,
                'dist_coeffs': dist_coeffs,
                'new_camera_mtx': new_cam_mtx,
                'flip_mapping_origin': flip,
                'homography': H  # could be None if detection fails
            }
            os.makedirs('data', exist_ok=True)
            with open('data/calibration_data_single.pkl', 'wb') as f:
                pickle.dump(cal_data, f)
            print("Single-image calibration + homography saved to data/calibration_data_single.pkl.")

        # If user wants to visualize, show undistorted w/ grid overlay
        if visualize:
            self.feed_running = False
            mapping_grid = (self.mapping_grid_cols_var.get(), self.mapping_grid_rows_var.get())
            mapping_square = self.mapping_square_size_var.get()
            cal_img = get_calibrated_image(
                self.coord_frame_path, cam_mtx, dist_coeffs, new_cam_mtx,
                mapping_grid, mapping_square, flip
            )
            if cal_img is not None:
                display_frame = self._aspect_ratio_resize(cal_img, self.display_width, self.display_height)
                display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(display_frame_rgb)
                imgtk = ImageTk.PhotoImage(image=pil_img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            else:
                messagebox.showwarning(
                    "Warning", "Chessboard not found in coordinate frame image for visualization."
                )

        messagebox.showinfo("Calibration Complete", "Single-image calibration complete.")

    def run_multi_image_calibration(self):
        """
        Multi-image calibration + compute homography (using coordinate frame image).
        """
        grid_size = (self.calib_grid_cols_var.get(), self.calib_grid_rows_var.get())
        square_size = self.calib_square_size_var.get()
        flip = self.flip_mapping_var.get()
        visualize = self.visualize_var.get()

        # Intrinsic calibration from multiple images
        cam_mtx, dist_coeffs, new_cam_mtx = calibrate_camera(
            self.calibration_dir, grid_size, square_size, flip_origin=flip, visualize=True
        )
        if cam_mtx is None:
            messagebox.showerror(
                "Calibration Failed",
                "No valid chessboard corners found or calibration error."
            )
            return

        # --- Compute Homography if user wants to save data ---
        H = None
        if self.save_calib_var.get():
            coord_img = cv2.imread(self.coord_frame_path)
            if coord_img is not None:
                undistorted_img = cv2.undistort(coord_img, cam_mtx, dist_coeffs, None, new_cam_mtx)

                map_grid = (self.mapping_grid_cols_var.get(), self.mapping_grid_rows_var.get())
                gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCornersSB(
                    gray, map_grid, flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
                )
                if ret:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                    objp_3D = generate_object_points(
                        map_grid, self.mapping_square_size_var.get(), flip_origin=flip
                    )
                    objp_2D = objp_3D[:, :2].astype("float32")
                    imgpts_2D = corners_refined.reshape(-1, 2).astype("float32")

                    H, _ = cv2.findHomography(imgpts_2D, objp_2D, 0)
                    if H is None:
                        print("Homography computation failed in multi-image calibration.")
                else:
                    print("Mapping chessboard not found; homography not computed.")
            else:
                print("No coordinate frame image loaded; cannot compute homography.")

            # Save everything
            cal_data = {
                'camera_matrix': cam_mtx,
                'dist_coeffs': dist_coeffs,
                'new_camera_mtx': new_cam_mtx,
                'flip_mapping_origin': flip,
                'homography': H
            }
            os.makedirs('data', exist_ok=True)
            with open('data/calibration_data_multi.pkl', 'wb') as f:
                pickle.dump(cal_data, f)
            print("Multi-image calibration + homography saved to data/calibration_data_multi.pkl.")

        # Visualization
        messagebox.showinfo(
            "Calibration Complete",
            "Multi-image calibration completed successfully. Check console for details."
        )
        if visualize:
            self.feed_running = False
            default_coord_path = os.path.join(self.calibration_dir, "coord_frame_00.jpg")
            if not os.path.exists(default_coord_path):
                messagebox.showerror("Error", "No coordinate frame image found.")
                return

            mapping_grid = (self.mapping_grid_cols_var.get(), self.mapping_grid_rows_var.get())
            mapping_square = self.mapping_square_size_var.get()
            cal_img = get_calibrated_image(
                self.coord_frame_path, cam_mtx, dist_coeffs, new_cam_mtx,
                mapping_grid, mapping_square, flip
            )
            if cal_img is not None:
                display_frame = self._aspect_ratio_resize(cal_img, self.display_width, self.display_height)
                display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(display_frame_rgb)
                imgtk = ImageTk.PhotoImage(image=pil_img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            else:
                messagebox.showwarning(
                    "Warning",
                    "Chessboard not found in coordinate frame image for visualization."
                )

    # ------------------------------------------------------------
    # Folder & Image Management
    # ------------------------------------------------------------
    def refresh_image_list(self):
        self.image_listbox.delete(0, tk.END)
        calib_pattern = os.path.join(self.calibration_dir, "calib_*.jpg")
        calib_files = sorted(glob.glob(calib_pattern))
        for f in calib_files:
            self.image_listbox.insert(tk.END, os.path.basename(f))

        coord_pattern = os.path.join(self.calibration_dir, "coord_frame_*.jpg")
        coord_files = sorted(glob.glob(coord_pattern))
        for f in coord_files:
            self.image_listbox.insert(tk.END, os.path.basename(f))

    def open_selected_image(self):
        selection = self.image_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "No image selected.")
            return
        filename = self.image_listbox.get(selection[0])
        filepath = os.path.join(self.calibration_dir, filename)
        if self.cap is not None:
            self.cap.release()
        self.feed_running = False

        img = cv2.imread(filepath)
        if img is None:
            messagebox.showerror("Error", f"Failed to load image {filepath}.")
            return

        display_frame = self._aspect_ratio_resize(img, self.display_width, self.display_height)
        img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def delete_selected_image(self):
        selection = self.image_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "No image selected.")
            return
        filename = self.image_listbox.get(selection[0])
        filepath = os.path.join(self.calibration_dir, filename)
        confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {filename}?")
        if confirm:
            try:
                os.remove(filepath)
                self.refresh_image_list()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete image: {e}")

    def make_main_coord_image(self):
        selection = self.image_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "No image selected.")
            return

        filename = self.image_listbox.get(selection[0])
        if not filename.startswith("coord_frame_"):
            messagebox.showwarning("Warning", "Selected image is not a coordinate frame image.")
            return

        if filename == "coord_frame_00.jpg":
            messagebox.showinfo("Info", "Selected image is already the main coordinate frame image.")
            return

        selected_path = os.path.join(self.calibration_dir, filename)
        main_path = os.path.join(self.calibration_dir, "coord_frame_00.jpg")
        temp_path = os.path.join(self.calibration_dir, "temp_coord_swap.jpg")

        try:
            if os.path.exists(main_path):
                os.rename(main_path, temp_path)
            os.rename(selected_path, main_path)
            if os.path.exists(temp_path):
                os.rename(temp_path, selected_path)
            messagebox.showinfo("Success", f"'{filename}' is now the main coordinate frame image.")
        except Exception as e:
            messagebox.showerror("Error", f"Error swapping files: {e}")

        self.refresh_image_list()

if __name__ == "__main__":
    app = CalibrationGUI()
    app.mainloop()
