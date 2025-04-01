import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# --- Utility functions from your original code ---

def ensure_odd(x):
    """Ensure the value is odd (if even, add 1)."""
    return x if x % 2 == 1 else x + 1

def apply_edge_pipeline(
    image,
    canny_t1,
    canny_t2,
    l2gradient,
    blur_mode,
    blur_kernel,
    de_enable,
    de_kernel,
    dilate_iter,
    erode_iter,
    post_blur_mode,
    post_blur_kernel
):
    """
    Applies a chain of operations to 'image':
      1) ROI masking and initial blur (none, Gaussian, or median)
      2) Canny edge detection
      3) (Optional) dilate then erode operations
      4) (Optional) post-processing blur (Gaussian or median)
    Returns the final edges image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray, dtype=np.uint8)
    # Hard-coded ROI; adjust as needed
    cv2.rectangle(mask, (26, 0), (1865, 1042), 255, thickness=-1)
    cropped_gray = cv2.bitwise_and(gray, gray, mask=mask)

    ksize = 1 if blur_kernel == 0 else ensure_odd(blur_kernel)
    if blur_mode == 0:
        processed = cropped_gray
    elif blur_mode == 1:
        processed = cv2.GaussianBlur(cropped_gray, (ksize, ksize), 0)
    elif blur_mode == 2:
        processed = cv2.medianBlur(cropped_gray, ksize)
    else:
        processed = cropped_gray

    edges = cv2.Canny(processed, canny_t1, canny_t2, L2gradient=bool(l2gradient))

    if de_enable == 1:
        kernel_size = ensure_odd(de_kernel)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
        edges = cv2.erode(edges, kernel, iterations=erode_iter)
    
    if post_blur_mode != 0:
        pb_ksize = 1 if post_blur_kernel == 0 else ensure_odd(post_blur_kernel)
        if post_blur_mode == 1:
            edges = cv2.GaussianBlur(edges, (pb_ksize, pb_ksize), 0)
        elif post_blur_mode == 2:
            edges = cv2.medianBlur(edges, pb_ksize)

    return edges

def resize_to_resolution(image, target_width, target_height):
    """Resize image to fit within target resolution while preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

# --- Tkinter GUI Application using Resolution-based Scaling ---

class EdgeParameterTuning(tk.Tk):
    def __init__(self, image_path, target_resolution=(800, 600)):
        super().__init__()
        self.title("Edge Parameter Tuning")
        self.geometry("1420x700")
        
        # Load image (BGR) for processing and resize to target resolution
        self.orig_image = cv2.imread(image_path)
        if self.orig_image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        self.orig_image = resize_to_resolution(self.orig_image, *target_resolution)
        
        # Create main containers: one for settings (left), one for preview (right)
        self.settings_frame = ttk.Frame(self)
        self.settings_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        self.preview_frame = ttk.Frame(self)
        self.preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create the preview Label
        self.preview_label = ttk.Label(self.preview_frame)
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        self.create_settings()
        
        # Begin auto-update loop for preview
        self.update_preview()

    def create_settings(self):
        """Creates grouped settings with labels and scales."""
        # --- Edge Detection Settings ---
        edge_frame = ttk.LabelFrame(self.settings_frame, text="Edge Detection Settings", padding=5)
        edge_frame.pack(fill=tk.X, pady=5)
        
        self.edge_t1 = self.create_scale(edge_frame, "Canny Threshold 1", 0, 500, 30)
        self.edge_t2 = self.create_scale(edge_frame, "Canny Threshold 2", 0, 500, 50)
        self.l2_grad = self.create_scale(edge_frame, "Use L2 Gradient (0/1)", 0, 1, 1, orient=tk.HORIZONTAL)
        
        # --- Pre-Processing Blur ---
        pre_blur_frame = ttk.LabelFrame(self.settings_frame, text="Pre-Processing Blur", padding=5)
        pre_blur_frame.pack(fill=tk.X, pady=5)
        
        self.blur_mode = self.create_scale(pre_blur_frame, "Blur Mode (0:None,1:Gauss,2:Median)", 0, 2, 2)
        self.blur_kernel = self.create_scale(pre_blur_frame, "Blur Kernel Size", 0, 31, 7)
        
        # --- Morphological Operations ---
        morph_frame = ttk.LabelFrame(self.settings_frame, text="Morphological Operations", padding=5)
        morph_frame.pack(fill=tk.X, pady=5)
        
        self.de_enable = self.create_scale(morph_frame, "Dilate/Erode Enable (0/1)", 0, 1, 1)
        self.de_kernel = self.create_scale(morph_frame, "Morph Kernel Size", 0, 31, 3)
        self.dilate_iter = self.create_scale(morph_frame, "Dilate Iterations", 0, 10, 1)
        self.erode_iter = self.create_scale(morph_frame, "Erode Iterations", 0, 10, 0)
        
        # --- Post-Processing Blur ---
        post_blur_frame = ttk.LabelFrame(self.settings_frame, text="Post-Processing Blur", padding=5)
        post_blur_frame.pack(fill=tk.X, pady=5)
        
        self.post_blur_mode = self.create_scale(post_blur_frame, "Post Blur Mode (0:None,1:Gauss,2:Median)", 0, 2, 0)
        self.post_blur_kernel = self.create_scale(post_blur_frame, "Post Blur Kernel Size", 0, 31, 5)

    def create_scale(self, parent, label_text, min_val, max_val, init_val, orient=tk.HORIZONTAL):
        """Helper function to create a labeled scale and pack it into its parent."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        # Fixed width label ensures full text is visible
        lbl = ttk.Label(frame, text=label_text, width=40, anchor=tk.W)
        lbl.pack(side=tk.LEFT)
        # Use a smaller slider length (e.g., 200 pixels) for the settings area
        scale = tk.Scale(frame, from_=min_val, to=max_val, orient=orient, length=200)
        scale.set(init_val)
        scale.pack(side=tk.RIGHT)
        return scale

    def update_preview(self):
        """Retrieve settings, process the image, and update the preview."""
        # Get current parameter values from scales
        canny_t1 = int(self.edge_t1.get())
        canny_t2 = int(self.edge_t2.get())
        l2gradient = int(self.l2_grad.get())
        blur_mode = int(self.blur_mode.get())
        blur_kernel = int(self.blur_kernel.get())
        de_enable = int(self.de_enable.get())
        de_kernel = int(self.de_kernel.get())
        dilate_iter = int(self.dilate_iter.get())
        erode_iter = int(self.erode_iter.get())
        post_blur_mode = int(self.post_blur_mode.get())
        post_blur_kernel = int(self.post_blur_kernel.get())
        
        # Apply the edge pipeline using current settings
        processed = apply_edge_pipeline(
            self.orig_image,
            canny_t1, canny_t2, l2gradient,
            blur_mode, blur_kernel,
            de_enable, de_kernel, dilate_iter, erode_iter,
            post_blur_mode, post_blur_kernel
        )
        
        # Convert processed image to RGB for PIL and then to PhotoImage for Tkinter
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        im = Image.fromarray(processed_rgb)
        imgtk = ImageTk.PhotoImage(image=im)
        
        # Update the preview label image
        self.preview_label.imgtk = imgtk  # keep a reference
        self.preview_label.configure(image=imgtk)
        
        # Schedule next update
        self.after(100, self.update_preview)

if __name__ == "__main__":
    # Provide the path to your test image here:
    IMAGE_PATH = "data/test_images/test_general.jpg"
    # Target resolution for the image (width, height)
    TARGET_RESOLUTION = (900, 506)
    app = EdgeParameterTuning(IMAGE_PATH, target_resolution=TARGET_RESOLUTION)
    app.mainloop()
