import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from scipy.optimize import differential_evolution
import utils.dxf_reading as dxf_reading
import tkinter as tk
from tkinter import ttk, filedialog
import sv_ttk
import pickle

# -------------------------------------------------------------------------
# Global Configuration
# -------------------------------------------------------------------------
GRIPPER_RADIUS = 10
INNER_CONTOUR_BUFFER = 3
HOLE_SAFETY_BUFFER = 7
MAX_GRIPPER_DISTANCE = 270.0

# -------------------------------------------------------------------------
# Helper Function
# -------------------------------------------------------------------------
def move_point_along_contour(contour, point, step_size=0.5, direction='clockwise'):
    """
    Move a point along a given shapely contour (LineString) by a specified step size.
    """
    if direction not in ('clockwise', 'counterclockwise'):
        raise ValueError("Direction must be 'clockwise' or 'counterclockwise'.")

    total_len = contour.length
    current_dist = contour.project(Point(point))
    delta = step_size if direction == 'clockwise' else -step_size
    new_dist = (current_dist + delta) % total_len
    new_pt = contour.interpolate(new_dist)
    return np.array([new_pt.x, new_pt.y])


def get_holes_union_with_buffer(final_poly, hole_safety_buffer=0.0):
    """
    Return a single geometry representing the union of all holes
    (optionally buffered by 'hole_safety_buffer').
    """
    # If no holes, return None or an empty geometry
    if not final_poly.interiors:
        return None
    
    hole_polygons = []
    for hole in final_poly.interiors:
        hole_poly = Polygon(hole.coords)
        # Buffer the hole polygon outward if you want a margin
        # (this effectively enlarges the hole boundary, adding safety distance)
        if hole_safety_buffer > 0.0:
            hole_poly = hole_poly.buffer(hole_safety_buffer)
        hole_polygons.append(hole_poly)

    # Union of all hole polygons (possibly buffered)
    return unary_union(hole_polygons)


# -------------------------------------------------------------------------
# Main GUI Class
# -------------------------------------------------------------------------
class ContourGUI:
    def __init__(self, inner_contour=None, gripping_points=None, outer_contour=None,
                 gripper_radius=GRIPPER_RADIUS):
        """
        Main GUI and data initialization.
        """
        # Storing references and basic configuration
        self.gripper_radius = gripper_radius
        self.inner_contour = inner_contour
        self.gripping_points = gripping_points
        self.outer_contour_coords = outer_contour
        self.polygons = []
        self.original_points = None
        self.final_poly = None  # The full polygon (with possible holes)

        # -------------------- Root and Styles --------------------
        self.root = tk.Tk()
        self.root.title("Gripping Point Optimizer")
        self.root.configure(bg='#D3D3D3')
        self.root.geometry("1200x600")
        self.root.resizable(False, False)
        if sv_ttk is not None:
            sv_ttk.set_theme("dark")

        # Grid layout for main frames
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, minsize=200)

        # -------------------- Frames --------------------
        self.fig_frame = ttk.Frame(self.root, padding=0)
        self.fig_frame.grid(row=0, column=0, sticky="nsew")
        self.control_frame = ttk.Frame(self.root, padding=10)
        self.control_frame.grid(row=0, column=1, sticky="nsew")
        self.control_frame.grid_propagate(False)

        # -------------------- Matplotlib Setup --------------------
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.fig.patch.set_facecolor('#CCCCCC')
        self.ax.set_facecolor('#CCCCCC')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.fig_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial text if no DXF is loaded
        self.ax.text(0.5, 0.5, "No DXF loaded.\nClick 'Open DXF' to load a file.",
                     transform=self.ax.transAxes, ha='center', va='center',
                     fontsize=18, color='gray')
        self.canvas.draw()

        # -------------------- Control Widgets --------------------
        ttk.Label(self.control_frame, text="Step Size").pack(pady=(0, 5))
        self.step_slider = ttk.Scale(self.control_frame, from_=5.0, to=0.1,
                                     orient='vertical', length=150)
        self.step_slider.set(1.0)
        self.step_slider.pack(pady=(0, 25))

        # Gripper movement buttons
        btn_frame = ttk.Frame(self.control_frame)
        btn_frame.pack(pady=5, fill=tk.X)

        # Gripper 1 controls
        g1_frame = ttk.Frame(btn_frame)
        g1_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(g1_frame, text="G1 CW",
                   command=lambda: self.move_gripper(0, 'clockwise')
                   ).pack(fill=tk.X, pady=2, padx=2)
        ttk.Button(g1_frame, text="G1 CCW",
                   command=lambda: self.move_gripper(0, 'counterclockwise')
                   ).pack(fill=tk.X, pady=2, padx=2)

        # Gripper 2 controls
        g2_frame = ttk.Frame(btn_frame)
        g2_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(g2_frame, text="G2 CW",
                   command=lambda: self.move_gripper(1, 'clockwise')
                   ).pack(fill=tk.X, pady=2, padx=2)
        ttk.Button(g2_frame, text="G2 CCW",
                   command=lambda: self.move_gripper(1, 'counterclockwise')
                   ).pack(fill=tk.X, pady=2, padx=2)

        # Optimize & Reset
        ttk.Label(self.control_frame, text="").pack(pady=(5, 0))
        ttk.Button(self.control_frame, text="Optimize Grippers",
                   command=self.optimize_grippers).pack(pady=4, padx=4, fill=tk.X)
        ttk.Button(self.control_frame, text="Reset",
                   command=self.reset_grippers).pack(pady=4, padx=4, fill=tk.X)

        # Open DXF, Save Data, Exit
        ttk.Label(self.control_frame, text="").pack(pady=(50, 0))
        ttk.Button(self.control_frame, text="Open DXF",
                   command=self.open_dxf).pack(pady=4, padx=4, fill=tk.X)
        ttk.Button(self.control_frame, text="Save Data",
                   command=self.save_data).pack(pady=4, padx=4, fill=tk.X)
        ttk.Button(self.control_frame, text="Exit",
                   command=self.root.quit).pack(pady=4, padx=4, fill=tk.X)

    # ---------------------------------------------------------------------
    # Update & Movement Methods
    # ---------------------------------------------------------------------
    def move_gripper(self, index, direction):
        """
        Shift the specified gripper along the inner contour. Then refresh the plot.
        """
        if self.inner_contour is None or self.gripping_points is None:
            return

        step = float(self.step_slider.get())
        old_pt = self.gripping_points[index].copy()
        new_pt = move_point_along_contour(self.inner_contour, old_pt, step, direction)

        # Update in-memory positions
        self.gripping_points[index] = new_pt

        # Update label annotation on the plot
        self.annotations[index].remove()
        self.annotations[index] = self.ax.annotate(
            f"G{index+1}", new_pt, textcoords="offset points", xytext=(0, 10),
            ha='center', color='red'
        )
        # Update circle patch & scatter positions
        self.gripper_patches[index].center = tuple(new_pt)
        self.gripping_scatter.set_offsets(self.gripping_points)

        self.update_safe_area()

    def reset_grippers(self):
        """
        Reset grippers to their original positions.
        """
        if self.original_points is None or self.gripping_points is None:
            return
        self.gripping_points[:] = self.original_points.copy()

        # Refresh annotations and circle patches
        for i, pt in enumerate(self.gripping_points):
            self.annotations[i].remove()
            self.annotations[i] = self.ax.annotate(
                f"G{i+1}", pt, textcoords="offset points", xytext=(0, 10),
                ha='center', color='red'
            )
            self.gripper_patches[i].center = tuple(pt)

        self.gripping_scatter.set_offsets(self.gripping_points)
        self.update_safe_area()

    def update_safe_area(self):
        """
        Recompute and redraw the safe gripping shape, plus relevant distances.
        """
        if self.gripping_points is None:
            return

        # 1. Safe quadrilateral (convex hull around circles)
        union_of_grips = unary_union([Point(pt).buffer(self.gripper_radius)
                                      for pt in self.gripping_points])
        hull = union_of_grips.convex_hull
        sx, sy = hull.exterior.xy
        self.safe_quadrilateral.set_data(sx, sy)

        # Refresh fill by removing old fill collection and re-filling
        for coll in self.safe_fill:
            coll.remove()
        self.safe_fill[:] = self.ax.fill(sx, sy, alpha=0.2, color='magenta',
                                         label='Safe Gripping Area')

        # 2. Compute area
        safe_polygon = Polygon(np.array(hull.exterior.coords))
        area_val = safe_polygon.area
        self.area_annotation.set_text(f'Area: {area_val:.2f}')

        # 3. Compute maximum offset from outer boundary
        if self.outer_contour_coords is not None:
            outer_poly = Polygon(self.outer_contour_coords)
            boundary_pts = [Point(bp) for bp in outer_poly.exterior.coords]
            distances = [safe_polygon.exterior.distance(bpt) for bpt in boundary_pts]
            max_dist = max(distances)
            self.max_offset_annotation.set_text(f'Max Offset: {max_dist:.2f}')

            # For a dashed line from that boundary point to hull
            boundary_pt = boundary_pts[np.argmax(distances)]
            closest_pt = safe_polygon.exterior.interpolate(
                safe_polygon.exterior.project(boundary_pt)
            )
            self.max_offset_line.set_data([boundary_pt.x, closest_pt.x],
                                          [boundary_pt.y, closest_pt.y])

        # 4. Distance between grippers
        dist_between = np.linalg.norm(self.gripping_points[0] - self.gripping_points[1])
        self.gripper_distance_annotation.set_text(f'Gripper Distance: {dist_between:.2f}')

        # Final refresh
        self.ax.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=8)
        self.canvas.draw_idle()

    # ---------------------------------------------------------------------
    # Optimization Methods
    # ---------------------------------------------------------------------
    def optimize_grippers_diff_evo(self, objective_type="min_max_distance"):
        """
        Optimizes a single gripping point along the inner contour.
        The second gripping point is set at a fixed half contour length away,
        ensuring symmetry based on the physical gripper design.
        """
        if self.inner_contour is None:
            print("No shape loaded; cannot optimize.")
            return

        # Get inner contour length and outer boundary points
        contour_len = self.inner_contour.length
        outer_poly = Polygon(self.outer_contour_coords)
        boundary_pts = [Point(bp) for bp in outer_poly.exterior.coords]

        # Helper: Convert a parametric value into an XY point on the inner contour
        def get_point(s):
            # Convert input to a scalar if necessary
            s_val = s if np.isscalar(s) else s[0]
            s_val %= contour_len
            p = self.inner_contour.interpolate(s_val)
            return np.array([p.x, p.y])


        # Define the base objective function using both symmetric points
        def base_objective(s):
            p1 = get_point(s)
            p2 = get_point(s + contour_len/2)
            pts = [p1, p2]
            if objective_type == "min_max_distance":
                # Compute the convex hull of the buffered gripper circles
                union_buff = unary_union([Point(pt).buffer(GRIPPER_RADIUS) for pt in pts])
                hull = union_buff.convex_hull
                # The objective is to minimize the maximum distance from the boundary
                return max(hull.exterior.distance(bpt) for bpt in boundary_pts)
            else:
                raise ValueError("Invalid objective_type for symmetric optimization.")

        # Dynamic penalty function that scales with the degree of violation
        def penalty(s):
            penalty_val = 0.0
            p1 = get_point(s)
            p2 = get_point(s + contour_len/2)
            for pt in [p1, p2]:
                circle = Point(pt).buffer(GRIPPER_RADIUS)
                if self.holes_union is not None and not self.holes_union.is_empty and circle.intersects(self.holes_union):
                    # Use intersection area as a smooth penalty measure
                    intersect_area = circle.intersection(self.holes_union).area
                    penalty_val += intersect_area * 100  # Adjust scaling factor as needed
            return penalty_val

        # Combined objective that includes the dynamic penalty
        def combined_obj(s):
            return base_objective(s) + penalty(s)

        # Optimize using differential evolution (1-dimensional problem)
        bounds = [(0, contour_len)]
        result = differential_evolution(
            combined_obj,
            bounds=bounds,
            maxiter=100,
            popsize=15,
            mutation=(0.5, 1.5),
            recombination=0.7,
            strategy='best1bin',
            tol=0.05,
            polish=True,
            disp=True
        )

        if result.success:
            s_optimized = result.x[0]
            print(f"Optimization successful")
        else:
            print("Optimization did not meet tolerance; using best solution from last iteration.")
            s_optimized = result.x[0]

        # Compute the two gripping points from the optimized parameter
        new_points = [get_point(s_optimized), get_point(s_optimized + contour_len/2)]
        self.gripping_points[:] = np.array(new_points)

        # Update annotations and circle patches
        for i, pt in enumerate(self.gripping_points):
            self.annotations[i].remove()
            self.annotations[i] = self.ax.annotate(
                f"G{i+1}", pt, textcoords="offset points", xytext=(0, 10),
                ha='center', color='red'
            )
            self.gripper_patches[i].center = tuple(pt)

        self.gripping_scatter.set_offsets(self.gripping_points)
        self.update_safe_area()

    def optimize_grippers(self):
        """
        Simplified 'master' method to choose an optimization approach.
        Currently uses Differential Evolution with 'min_max_distance'.
        """
        self.optimize_grippers_diff_evo(objective_type="min_max_distance")
    # ---------------------------------------------------------------------
    # File I/O and DXF Loading
    # ---------------------------------------------------------------------
    def save_data(self):
        """
        Save the final polygon (with holes), gripper points, and gripper distance
        to a pickle file.
        """
        if not self.final_poly:
            print("No polygon data available to save.")
            return

        # Compute gripper distance and additional angle from G1 to G2
        gripper_distance = np.linalg.norm(self.gripping_points[0] - self.gripping_points[1])
        gripper_line_angle = self.compute_gripper_line_angle()
        print(gripper_line_angle)

        # Prepare a dictionary with the complete data structure.
        data = {
            "final_poly": self.final_poly,  # Shapely polygon (including holes)
            "gripping_points": self.gripping_points,  # Gripping points array
            "gripper_distance": gripper_distance,
            "gripper_line_angle": gripper_line_angle
        }

        output_file = 'data/gripping_data.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)

        print(f"Data saved to {output_file} successfully.")


    def open_dxf(self):
        """
        Load a DXF file and extract the outer/inner polygons. Initialize default gripper points.
        """
        file_path = filedialog.askopenfilename(
            title="Open DXF",
            filetypes=[("DXF Files", "*.dxf"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        # Extract the final polygon (exterior + holes)
        self.final_poly = dxf_reading.extract_polygons_from_dxf(file_path, circle_approx_points=80)
        if self.final_poly is None:
            print("No valid polygon found or shape not closed.")
            return
        
                # Inside open_dxf right after setting self.final_poly:
        self.holes_union = get_holes_union_with_buffer(self.final_poly, hole_safety_buffer=HOLE_SAFETY_BUFFER)

        self.ax.clear()
        outer_coords = np.array(self.final_poly.exterior.coords)
        self.ax.plot(outer_coords[:, 0], outer_coords[:, 1], 'b-', label="Outer Boundary")
        self.ax.fill(outer_coords[:, 0], outer_coords[:, 1], alpha=0.2, fc='blue')

        # Plot holes (if any)
        hole_label = True
        for hole in self.final_poly.interiors:
            hole_coords = np.array(hole.coords)
            self.ax.plot(hole_coords[:, 0], hole_coords[:, 1], 'r-',
                         label=("Hole Boundary" if hole_label else None))
            self.ax.fill(hole_coords[:, 0], hole_coords[:, 1], fc='white')
            hole_label = False

        # Axes settings
        self.ax.set_title('Gripping Point Optimizer', fontsize=14, weight='bold')
        self.ax.set_xlabel('X-coordinate')
        self.ax.set_ylabel('Y-coordinate')
        self.ax.grid(True)
        self.ax.axis('equal')

        # Save outer polygon info for offset calculation
        self.outer_contour_coords = outer_coords
        self.polygons = [outer_coords]

        # -------------------- Inner Contour --------------------
        inset_dist = GRIPPER_RADIUS + INNER_CONTOUR_BUFFER
        outer_polygon = Polygon(self.final_poly.exterior.coords)
        inner_polygon = outer_polygon.buffer(-inset_dist)

        if inner_polygon.is_empty:
            print("Inner contour is empty. Radius + buffer may be too large.")
            self.inner_contour = None
            self.canvas.draw_idle()
            return

        # If there's a MultiPolygon, pick the largest piece
        if inner_polygon.geom_type == 'MultiPolygon':
            inner_polygon = max(inner_polygon.geoms, key=lambda p: p.area)

        in_coords = np.array(inner_polygon.exterior.coords)
        self.inner_contour = LineString(np.vstack([in_coords, in_coords[0]]))
        self.ax.plot(in_coords[:, 0], in_coords[:, 1], 'g--', label="Inner Contour")

        # -------------------- Default Gripper Positions --------------------
        c_len = self.inner_contour.length
        p1 = self.inner_contour.interpolate(0.0)
        p2 = self.inner_contour.interpolate(c_len / 2)
        self.gripping_points = np.array([[p1.x, p1.y], [p2.x, p2.y]])
        self.original_points = self.gripping_points.copy()

        # Scatter + Circles
        self.gripping_scatter = self.ax.scatter(
            self.gripping_points[:, 0], self.gripping_points[:, 1],
            s=100, color='red', label="Gripping Points"
        )
        self.annotations = []
        self.gripper_patches = []
        for i, pt in enumerate(self.gripping_points, start=1):
            ann = self.ax.annotate(f"G{i}", pt, textcoords="offset points", xytext=(0, 10),
                                   ha='center', color='red')
            self.annotations.append(ann)
            cir = plt.Circle((pt[0], pt[1]), GRIPPER_RADIUS, color='cyan', alpha=0.3)
            self.gripper_patches.append(cir)
            self.ax.add_patch(cir)

        # -------------------- Safe Area Artifacts --------------------
        self._build_safe_area_artifacts()
        self.update_safe_area()

        self.ax.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=8)
        self.canvas.draw()

    def _build_safe_area_artifacts(self):
        """
        Initialize empty placeholders for safe quadrilateral line,
        fill color, and text annotations for max offset, area, etc.
        """
        union_of_grips = unary_union(
            [Point(pt).buffer(self.gripper_radius) for pt in self.gripping_points]
        )
        hull = union_of_grips.convex_hull
        xh, yh = hull.exterior.xy

        # Safe quadrilateral lines/fill
        self.safe_quadrilateral, = self.ax.plot(xh, yh, 'm-')
        self.safe_fill = self.ax.fill(xh, yh, alpha=0.2, color='magenta',
                                      label="Safe Gripping Area")

        # Offset line & annotation placeholders
        self.max_offset_line, = self.ax.plot([], [], 'r--', lw=2)
        anno_box = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6)
        self.gripper_distance_annotation = self.ax.annotate(
            "Gripper Distance: ???", xy=(0.02, 0.98), xycoords="axes fraction",
            ha='left', va='top', color='black', bbox=anno_box)
        self.max_offset_annotation = self.ax.annotate(
            "Max Offset: ???", xy=(0.02, 0.92), xycoords="axes fraction",
            ha='left', va='top', color='black', bbox=anno_box)
        self.area_annotation = self.ax.annotate(
            "Area: ???", xy=(0.02, 0.86), xycoords="axes fraction",
            ha='left', va='top', color='black', bbox=anno_box)
        
    def compute_gripper_line_angle(self):
        """
        Compute the angle (in degrees) of the line connecting G1 and G2 relative to the horizontal axis.
        The angle is normalized to be within [-90, 90] degrees.
        """
        if self.gripping_points is None or len(self.gripping_points) < 2:
            return 0.0
        p1 = self.gripping_points[0]
        p2 = self.gripping_points[1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = np.degrees(np.arctan2(dy, dx))
        # Normalize the angle to be within [-90, 90] degrees.
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180
        return angle



    def run(self):
        self.root.mainloop()

def main():
    gui = ContourGUI()
    gui.run()


if __name__ == "__main__":
    main()
