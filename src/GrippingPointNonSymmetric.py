import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point, MultiPoint
from shapely.ops import unary_union
from matplotlib.widgets import Slider, Button
from scipy.optimize import minimize
import utils.Dxf_reading as dxf_reading
import csv


def calculate_centroid(coords):
    """
    Calculate the centroid of a polygon using Shapely's area-weighted method.

    Parameters:
    - coords (np.ndarray): Array of shape (n, 2) representing polygon vertices.

    Returns:
    - np.ndarray: Centroid coordinates as a 1D array [x, y].
    """
    # Ensure the polygon is closed (first and last points are the same)
    if not np.array_equal(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])

    # Create a Shapely Polygon object
    polygon = Polygon(coords)

    # Validate the polygon
    if not polygon.is_valid:
        raise ValueError("Invalid polygon for centroid calculation.")

    # Calculate the centroid
    centroid = polygon.centroid
    return np.array([centroid.x, centroid.y])

def calculate_internal_angles(coords):
    """
    Calculate internal angles at each vertex of a polygon.

    Parameters:
    - coords (np.ndarray): Array of shape (n, 2) representing polygon vertices.

    Returns:
    - np.ndarray: Array of internal angles in degrees for each vertex.
    """
    # Remove the last point if it duplicates the first to avoid redundancy
    if np.array_equal(coords[0], coords[-1]):
        coords = coords[:-1]

    # Remove consecutive duplicate points
    unique_coords = [coords[0]]
    for point in coords[1:]:
        if not np.array_equal(point, unique_coords[-1]):
            unique_coords.append(point)
    coords = np.array(unique_coords)

    num_points = len(coords)
    angles = np.zeros(num_points)

    for i in range(num_points):
        p_prev = coords[i - 1]
        p_current = coords[i]
        p_next = coords[(i + 1) % num_points]

        v1 = p_prev - p_current
        v2 = p_next - p_current

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        cos_angle = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angles[i] = np.degrees(angle_rad)

    return angles


def find_closest_intersection(line, boundary, point):
    """
    Find the closest intersection point between a line and a boundary.

    Parameters:
    - line (LineString): The line to intersect with the boundary.
    - boundary (LineString): The boundary to find intersections with.
    - point (np.ndarray): Original point for distance reference.

    Returns:
    - np.ndarray or None: The closest intersection point or None if no intersection.
    """
    intersection = line.intersection(boundary)

    if intersection.is_empty:
        return None
    elif isinstance(intersection, Point):
        return np.array([intersection.x, intersection.y])
    elif isinstance(intersection, MultiPoint):
        distances = [np.linalg.norm(np.array([pt.x, pt.y]) - point) for pt in intersection]
        min_idx = np.argmin(distances)
        closest_point = intersection[min_idx]
        return np.array([closest_point.x, closest_point.y])
    elif isinstance(intersection, LineString):
        # Select the endpoint closest to the original point
        start, end = intersection.boundary
        start_dist = np.linalg.norm(np.array([start.x, start.y]) - point)
        end_dist = np.linalg.norm(np.array([end.x, end.y]) - point)
        closest_end = start if start_dist < end_dist else end
        return np.array([closest_end.x, closest_end.y])
    else:
        return None  # Unexpected geometry type


def adjust_point_to_contour(point, centroid, boundary, verbose: bool = False, extension_factor: float = 2.0):
    """
    Adjust a point by moving it towards the centroid until it intersects the boundary.

    Parameters:
    - point (np.ndarray): Original point to adjust.
    - centroid (np.ndarray): Centroid of the shape.
    - boundary (LineString): Boundary to intersect with.
    - verbose (bool): If True, prints debug messages.
    - extension_factor (float): Factor to extend the line if no intersection is found.

    Returns:
    - np.ndarray: Adjusted point on the boundary.
    """
    # Direct line from point to centroid
    line = LineString([point, centroid])
    closest_point = find_closest_intersection(line, boundary, point)

    if closest_point is None:
        if verbose:
            print("[DEBUG] No intersection on initial line. Extending line for broader search.")

        direction = point - centroid
        extended_point = point + direction * extension_factor
        extended_line = LineString([centroid, extended_point])

        closest_point = find_closest_intersection(extended_line, boundary, point)

    return closest_point if closest_point is not None else point


def select_gripping_points(coords, angles, centroid, boundary, angle_threshold=150.0, gripper_radius=1.0, verbose=False):
    """
    Select four gripping points independently based on internal angles and distribution.

    Parameters:
    - coords (np.ndarray): Array of shape (n, 2) representing polygon vertices.
    - angles (np.ndarray): Array of internal angles in degrees corresponding to the vertices.
    - centroid (np.ndarray): Centroid of the shape.
    - boundary (LineString): Boundary to intersect with.
    - angle_threshold (float): Maximum angle to consider a vertex significant.
    - gripper_radius (float): Radius of the gripper to enforce minimum distance.
    - verbose (bool): If True, prints debug messages.

    Returns:
    - np.ndarray: Array of selected gripping points.
    """
    # Filter vertices based on angle_threshold
    candidate_points = coords[angles < angle_threshold]

    if verbose:
        print(f"[DEBUG] Number of candidate points: {len(candidate_points)}")

    if len(candidate_points) < 4:
        raise ValueError("Not enough candidate points to select gripping points.")

    # Initialize gripping points list
    gripping_points = []

    # Start with a point farthest from the centroid
    distances_to_centroid = np.linalg.norm(candidate_points - centroid, axis=1)
    idx = np.argmax(distances_to_centroid)
    gripping_points.append(candidate_points[idx])
    candidate_points = np.delete(candidate_points, idx, axis=0)

    # Select remaining points to maximize the minimum distance between them
    for _ in range(3):
        max_min_dist = -np.inf
        best_candidate = None
        for candidate in candidate_points:
            min_dist = min(np.linalg.norm(candidate - gp) for gp in gripping_points)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_candidate = candidate
        gripping_points.append(best_candidate)
        candidate_points = candidate_points[np.any(candidate_points != best_candidate, axis=1)]

    gripping_points = np.array(gripping_points)

    # Validate minimum distances between gripping points
    min_distance = 2 * gripper_radius
    for i in range(len(gripping_points)):
        for j in range(i + 1, len(gripping_points)):
            distance = np.linalg.norm(gripping_points[i] - gripping_points[j])
            if distance < min_distance:
                raise ValueError(f"Gripping points {i + 1} and {j + 1} are too close: {distance:.2f} < {min_distance}")

    if verbose:
        print("\n[DEBUG] Selected Gripping Points:")
        for idx, point in enumerate(gripping_points, 1):
            print(f"  G{idx}: {point}")

    return gripping_points


def move_point_along_contour(inner_contour, point, step_size=0.5, direction='clockwise'):
    """
    Move a point along the inner contour by a specified step size and direction.

    Parameters:
    - inner_contour (LineString): Shapely LineString representing the closed inner contour.
    - point (np.ndarray): Original point coordinates.
    - step_size (float): Distance to move the point.
    - direction (str): Direction to move ('clockwise' or 'counterclockwise').

    Returns:
    - np.ndarray: New point coordinates after movement.
    """
    if direction not in ['clockwise', 'counterclockwise']:
        raise ValueError("Direction must be 'clockwise' or 'counterclockwise'.")

    total_length = inner_contour.length
    current_distance = inner_contour.project(Point(point))

    if direction == 'clockwise':
        new_distance = current_distance + step_size
    else:
        new_distance = current_distance - step_size

    new_distance %= total_length
    new_point = inner_contour.interpolate(new_distance)

    return np.array([new_point.x, new_point.y])


def interactive_contour_movement(inner_contour, gripping_points, outer_contour=None, gripper_radius=1.0):
    """
    Create an interactive GUI to move gripping points along the inner contour.

    Parameters:
    - inner_contour (LineString): Shapely LineString representing the closed inner contour.
    - gripping_points (np.ndarray): Array of current gripping points.
    - outer_contour (np.ndarray): Array of outer contour vertices for visualization.
    - gripper_radius (float): Radius representing the extent of each gripper.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(right=0.79)  # Reserve space on the right side

    # Define outer_polygon from outer_contour
    if outer_contour is not None:
        outer_polygon = Polygon(outer_contour)
    else:
        raise ValueError("outer_contour must be provided to define outer_polygon.")

    # Plot outer contour if provided
    if outer_contour is not None:
        closed_outer = outer_contour
        if not np.array_equal(outer_contour[0], outer_contour[-1]):
            closed_outer = np.vstack([outer_contour, outer_contour[0]])
        ax.plot(closed_outer[:, 0], closed_outer[:, 1], 'b-', label='Fabric Outline')
        ax.fill(closed_outer[:, 0], closed_outer[:, 1], alpha=0.05, color='blue', label='Fabric Area')

    # Plot inner contour
    ax.plot(*inner_contour.xy, 'g--', label='Inner Contour')
    ax.fill(*inner_contour.xy, alpha=0.1, color='green')

    # Plot gripping points
    gripping_scatter = ax.scatter(gripping_points[:, 0], gripping_points[:, 1],
                                  color='red', s=100, label='Gripping Points')

    # Annotate gripping points
    annotations = []
    for i, point in enumerate(gripping_points, 1):
        annotation = ax.annotate(f'G{i}', (point[0], point[1]),
                                 textcoords="offset points", xytext=(0, 10),
                                 ha='center', color='red')
        annotations.append(annotation)

    # Create gripper areas as circles
    gripper_patches = []
    for point in gripping_points:
        circle = plt.Circle((point[0], point[1]), gripper_radius,
                            color='cyan', alpha=0.3)
        if len(gripper_patches) == 0:  # Add label only once
            circle.set_label('Gripping Area')
        gripper_patches.append(circle)
        ax.add_patch(circle)

    # Initialize maximum offset line (initially empty)
    max_offset_line, = ax.plot([], [], 'r--', linewidth=2, label='Max Offset')

    # Create safe quadrilateral encompassing gripping areas
    def update_safe_quadrilateral():
        nonlocal safe_quadrilateral, safe_fill, safe_area, area_annotation, max_offset_annotation, max_offset_line
        gripping_buffers = [Point(pt).buffer(gripper_radius) for pt in gripping_points]
        combined_area = unary_union(gripping_buffers)
        safe_quad = combined_area.convex_hull

        # Update quadrilateral plot
        safe_quadrilateral.set_data(*safe_quad.exterior.xy)

        # Remove previous fill and recreate
        for fill in safe_fill:
            fill.remove()
        safe_fill[:] = ax.fill(*safe_quad.exterior.xy, alpha=0.2,
                               color='magenta', label='Safe Quadrilateral Area')

        # Calculate internal angles and area
        safe_coords = np.array(safe_quad.exterior.coords)
        safe_polygon = Polygon(safe_coords)
        safe_area = safe_polygon.area

        # Update area annotation
        area_annotation.set_position((safe_polygon.centroid.x, safe_polygon.centroid.y))
        area_annotation.set_text(f'Area: {safe_area:.2f}')

        # Update maximum offset annotation and line
        max_offset, boundary_point, closest_point = compute_maximum_offset(safe_polygon, outer_polygon)
        max_offset_annotation.set_position((safe_polygon.centroid.x, safe_polygon.centroid.y - 20))
        max_offset_annotation.set_text(f'Max Offset: {max_offset:.2f}')

        # Update maximum offset line
        max_offset_line.set_data([boundary_point[0], closest_point[0]],
                                 [boundary_point[1], closest_point[1]])

        plt.draw()

    # Function to compute maximum offset
    def compute_maximum_offset(safe_polygon, outer_polygon):
        """
        Compute the maximum offset distance from the outer polygon to the safe polygon,
        and identify the corresponding boundary and closest points.

        Returns:
        - max_distance (float): The maximum offset distance.
        - boundary_point_coords (np.ndarray): Coordinates of the boundary point with maximum offset.
        - closest_point_coords (np.ndarray): Coordinates of the closest point on the safe polygon.
        """
        # Sample points on the outer boundary
        boundary_points = np.array(outer_polygon.exterior.coords)
        boundary_points = [Point(bp) for bp in boundary_points]

        # Compute distances from boundary points to the safe quadrilateral
        distances = [safe_polygon.exterior.distance(bp) for bp in boundary_points]

        # Find the maximum distance and its index
        max_distance = max(distances)
        max_index = distances.index(max_distance)
        boundary_point = boundary_points[max_index]

        # Find the closest point on the safe_polygon to the boundary_point
        closest_point = safe_polygon.exterior.interpolate(safe_polygon.exterior.project(boundary_point))
        closest_coords = np.array([closest_point.x, closest_point.y])

        # Return both maximum distance and the points
        return max_distance, np.array([boundary_point.x, boundary_point.y]), closest_coords

    # Prepare initial safe quadrilateral and maximum offset
    gripping_area_polygons = [Point(pt).buffer(gripper_radius) for pt in gripping_points]
    combined_gripping_area = unary_union(gripping_area_polygons)
    safe_quad = combined_gripping_area.convex_hull
    safe_quadrilateral, = ax.plot(*safe_quad.exterior.xy, 'm-', label='Safe Quadrilateral')
    safe_fill = ax.fill(*safe_quad.exterior.xy,
                        alpha=0.2, color='magenta', label='Safe Quadrilateral Area')

    safe_coords = np.array(combined_gripping_area.convex_hull.exterior.coords)
    safe_polygon = Polygon(safe_coords)
    safe_area = safe_polygon.area

    # Annotate area
    centroid_safe = safe_polygon.centroid
    area_annotation = ax.annotate(f'Area: {safe_area:.2f}',
                                  (centroid_safe.x, centroid_safe.y),
                                  textcoords="offset points", xytext=(0, 0),
                                  ha='center', color='black', fontsize=14, weight='bold',
                                  bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # Annotate maximum offset
    max_offset, boundary_point, closest_point = compute_maximum_offset(safe_polygon, outer_polygon)
    max_offset_annotation = ax.annotate(f'Max Offset: {max_offset:.2f}',
                                        (centroid_safe.x, centroid_safe.y - 20),
                                        textcoords="offset points", xytext=(0, 0),
                                        ha='center', color='black', fontsize=14, weight='bold',
                                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # Initialize maximum offset line
    max_offset_line.set_data([boundary_point[0], closest_point[0]],
                             [boundary_point[1], closest_point[1]])

    # Legend and plot settings
    ax.legend(loc='upper left', bbox_to_anchor=(1,1.01))
    ax.axis('equal')
    ax.set_title('Interactive Gripping Area Visualization')
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.grid(True)

    # Slider for step size, placed at the top-right
    axcolor = 'lightgoldenrodyellow'
    slider_step = Slider(ax=plt.axes([0.955, 0.68, 0.03, 0.2], facecolor=axcolor),  # x, y, width, height
                        label='Step Size', valmin=0.1, valmax=2.0, valinit=1, valstep=0.1, color='blue',
                        orientation='vertical')  # Vertical slider for cleaner appearance

    # Create buttons for each gripping point
    button_g1_ccw = Button(plt.axes([0.80, 0.59, 0.095, 0.06]), 'G1 Up', color=axcolor, hovercolor='0.975')
    button_g1_cw = Button(plt.axes([0.90, 0.59, 0.095, 0.06]), 'G1 Down', color=axcolor, hovercolor='0.975')

    button_g2_ccw = Button(plt.axes([0.80, 0.52, 0.095, 0.06]), 'G2 Up', color=axcolor, hovercolor='0.975')
    button_g2_cw = Button(plt.axes([0.90, 0.52, 0.095, 0.06]), 'G2 Down', color=axcolor, hovercolor='0.975')

    button_g3_ccw = Button(plt.axes([0.80, 0.45, 0.095, 0.06]), 'G3 Up', color=axcolor, hovercolor='0.975')
    button_g3_cw = Button(plt.axes([0.90, 0.45, 0.095, 0.06]), 'G3 Down', color=axcolor, hovercolor='0.975')

    button_g4_ccw = Button(plt.axes([0.80, 0.38, 0.095, 0.06]), 'G4 Up', color=axcolor, hovercolor='0.975')
    button_g4_cw = Button(plt.axes([0.90, 0.38, 0.095, 0.06]), 'G4 Down', color=axcolor, hovercolor='0.975')

    # Optimize and Reset buttons, stacked at the bottom
    button_opt = Button(plt.axes([0.80, 0.29, 0.195, 0.08]), 'Optimize Grippers', color='lightgreen', hovercolor='0.975')
    button_reset = Button(plt.axes([0.80, 0.20, 0.195, 0.08]), 'Reset', color=axcolor, hovercolor='0.975')
    button_save = Button(plt.axes([0.80, 0.11, 0.195, 0.08]), 'Save Data', color='lightblue', hovercolor='0.975')

    # Store original positions for reset
    original_points = gripping_points.copy()

 # Define the save function
    def save_data(event):
        """
        Saves the polygon coordinates and gripping points to a CSV file.
        """
        global polygons, gripping_points # Access the global variables
        
        # Define the output CSV file path
        output_file = 'data/gripping_data.csv'
        
        # Extract polygon coordinates (assuming the first polygon is the primary one)
        polygon = polygons[0]
        polygon_list = polygon.tolist()

        # Extract gripping points
        gripping_points_list = gripping_points.tolist()

        # Write to CSV
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Polygon Coordinates'])
            for coord in polygon_list:
                writer.writerow(coord)
            writer.writerow([])  # Empty line for separation
            writer.writerow(['Gripping Points'])
            writer.writerow(['G1_X', 'G1_Y'])
            writer.writerow(gripping_points_list[0])
            writer.writerow(['G2_X', 'G2_Y'])
            writer.writerow(gripping_points_list[1])
            writer.writerow(['G3_X', 'G3_Y'])
            writer.writerow(gripping_points_list[2])
            writer.writerow(['G4_X', 'G4_Y'])
            writer.writerow(gripping_points_list[3])
        
        print(f"Data saved to {output_file} successfully.")

    # Update move_gripper function
    def move_gripper(index, direction):
        """
        Move a gripping point in the specified direction.

        Parameters:
        - index (int): Index of the gripping point (0 to 3).
        - direction (str): 'clockwise' or 'counterclockwise'.
        """
        step = slider_step.val
        primary_point = gripping_points[index].copy()
        new_point = move_point_along_contour(inner_contour, primary_point, step, direction)
        gripping_points[index] = new_point

        # Update annotation and gripper patch
        annotations[index].remove()
        annotations[index] = ax.annotate(f'G{index + 1}', (gripping_points[index][0], gripping_points[index][1]),
                                        textcoords="offset points", xytext=(0, 10),
                                        ha='center', color='red')
        gripper_patches[index].center = (gripping_points[index][0], gripping_points[index][1])

        # Update scatter plot
        gripping_scatter.set_offsets(gripping_points)

        # Update safe quadrilateral
        update_safe_quadrilateral()

        plt.draw()

    def reset_grippers(event):
        """
        Reset gripping points to their original positions.
        """
        nonlocal gripping_points
        gripping_points[:] = original_points.copy()

        # Update annotations and gripper patches
        for i in range(4):
            annotations[i].remove()
            annotations[i] = ax.annotate(f'G{i+1}', (gripping_points[i][0], gripping_points[i][1]),
                                        textcoords="offset points", xytext=(0, 10),
                                        ha='center', color='red')
            gripper_patches[i].center = (gripping_points[i][0], gripping_points[i][1])

        # Update scatter plot
        gripping_scatter.set_offsets(gripping_points)

        # Update safe quadrilateral
        update_safe_quadrilateral()

        plt.draw()

    def optimize_grippers(event):
        """
        Optimize the positions of the gripping points to minimize the maximum offset.
        """
        contour_length = inner_contour.length

        def get_point(s):
            s = s % contour_length
            point = inner_contour.interpolate(s)
            return np.array([point.x, point.y])

        # Sample points on the outer boundary
        boundary_points = np.array(outer_polygon.exterior.coords)
        boundary_points = [Point(bp) for bp in boundary_points]

        # Objective function for optimization
        def objective(s):
            # Construct gripper buffers
            G_points = [get_point(si) for si in s]
            gripper_buffers = [Point(pt).buffer(gripper_radius) for pt in G_points]
            safe_quad = unary_union(gripper_buffers).convex_hull

            # Compute distances from boundary points to the safe quadrilateral
            distances = [safe_quad.exterior.distance(bp) for bp in boundary_points]
            # Return the maximum distance
            return max(distances)

        # Constraints to ensure gripping points are sufficiently separated
        def constraint_min_distance(s):
            G_points = [get_point(si) for si in s]
            distances = []
            min_distance = 2 * gripper_radius
            for i in range(len(G_points)):
                for j in range(i + 1, len(G_points)):
                    dist = np.linalg.norm(G_points[i] - G_points[j]) - min_distance
                    distances.append(dist)
            return distances  # Must be greater than zero

        # Initial guess: current positions of the gripping points
        s_initial = [inner_contour.project(Point(gp)) for gp in gripping_points]

        # Bounds: s values must be within the contour length
        bounds = [(0, contour_length) for _ in range(4)]

        # Constraints
        constraints = {
            'type': 'ineq',
            'fun': constraint_min_distance
        }


        # Separate optimization configurations
        def optimize_with_slsqp():
            return minimize(
                objective,
                x0=s_initial,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={
                    'disp': True,
                    'xtol': 1e-8,
                    'maxiter': 200
                }
            )

        def optimize_with_trust_constr():
            return minimize(
                fun=objective,
                x0=s_initial,
                method='trust-constr',
                bounds=bounds,
                constraints=constraints,
                options={
                    'disp': True,
                    'maxiter': 500,
                    'gtol': 1e-4,
                    'xtol': 1e-8,
                    'barrier_tol': 0.01,        # Allow temporary constraint violations
                    'initial_tr_radius': 2.0,   # Start with a larger trust region
                    'finite_diff_rel_step': 1e-4  # Slightly larger finite difference step size
                }
            )

        #result = optimize_with_slsqp()
        result = optimize_with_trust_constr()

        if result.success:
            optimized_s = result.x
            G_opt = [get_point(si) for si in optimized_s]
            gripping_points[:] = np.array(G_opt)

            # Update visual elements
            for i in range(4):
                annotations[i].remove()
                annotations[i] = ax.annotate(f'G{i+1}', (gripping_points[i][0], gripping_points[i][1]),
                                            textcoords="offset points", xytext=(0, 10),
                                            ha='center', color='red')
                gripper_patches[i].center = (gripping_points[i][0], gripping_points[i][1])

            gripping_scatter.set_offsets(gripping_points)
            update_safe_quadrilateral()
            plt.draw()
            print("Optimization successful!")
        else:
            print("Optimization failed:", result.message)

    # Connect buttons to their callbacks
    button_g1_cw.on_clicked(lambda event: move_gripper(0, 'clockwise'))
    button_g1_ccw.on_clicked(lambda event: move_gripper(0, 'counterclockwise'))
    button_g2_cw.on_clicked(lambda event: move_gripper(1, 'clockwise'))
    button_g2_ccw.on_clicked(lambda event: move_gripper(1, 'counterclockwise'))
    button_g3_cw.on_clicked(lambda event: move_gripper(2, 'clockwise'))
    button_g3_ccw.on_clicked(lambda event: move_gripper(2, 'counterclockwise'))
    button_g4_cw.on_clicked(lambda event: move_gripper(3, 'clockwise'))
    button_g4_ccw.on_clicked(lambda event: move_gripper(3, 'counterclockwise'))
    button_opt.on_clicked(optimize_grippers)
    button_reset.on_clicked(reset_grippers)
    button_save.on_clicked(save_data)

    plt.show()


def main(dxf_file_path=None):
    global polygons, gripping_points
    """
    Main function to execute gripping points selection, movement, optimization, and visualization.
    """
    # User-Configurable Parameters
    inner_contour_distance = 11  # Buffer distance for inner contour
    angle_threshold = 150  # Max angle to consider a vertex significant
    gripper_radius = 10  # Radius of each gripper

    if dxf_file_path:
        # Use the DXF file to get the outline
        # Call the function from dxf_reading module
        polygons_extracted = dxf_reading.extract_polygons_from_dxf(
            dxf_path=dxf_file_path,
            circle_approx_points=80,
            layers=None  # Set to specific layers if needed
        )

        if not polygons_extracted:
            print("No polygons were extracted from the DXF file.")
            return

        # For simplicity, we'll use the first polygon extracted
        outline_coords = polygons_extracted[0]

    else:
        # Predefined Shapes
        shapes = {
            "original": 20 * np.array([
                [0, 0], [2, 4], [4, 5], [6, 4], [8, 0], [6, -3], [4, -4], [2, -3], [0, 0]
            ]),
            "difficult_polygon": 20 * np.array([
                [0, 0], [3, 1], [2, 2], [4, 4], [2, 5], [3, 6], [0, 5],
                [-3, 6], [-2, 5], [-4, 4], [-2, 2], [-3, 1]
            ]),
            "hexagon": 50 * np.array([
                [1, 0], [0.5, np.sqrt(3) / 2], [-0.5, np.sqrt(3) / 2],
                [-1, 0], [-0.5, -np.sqrt(3) / 2], [0.5, -np.sqrt(3) / 2]
            ]),
            "star": 25 * np.array([
                [0, 3], [1, 1], [3, 1], [1.5, -1], [2, -3],
                [0, -2], [-2, -3], [-1.5, -1], [-3, 1], [-1, 1]
            ]),
            "concave_pentagon": 20 * np.array([
                [0, 6], [4, 3], [3, 0], [6, -3], [0, -6],
                [-6, -3], [-3, 0], [-4, 3], [0, 6]
            ]),
            "diamond": 15 * np.array([
                [0, 6], [2, 2], [6, 0], [2, -2], [0, -6], [-2, -2], [-6, 0], [-2, 2]
            ]),
            "rectangle": 20 * np.array([[10, 5], [-10, 5], [-10, -5], [10, -5]])}

        selected_shape_key = "original"  # Select the desired shape here
        outline_coords = shapes[selected_shape_key]

    # Remove duplicate last point if present
    if np.allclose(outline_coords[0], outline_coords[-1]):
        outline_coords = outline_coords[:-1]

    # Calculate centroid
    centroid = calculate_centroid(outline_coords)
    print("Calculated centroid (area-weighted):", centroid)

    # Center the shape by subtracting the centroid
    centered_coords = outline_coords - centroid

    # Create outer polygon
    outer_polygon = Polygon(centered_coords)

    # Create inner contour by offsetting the outer polygon inward
    inner_polygon = outer_polygon.buffer(-inner_contour_distance)

    # Validate inner contour
    if inner_polygon.is_empty:
        raise ValueError("Inner contour is empty. The buffer distance may be too large.")

    # Handle multiple polygons resulting from buffer
    if inner_polygon.geom_type == 'MultiPolygon':
        inner_polygon = max(inner_polygon, key=lambda p: p.area)

    inner_coords = np.array(inner_polygon.exterior.coords)

    # Remove duplicate last point from inner contour
    if np.allclose(inner_coords[0], inner_coords[-1]):
        inner_coords = inner_coords[:-1]

    # **Ensure `polygons` is defined for both DXF and preset shapes**
    if dxf_file_path:
        # After centering, assign `polygons` to the centered coordinates
        polygons = [centered_coords]
    else:
        # When using preset shapes, assign `polygons` to include the outer polygon
        polygons = [centered_coords]  # This ensures `polygons` is always defined

    # Calculate internal angles of the inner contour
    internal_angles = calculate_internal_angles(inner_coords)

    # Select gripping points
    try:
        gripping_points = select_gripping_points(
            inner_coords,
            internal_angles,
            centroid=np.array([0.0, 0.0]),
            boundary=inner_polygon.exterior,
            angle_threshold=angle_threshold,
            gripper_radius=gripper_radius,
            verbose=True
        )
    except ValueError as e:
        print(f"Error selecting gripping points: {e}")
        return

    # Create LineString for inner contour
    inner_contour = LineString(np.vstack([inner_coords, inner_coords[0]]))

    # Launch interactive GUI
    interactive_contour_movement(
        inner_contour,
        gripping_points,
        outer_contour=centered_coords,
        gripper_radius=gripper_radius
    )


if __name__ == "__main__":
    # To use a DXF file, provide the path as an argument to main()
    # Example: main(dxf_file_path='path/to/your/file.dxf')
    main()  # Use preset shapes
    #main(dxf_file_path='data/dxf_shapes/original.dxf')
