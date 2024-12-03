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


def select_gripping_points_symmetrical(coords, angles, centroid, boundary, angle_threshold = 150.0, gripper_radius = 1.0,
                                       exclude_y_axis = True, exclude_x_axis = False, verbose = False):
    """
    Select gripping points ensuring symmetry over the Y-axis by dividing the shape into quadrants.

    Parameters:
    - coords (np.ndarray): Array of shape (n, 2) representing polygon vertices.
    - angles (np.ndarray): Array of internal angles in degrees corresponding to the vertices.
    - centroid (np.ndarray): Centroid of the shape.
    - boundary (LineString): Boundary of the inner contour.
    - angle_threshold (float): Maximum angle to consider a vertex significant.
    - gripper_radius (float): Radius of the gripper to enforce minimum distance.
    - exclude_y_axis (bool): Exclude vertices on the Y-axis.
    - exclude_x_axis (bool): Exclude vertices on the X-axis.
    - verbose (bool): If True, prints debug messages.

    Returns:
    - np.ndarray: Array of selected gripping points.
    - np.ndarray: Array of candidate points after filtering.
    """
    # Filter polygon's vertices based on angle_threshold and exclude axis flags
    mask = np.ones(len(coords), dtype=bool)
    if exclude_y_axis:
        mask &= coords[:, 0] != 0
    if exclude_x_axis:
        mask &= coords[:, 1] != 0
    mask &= angles < angle_threshold

    filtered_coords = coords[mask]

    if verbose:
        print(f"[DEBUG] Candidate points after filtering: {len(filtered_coords)}")

    if len(filtered_coords) < 4:
        raise ValueError("Not enough candidate points to select gripping points.")

    # Assign points to quadrants
    quadrants = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}
    for point in filtered_coords:
        x, y = point
        if x > 0 and y >= 0:
            quadrants['Q1'].append(point)
        elif x < 0 and y >= 0:
            quadrants['Q2'].append(point)
        elif x < 0 and y < 0:
            quadrants['Q3'].append(point)
        elif x > 0 and y < 0:
            quadrants['Q4'].append(point)
        # Handle points on axes
        if x == 0 and y > 0:
            quadrants['Q1'].append(point)
            quadrants['Q2'].append(point)
        elif x == 0 and y < 0:
            quadrants['Q3'].append(point)
            quadrants['Q4'].append(point)
        elif y == 0 and x > 0:
            quadrants['Q1'].append(point)
            quadrants['Q4'].append(point)
        elif y == 0 and x < 0:
            quadrants['Q2'].append(point)
            quadrants['Q3'].append(point)

    if verbose:
        print("[DEBUG] Quadrant Points after Filtering:")
        for q, pts in quadrants.items():
            print(f"  {q}: {pts}")

    def select_gripping_point(points):
        """
        Select and adjust a gripping point within a quadrant.

        Parameters:
        - points (list): List of points in the quadrant.

        Returns:
        - np.ndarray: Adjusted gripping point.
        """
        if not points:
            return None

        if len(points) == 1:
            return adjust_point_to_contour(points[0], centroid, boundary)

        elif len(points) == 2:
            midpoint = np.mean(points, axis=0)
            return adjust_point_to_contour(midpoint, centroid, boundary)

        else:
            # Select the midpoint between the two furthest points
            max_dist = 0
            pair = (points[0], points[0])
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dist = np.linalg.norm(points[i] - points[j])
                    if dist > max_dist:
                        max_dist = dist
                        pair = (points[i], points[j])
            midpoint = (pair[0] + pair[1]) / 2
            return adjust_point_to_contour(midpoint, centroid, boundary)

    gripping_points = []
    # Iterate through each quadrant, selecting a gripping point using the select_gripping_point function.
    for quadrant, points in quadrants.items():
        point = select_gripping_point(points)
        if point is not None:
            gripping_points.append(point)
        else:
            if verbose:
                print(f"[DEBUG] No candidate points in {quadrant}")

    gripping_points = np.array(gripping_points)

    if len(gripping_points) != 4:
        raise ValueError("Expected 4 gripping points (one from each quadrant), but got a different number.")

    # Ensure symmetry across the Y-axis
    G1, G2, G3, G4 = gripping_points
    gripping_points[1] = np.array([-G1[0], G1[1]])
    gripping_points[3] = np.array([-G3[0], G3[1]])

    if verbose:
        print("\n[DEBUG] Final Selected Gripping Points:")
        for i, point in enumerate(gripping_points, 1):
            print(f"  G{i}: {point}")

    # Validate minimum distances
    distances = {
        "G1-G2": np.linalg.norm(gripping_points[0] - gripping_points[1]),
        "G3-G4": np.linalg.norm(gripping_points[2] - gripping_points[3])
    }

    for pair, distance in distances.items():
        if distance < 2 * gripper_radius:
            raise ValueError(f"Gripping points {pair} are too close: {distance:.2f} < {2 * gripper_radius}")

    return gripping_points, filtered_coords


def move_point_along_contour(inner_contour, point, step_size = 0.5, direction = 'clockwise'):
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

def compute_maximum_offset(safe_polygon, outer_polygon, tolerance=1e-4):
    """
    Compute all maximum offset distances from the outer polygon to the safe polygon,
    within a specified tolerance.

    Parameters:
    - safe_polygon (Polygon): The safe quadrilateral polygon.
    - outer_polygon (Polygon): The outer fabric outline polygon.
    - tolerance (float): The allowable difference from the maximum distance to consider.

    Returns:
    - max_distance (float): The maximum offset distance.
    - boundary_points_coords (List[np.ndarray]): Coordinates of boundary points with maximum offset.
    - closest_points_coords (List[np.ndarray]): Coordinates of the closest points on the safe polygon.
    """
    # Sample points on the outer boundary
    boundary_points = np.array(outer_polygon.exterior.coords)
    boundary_points = [Point(bp) for bp in boundary_points]

    # Compute distances from boundary points to the safe quadrilateral
    distances = [safe_polygon.exterior.distance(bp) for bp in boundary_points]

    # Find the maximum distance
    max_distance = max(distances)

    # Identify all points within the tolerance of the maximum distance
    boundary_points_max = [bp for bp, d in zip(boundary_points, distances) if (max_distance - d) <= tolerance]

    closest_points_max = [
        safe_polygon.exterior.interpolate(safe_polygon.exterior.project(bp)).coords[0]
        for bp in boundary_points_max
    ]

    # Convert to numpy arrays
    boundary_points_coords = [np.array([bp.x, bp.y]) for bp in boundary_points_max]
    closest_points_coords = [np.array(cp) for cp in closest_points_max]

    return max_distance, boundary_points_coords, closest_points_coords


def interactive_contour_movement_symmetrical(inner_contour, gripping_points, outer_contour=None, gripper_radius=1.0):
    """
    Create an interactive GUI to move gripping points along the inner contour symmetrically.

    Parameters:
    - inner_contour (LineString): Shapely LineString representing the closed inner contour.
    - gripping_points (np.ndarray): Array of current gripping points (G1, G2, G3, G4).
    - outer_contour (np.ndarray): Array of outer contour vertices for visualization.
    - gripper_radius (float): Radius representing the extent of each gripper.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(right=0.79)  # Reserve space on the right side

    # Initialize a list to store maximum offset lines
    max_offset_lines = []

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

    # Create safe quadrilateral encompassing gripping areas
    def update_safe_quadrilateral():
        nonlocal safe_quadrilateral, safe_fill, safe_area, area_annotation, max_offset_annotation, max_offset_lines
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

        # Compute maximum offsets
        max_offset, boundary_points, closest_points = compute_maximum_offset(safe_polygon, outer_polygon)

        # Update maximum offset annotation
        max_offset_annotation.set_position((safe_polygon.centroid.x, safe_polygon.centroid.y - 20))
        max_offset_annotation.set_text(f'Max Offset: {max_offset:.2f}')

        # Remove existing max offset lines
        for line in max_offset_lines:
            line.remove()
        max_offset_lines.clear()

        # Plot new max offset lines
        max_offset_label_added = False  # Initialize the flag
        for bp, cp in zip(boundary_points, closest_points):
            if not max_offset_label_added:
                line, = ax.plot([bp[0], cp[0]], [bp[1], cp[1]],
                                'r--', linewidth=2, label='Max Offset')
                max_offset_label_added = True  # Set the flag to True after adding the label
            else:
                line, = ax.plot([bp[0], cp[0]], [bp[1], cp[1]],
                                'r--', linewidth=2)
            max_offset_lines.append(line)

        plt.draw()

    # Compute initial safe quadrilateral and maximum offsets
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

    # Compute and annotate maximum offset
    max_offset, boundary_point, closest_point = compute_maximum_offset(safe_polygon, outer_polygon)
    max_offset_annotation = ax.annotate(f'Max Offset: {max_offset:.2f}',
                                        (centroid_safe.x, centroid_safe.y - 20),
                                        textcoords="offset points", xytext=(0, 0),
                                        ha='center', color='black', fontsize=14, weight='bold',
                                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # Initialize maximum offset lines
    max_offset_label_added = False  # Initialize the flag
    if isinstance(boundary_point, np.ndarray) and isinstance(closest_point, np.ndarray):
        # Single maximum offset
        line, = ax.plot([boundary_point[0], closest_point[0]],
                        [boundary_point[1], closest_point[1]],
                        'r--', linewidth=2, label='Max Offset')
        max_offset_lines.append(line)
        max_offset_label_added = True  # Set the flag since label is added
    elif isinstance(boundary_point, list) and isinstance(closest_point, list):
        # Multiple maximum offsets
        for bp, cp in zip(boundary_point, closest_point):
            if not max_offset_label_added:
                line, = ax.plot([bp[0], cp[0]],
                                [bp[1], cp[1]],
                                'r--', linewidth=2, label='Max Offset')
                max_offset_label_added = True  # Set the flag since label is added
            else:
                line, = ax.plot([bp[0], cp[0]],
                                [bp[1], cp[1]],
                                'r--', linewidth=2)
            max_offset_lines.append(line)

    # Legend and plot settings
    ax.legend(loc='upper left', bbox_to_anchor=(1,1.01))
    ax.axis('equal')
    ax.set_title('Interactive Symmetrical Gripping Area Visualization')
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.grid(True)

    # Slider for step size, placed at the top-right
    axcolor = 'lightgoldenrodyellow'
    slider_step = Slider(ax=plt.axes([0.955, 0.68, 0.03, 0.2], facecolor=axcolor),  # x, y, width, height
                        label='Step Size', valmin=0.1, valmax=2.0, valinit=1, valstep=0.1, color='blue',
                        orientation='vertical')  # Vertical slider for cleaner appearance

    # G1 buttons, side by side, below the slider
    button_g1_ccw = Button(plt.axes([0.80, 0.57, 0.095, 0.06]), 'G1 Up', color=axcolor, hovercolor='0.975')
    button_g1_cw = Button(plt.axes([0.9, 0.57, 0.095, 0.06]), 'G1 Down', color=axcolor, hovercolor='0.975')

    # G4 buttons, side by side, below G1 buttons
    button_g4_ccw = Button(plt.axes([0.80, 0.50, 0.095, 0.06]), 'G4 Up', color=axcolor, hovercolor='0.975')
    button_g4_cw = Button(plt.axes([0.9, 0.50, 0.095, 0.06]), 'G4 Down', color=axcolor, hovercolor='0.975')

    # Optimize and Reset buttons, stacked at the bottom
    button_opt = Button(plt.axes([0.80, 0.40, 0.195, 0.08]), 'Optimize Grippers', color='lightgreen', hovercolor='0.975')
    button_reset = Button(plt.axes([0.80, 0.31, 0.195, 0.08]), 'Reset', color=axcolor, hovercolor='0.975')
    
    button_save = Button(plt.axes([0.80, 0.22, 0.195, 0.08]), 'Save Data', color='lightblue', hovercolor='0.975')

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

    def move_gripper(index, direction):
        """
        Move a gripping point in the specified direction.

        Parameters:
        - index (int): Index of the gripping point (0 for G1, 3 for G4).
        - direction (str): 'clockwise' or 'counterclockwise'.
        """
        step = slider_step.val
        primary_point = gripping_points[index].copy()
        new_point = move_point_along_contour(inner_contour, primary_point, step, direction)
        gripping_points[index] = new_point

        # Update symmetric point based on symmetry relationships
        if index == 0:  # G1
            symmetric_index = 1  # G2
            gripping_points[symmetric_index] = np.array([-new_point[0], new_point[1]])  # Symmetry for G2
        elif index == 3:  # G4
            symmetric_index = 2  # G3
            gripping_points[symmetric_index] = np.array([-new_point[0], new_point[1]])  # Symmetry for G3
        else:
            raise ValueError("Invalid index. Only G1 (0) and G4 (3) can be moved directly.")

        # Update annotations
        for i in [index, symmetric_index]:
            annotations[i].remove()
            annotations[i] = ax.annotate(f'G{i+1}', (gripping_points[i][0], gripping_points[i][1]),
                                        textcoords="offset points", xytext=(0, 10),
                                        ha='center', color='red')

            # Update gripper patches
            gripper_patches[i].center = (gripping_points[i][0], gripping_points[i][1])

        # Update scatter plot
        gripping_scatter.set_offsets(gripping_points)

        # Update safe quadrilateral and maximum offset lines
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

        # Update safe quadrilateral and maximum offset lines
        update_safe_quadrilateral()

        plt.draw()

    def optimize_grippers(event):
        """
        Optimize the positions of the gripping points to maximize the safe area.
        """
        contour_length = inner_contour.length

        def get_point(s):
            s = s % contour_length
            point = inner_contour.interpolate(s)
            return np.array([point.x, point.y])

        # Objective function for optimization
        def objective(s):
            s1, s4 = s  # Optimize s1 (G1) and s4 (G4)
            G1 = get_point(s1)
            G4 = get_point(s4)
            G2 = np.array([-G1[0], G1[1]])  # Symmetry for G1
            G3 = np.array([-G4[0], G4[1]])  # Symmetry for G4
            temp_points = np.vstack([G1, G2, G3, G4])
            gripper_areas = [Point(pt).buffer(gripper_radius) for pt in temp_points]
            combined_area = unary_union(gripper_areas)
            safe_quad = combined_area.convex_hull
            return -safe_quad.area  # Negative for maximization

        # Constraints to ensure grippers remain separated by at least 2 * gripper_radius
        def constraint_min_distance(s):
            s1, s4 = s  # Optimize s1 (G1) and s4 (G4)
            G1 = get_point(s1)
            G4 = get_point(s4)
            G2 = np.array([-G1[0], G1[1]])  # Symmetry for G1
            G3 = np.array([-G4[0], G4[1]])  # Symmetry for G4
            temp_points = np.vstack([G1, G2, G3, G4])

            # Calculate pairwise distances
            distances = [
                np.linalg.norm(temp_points[i] - temp_points[j]) - 2 * gripper_radius
                for i in range(4) for j in range(i + 1, 4)
            ]
            return distances  # Must be greater than zero

        def get_quadrant_bounds(inner_contour):
            """
            Determine arc-length bounds for G1 (Q1: x > 0, y > 0) and G4 (Q4: x > 0, y < 0).

            Returns:
            - bounds_s1: Tuple of bounds for s1 (Q1).
            - bounds_s4: Tuple of bounds for s4 (Q4).
            """
            contour_length = inner_contour.length
            s_values = np.linspace(0, contour_length, 1000)  # Sample 1000 points along the contour

            # Determine bounds for Q1 (x > 0, y > 0)
            q1_bounds = []
            for s in s_values:
                point = inner_contour.interpolate(s)
                if point.x > 0 and point.y > 0:
                    q1_bounds.append(s)

            # Determine bounds for Q4 (x > 0, y < 0)
            q4_bounds = []
            for s in s_values:
                point = inner_contour.interpolate(s)
                if point.x > 0 and point.y < 0:
                    q4_bounds.append(s)

            # Return the min and max of each valid range
            return (min(q1_bounds), max(q1_bounds)), (min(q4_bounds), max(q4_bounds))

        # Initial guess
        s1_initial = inner_contour.project(Point(gripping_points[0]))  # G1
        s4_initial = inner_contour.project(Point(gripping_points[3]))  # G4
        initial_guess = np.array([s1_initial, s4_initial])

        # Bounds
        bounds_s1, bounds_s4 = get_quadrant_bounds(inner_contour)
        bounds = [bounds_s1, bounds_s4]

        # Constraints
        constraints = {
            'type': 'ineq',
            'fun': constraint_min_distance
        }

        # Separate optimization configurations
        def optimize_with_slsqp():
            return minimize(
                objective,
                initial_guess,
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
                x0=initial_guess,
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

        result = optimize_with_slsqp()
        #result = optimize_with_trust_constr()

        if result.success:
            optimized_s = result.x
            G1_opt = get_point(optimized_s[0])
            G4_opt = get_point(optimized_s[1])
            G2_opt = np.array([-G1_opt[0], G1_opt[1]])  # Symmetry for G1
            G3_opt = np.array([-G4_opt[0], G4_opt[1]])  # Symmetry for G4
            gripping_points[:] = np.vstack([G1_opt, G2_opt, G3_opt, G4_opt])

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
    button_g4_cw.on_clicked(lambda event: move_gripper(3, 'clockwise'))
    button_g4_ccw.on_clicked(lambda event: move_gripper(3, 'counterclockwise'))
    button_opt.on_clicked(optimize_grippers)
    button_save.on_clicked(save_data)
    button_reset.on_clicked(reset_grippers)

    plt.show()


def main(dxf_file_path=None):
    global polygons, gripping_points
    """
    Main function to execute gripping points selection, movement, optimization, and visualization
    with symmetry by dividing the shape into quadrants and forming a quadrilateral.
    """
    # User-Configurable Parameters
    inner_contour_distance = 11  # Buffer distance for inner contour
    angle_threshold = 172  # Max angle to consider a vertex significant
    gripper_radius = 10  # Radius of each gripper
    exclude_y_axis = False  # Exclude points on the Y-axis
    exclude_x_axis = False  # Exclude points on the X-axis

    if dxf_file_path:
        # Use the DXF file to get the outline
        # Call the function from dxf_reading module
        polygons_extracted = dxf_reading.extract_polygons_from_dxf(
            dxf_path=dxf_file_path,
            circle_approx_points=60,
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
            "rectangle": 10 * np.array([[0, 0], [0, 10], [10, 10], [10, 0], [0,0]])}
        

        selected_shape_key = "rectangle"  # Select the desired shape here
        outline_coords = shapes[selected_shape_key]

    # Remove duplicate last point if present
    if np.allclose(outline_coords[0], outline_coords[-1]):
        outline_coords = outline_coords[:-1]

    # Calculate centroid
    centroid = calculate_centroid(outline_coords)
    print("Calculated centroid (area-weighted):", centroid)

    # Center the shape by subtracting the centroid
    centered_coords = outline_coords - centroid
    print(centered_coords)
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

    # Select gripping points with symmetry
    try:
        gripping_points, filtered_coords = select_gripping_points_symmetrical(
            inner_coords,
            internal_angles,
            centroid=np.array([0.0, 0.0]),  # Since centered
            boundary=inner_polygon.exterior,
            angle_threshold=angle_threshold,
            gripper_radius=gripper_radius,
            exclude_y_axis=exclude_y_axis,
            exclude_x_axis=exclude_x_axis,
            verbose=True
        )
    except ValueError as e:
        print(f"Error selecting gripping points: {e}")
        return

    # Create LineString for inner contour
    inner_contour = LineString(np.vstack([inner_coords, inner_coords[0]]))

    # Launch interactive GUI
    interactive_contour_movement_symmetrical(
        inner_contour,
        gripping_points,
        outer_contour=centered_coords,
        gripper_radius=gripper_radius
    )


if __name__ == "__main__":
    # To use a DXF file, provide the path as an argument to main()
    # Example: main(dxf_file_path='path/to/your/file.dxf')
    #main()  # Use preset shapes
    main(dxf_file_path='data/dxf_shapes/original.dxf')
