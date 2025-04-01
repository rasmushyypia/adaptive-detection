import ezdxf
import numpy as np
import logging
import math
from shapely.geometry import Polygon
from shapely.ops import orient
from collections import defaultdict
import matplotlib.pyplot as plt

# --------------------------------------------------
# Arc and Shape Approximators
# --------------------------------------------------
def approximate_arc(center, radius, start_angle, end_angle, num_points=40):
    """
    Approximate an arc by a series of points, ensuring the first and last points
    match the exact geometric endpoints for robust chaining.
    """
    # Convert angles to radians
    start_rad = np.deg2rad(start_angle)
    end_rad = np.deg2rad(end_angle)
    if end_rad <= start_rad:
        end_rad += 2 * np.pi

    # Create angles array
    angles = np.linspace(start_rad, end_rad, num=num_points)

    # Approximate the entire arc
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    arc_points = np.column_stack((x, y))

    # Overwrite first and last with exact endpoints
    arc_points[0] = [
        center[0] + radius * math.cos(start_rad),
        center[1] + radius * math.sin(start_rad)
    ]
    arc_points[-1] = [
        center[0] + radius * math.cos(end_rad),
        center[1] + radius * math.sin(end_rad)
    ]

    return arc_points


def approximate_circle(center, radius, num_points=40):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack((x, y))

def approximate_ellipse(center, major_axis, axis_ratio, rotation_angle, num_points=40):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = major_axis * np.cos(angles)
    y = major_axis * axis_ratio * np.sin(angles)
    if rotation_angle != 0:
        rotation_rad = np.deg2rad(rotation_angle)
        cos_ang = math.cos(rotation_rad)
        sin_ang = math.sin(rotation_rad)
        x_rot = x * cos_ang - y * sin_ang
        y_rot = x * sin_ang + y * cos_ang
        x, y = x_rot, y_rot
    x += center[0]
    y += center[1]
    return np.column_stack((x, y))

def approximate_bulge(p1, p2, bulge, num_points=40):
    if bulge == 0:
        return [p1, p2]

    # Compute chord vector and its length
    chord = np.array(p2) - np.array(p1)
    chord_length = np.linalg.norm(chord)
    
    # Calculate the total sweep angle (can be negative for clockwise arcs)
    angle = 4 * math.atan(bulge)
    abs_angle = abs(angle)
    
    # Compute the positive radius using the absolute angle
    radius = chord_length / (2 * math.sin(abs_angle / 2))
    
    # Midpoint of the chord
    midpoint = (np.array(p1) + np.array(p2)) / 2
    
    # Determine the perpendicular vector:
    # Use 90° counterclockwise rotation for positive bulge,
    # and flip it for negative bulge (to get clockwise rotation).
    perp = np.array([-chord[1], chord[0]])
    if bulge < 0:
        perp = -perp
    perp_unit = perp / np.linalg.norm(perp)
    
    # Offset from midpoint is based on the radius and half the angle.
    offset = radius * math.cos(abs_angle / 2)
    center = midpoint + perp_unit * offset

    # Calculate start angle (from center to p1)
    start_angle = math.atan2(p1[1] - center[1], p1[0] - center[0])
    # Set end angle based on the arc direction
    if bulge < 0:
        end_angle = start_angle - abs_angle
    else:
        end_angle = start_angle + abs_angle

    # Generate arc points from start_angle to end_angle
    angles = np.linspace(start_angle, end_angle, num_points)
    arc_points = [(center[0] + radius * math.cos(a),
                   center[1] + radius * math.sin(a)) for a in angles]
    return arc_points

# --------------------------------------------------
# Endpoint Snapping and Segment Chaining
# --------------------------------------------------
def snap_point(pt, tol=1e-5):
    """Round point coordinates to eliminate numerical noise."""
    return (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)

def chain_segments(segments, tol=1e-6):
    """
    Chains a list of segments (each a list of (x,y) points) together by matching
    endpoints using a snapping tolerance. Returns a list of chained point lists.
    """
    chains = []
    used = [False] * len(segments)
    
    for i, seg in enumerate(segments):
        if used[i]:
            continue
        current_chain = seg[:]  # copy the segment
        used[i] = True
        extended = True
        while extended:
            extended = False
            for j, other in enumerate(segments):
                if used[j]:
                    continue
                # Try connecting the end of current_chain with the start of other
                if snap_point(current_chain[-1], tol) == snap_point(other[0], tol):
                    current_chain.extend(other[1:])
                    used[j] = True
                    extended = True
                    break
                # Or connecting end to other’s end (reverse other)
                elif snap_point(current_chain[-1], tol) == snap_point(other[-1], tol):
                    current_chain.extend(other[-2::-1])
                    used[j] = True
                    extended = True
                    break
                # Also try connecting at the beginning of current_chain
                elif snap_point(current_chain[0], tol) == snap_point(other[-1], tol):
                    current_chain = other[:-1] + current_chain
                    used[j] = True
                    extended = True
                    break
                elif snap_point(current_chain[0], tol) == snap_point(other[0], tol):
                    current_chain = other[1:][::-1] + current_chain
                    used[j] = True
                    extended = True
                    break
        chains.append(current_chain)
    return chains

# --------------------------------------------------
# DXF Extraction and Polygon Reconstruction
# --------------------------------------------------
def extract_polygons_from_dxf(dxf_path, circle_approx_points=80, tol=1e-5, layers=None):
    """
    Reads the DXF file and returns one Shapely Polygon representing:
      - The outer boundary (largest shape)
      - Any holes fully inside that boundary.

    This function preserves entities that are already closed while also
    collecting open segments from entities (for example, non-closed LWPOLYLINEs,
    LINEs, and ARCs that do not span 360°) and attempts to chain them together.
    """
    try:
        doc = ezdxf.readfile(dxf_path)
    except IOError:
        logging.error(f"Cannot open the DXF file: {dxf_path}")
        return None
    except ezdxf.DXFStructureError:
        logging.error(f"Invalid or corrupted DXF file: {dxf_path}")
        return None

    msp = doc.modelspace()

    closed_polys = []  # Arrays of points from entities already closed
    open_segments = [] # Lists of points for open curves to be chained

    for entity in msp:
        if layers and entity.dxf.layer not in layers:
            continue

        dxftype = entity.dxftype()
        pts = None

        if dxftype == 'LWPOLYLINE':
            points = entity.get_points(format='xyb')
            is_closed = entity.closed
            poly_points = []
            num_vertices = len(points)
            for i in range(num_vertices):
                p1 = (points[i][0], points[i][1])
                p2 = (points[(i + 1) % num_vertices][0], points[(i + 1) % num_vertices][1])
                bulge = points[i][2]
                if bulge == 0:
                    if not poly_points:
                        poly_points.append(p1)
                    poly_points.append(p2)
                else:
                    arc_pts = approximate_bulge(p1, p2, bulge, 
                                                num_points=int(circle_approx_points / 10))
                    # Skip duplicate connecting point
                    poly_points.extend(arc_pts[1:])
            if is_closed or np.linalg.norm(np.array(poly_points[0]) - np.array(poly_points[-1])) < tol * 10:
                closed_polys.append(np.array(poly_points))
            else:
                open_segments.append(poly_points)

        elif dxftype == 'POLYLINE':
            if entity.is_3d_polyline:
                continue
            vertices = [tuple(vertex.dxf.location[:2]) for vertex in entity.vertices]
            if entity.is_closed or np.linalg.norm(np.array(vertices[0]) - np.array(vertices[-1])) < tol * 10:
                closed_polys.append(np.array(vertices))
            else:
                open_segments.append(vertices)

        elif dxftype == 'CIRCLE':
            center = (entity.dxf.center.x, entity.dxf.center.y)
            radius = entity.dxf.radius
            circle_poly = approximate_circle(center, radius, num_points=circle_approx_points).tolist()
            closed_polys.append(np.array(circle_poly))

        elif dxftype == 'ARC':
            center = (entity.dxf.center.x, entity.dxf.center.y)
            radius = entity.dxf.radius
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            # Determine the sweep of the arc
            sweep = (end_angle - start_angle) % 360
            arc_poly = approximate_arc(center, radius, start_angle, end_angle, 
                                       num_points=circle_approx_points).tolist()
            # If the arc spans nearly 360°, treat it as closed; otherwise as open.
            if abs(sweep - 360) < 1e-3:
                closed_polys.append(np.array(arc_poly))
            else:
                open_segments.append(arc_poly)

        elif dxftype == 'ELLIPSE':
            center = (entity.dxf.center.x, entity.dxf.center.y)
            major_axis_vector = entity.dxf.major_axis
            major_axis_length = math.hypot(major_axis_vector[0], major_axis_vector[1])
            axis_ratio = entity.dxf.axis_ratio
            rotation_angle = math.degrees(math.atan2(major_axis_vector[1],
                                                     major_axis_vector[0]))
            ellipse_poly = approximate_ellipse(center, major_axis_length, axis_ratio,
                                               rotation_angle,
                                               num_points=circle_approx_points).tolist()
            closed_polys.append(np.array(ellipse_poly))

        elif dxftype == 'LINE':
            seg = [(entity.dxf.start.x, entity.dxf.start.y),
                   (entity.dxf.end.x, entity.dxf.end.y)]
            open_segments.append(seg)

    # Also handle LINE entities via query if any (this may add duplicates, so it is optional)
    lines = list(msp.query('LINE'))
    if lines:
        for line in lines:
            seg = [(line.dxf.start.x, line.dxf.start.y),
                   (line.dxf.end.x, line.dxf.end.y)]
            open_segments.append(seg)

    # Attempt to chain open segments into closed loops.
    chained = chain_segments(open_segments, tol=tol)
    for chain in chained:
        gap = np.linalg.norm(np.array(chain[0]) - np.array(chain[-1]))
        # Force closure if gap is less than a fixed threshold (e.g., 20 units)
        if gap < max(tol * 10, 120):
            # Optionally, append the first point if it's not already there.
            if gap > 0:
                chain.append(chain[0])
            closed_polys.append(np.array(chain))

    # Convert all candidate arrays to Shapely Polygons (using buffer(0) to heal minor issues)
    shapely_polygons = []
    for arr in closed_polys:
        if len(arr) < 3:
            continue
        try:
            poly_obj = Polygon(arr).buffer(0)
            if not poly_obj.is_empty and poly_obj.is_valid and poly_obj.area > 0:
                shapely_polygons.append(poly_obj)
        except Exception as e:
            logging.warning(f"Error constructing polygon: {e}")
            continue

    if not shapely_polygons:
        return None

    # Identify the largest polygon as the main outer boundary.
    main_poly = max(shapely_polygons, key=lambda p: p.area)

    # Identify holes: any polygon fully contained in the main polygon.
    holes = []
    for p in shapely_polygons:
        if p is main_poly:
            continue
        if p.within(main_poly):
            holes.append(p.exterior.coords[:])
        else:
            logging.warning("Found a polygon not inside the largest polygon; skipping it.")

    # Build the final polygon with holes.
    final_poly = Polygon(main_poly.exterior.coords[:], holes=holes)
    final_poly = orient(final_poly, sign=1.0)  # Ensure standard orientation (CCW outer, CW holes)
    return final_poly

# --------------------------------------------------
# Main Function and Visualization
# --------------------------------------------------
def main():
    """
    Reads a DXF file, reconstructs the polygon by combining closed entities
    and by chaining open segments, and visualizes the resulting shape.
    """
    # Change the file path below to test different DXF files.
    # For example, "gasket_rectangle_small.dxf" works as before,
    # while "gasket_custom_2.dxf" now should chain open segments properly.
    dxf_path = "data/dxf_shapes/gasket_custom_2.dxf"  # Update path as needed
    final_poly = extract_polygons_from_dxf(dxf_path, circle_approx_points=80, tol=1)

    if final_poly is None:
        print("No valid polygon found in the DXF.")
        return

    print(f"Outer boundary area: {final_poly.area:.2f}")
    print(f"Number of holes: {len(final_poly.interiors)}")
    for i, hole in enumerate(final_poly.interiors, start=1):
        hole_area = Polygon(hole).area
        print(f"  Hole #{i} area: {hole_area:.2f}")

    # Visualization of the resulting shape.
    fig, ax = plt.subplots()
    x, y = final_poly.exterior.xy
    ax.plot(x, y, 'b-', linewidth=2, label="Outer Boundary")
    ax.fill(x, y, alpha=0.3, fc='blue', ec='blue')

    for idx, interior in enumerate(final_poly.interiors, start=1):
        xh, yh = zip(*interior.coords)
        ax.plot(xh, yh, 'r-', linewidth=2, label=f"Hole {idx}" if idx == 1 else None)
        ax.fill(xh, yh, alpha=0.3, fc='red', ec='red')

    ax.set_aspect('equal', 'box')
    ax.set_title("DXF Shape Reconstruction")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
