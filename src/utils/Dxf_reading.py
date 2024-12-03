# dxf_reading.py

import ezdxf
import numpy as np
import logging
from collections import defaultdict
import math

def approximate_arc(center, radius, start_angle, end_angle, num_points=40):
    start_rad = np.deg2rad(start_angle)
    end_rad = np.deg2rad(end_angle)
    if end_rad <= start_rad:
        end_rad += 2 * np.pi
    angles = np.linspace(start_rad, end_rad, num=num_points)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack((x, y))

def approximate_circle(center, radius, num_points=40):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack((x, y))

def approximate_ellipse(center, major_axis, axis_ratio, rotation_angle, num_points=40):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = major_axis * np.cos(angles)
    y = (major_axis * axis_ratio) * np.sin(angles)

    # Apply rotation if necessary
    if rotation_angle != 0:
        rotation_rad = np.deg2rad(rotation_angle)
        cos_angle = np.cos(rotation_rad)
        sin_angle = np.sin(rotation_rad)
        x_rot = x * cos_angle - y * sin_angle
        y_rot = x * sin_angle + y * cos_angle
        x, y = x_rot, y_rot

    x += center[0]
    y += center[1]
    return np.column_stack((x, y))

def approximate_bulge(p1, p2, bulge, num_points=40):
    if bulge == 0:
        return [p1, p2]

    angle = 4 * math.atan(bulge)
    chord = np.array(p2) - np.array(p1)
    chord_length = np.linalg.norm(chord)
    radius = chord_length / (2 * math.sin(angle / 2))
    midpoint = (np.array(p1) + np.array(p2)) / 2
    perp = np.array([-chord[1], chord[0]])

    if bulge < 0:
        perp = -perp

    perp_length = radius * math.cos(angle / 2)
    perp_unit = perp / np.linalg.norm(perp)
    center = midpoint + perp_unit * perp_length

    start_angle = math.atan2(p1[1] - center[1], p1[0] - center[0])
    end_angle = start_angle + angle

    angles = np.linspace(start_angle, end_angle, num_points)
    arc_points = [(center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)) for a in angles]

    return arc_points

def extract_polygons_from_dxf(dxf_path, circle_approx_points=80, layers=None):
    try:
        doc = ezdxf.readfile(dxf_path)
    except IOError:
        logging.error(f"Cannot open the DXF file: {dxf_path}")
        return []
    except ezdxf.DXFStructureError:
        logging.error(f"Invalid or corrupted DXF file: {dxf_path}")
        return []

    msp = doc.modelspace()
    polygons = []

    for entity in msp:
        if layers and entity.dxf.layer not in layers:
            continue

        if entity.dxftype() == 'LWPOLYLINE':
            points = entity.get_points(format='xyb')
            num_vertices = len(points)
            is_closed = entity.closed
            if not is_closed:
                continue

            polygon_points = []
            for i in range(num_vertices):
                p1 = (points[i][0], points[i][1])
                p2 = (points[(i + 1) % num_vertices][0], points[(i + 1) % num_vertices][1])
                bulge = points[i][2]

                if bulge == 0:
                    if not polygon_points:
                        polygon_points.append(p1)
                    polygon_points.append(p2)
                else:
                    arc_pts = approximate_bulge(p1, p2, bulge, num_points=int(circle_approx_points / 10))
                    polygon_points.extend(arc_pts[1:])

            polygons.append(np.array(polygon_points))

        elif entity.dxftype() == 'POLYLINE':
            if entity.is_3d_polyline:
                continue
            vertices = [tuple(vertex.dxf.location[:2]) for vertex in entity.vertices]
            is_closed = entity.is_closed

            if is_closed:
                polygons.append(np.array(vertices))

        elif entity.dxftype() == 'CIRCLE':
            center = (entity.dxf.center.x, entity.dxf.center.y)
            radius = entity.dxf.radius
            circle_poly = approximate_circle(center, radius, num_points=circle_approx_points)
            polygons.append(circle_poly)

        elif entity.dxftype() == 'ARC':
            center = (entity.dxf.center.x, entity.dxf.center.y)
            radius = entity.dxf.radius
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            arc_poly = approximate_arc(center, radius, start_angle, end_angle, num_points=circle_approx_points)
            polygons.append(arc_poly)

        elif entity.dxftype() == 'ELLIPSE':
            center = (entity.dxf.center.x, entity.dxf.center.y)
            major_axis_vector = entity.dxf.major_axis
            major_axis_length = math.hypot(major_axis_vector[0], major_axis_vector[1])
            axis_ratio = entity.dxf.axis_ratio
            rotation_angle = math.degrees(math.atan2(major_axis_vector[1], major_axis_vector[0]))
            ellipse_poly = approximate_ellipse(center, major_axis_length, axis_ratio, rotation_angle, num_points=circle_approx_points)
            polygons.append(ellipse_poly)

    # Handle individual LINE entities
    lines = list(msp.query('LINE'))
    if lines:
        polygons_from_lines = reconstruct_polygons_from_lines(lines)
        polygons.extend(polygons_from_lines)

    return polygons

def reconstruct_polygons_from_lines(lines, tolerance=1e-5):
    def round_point(pt):
        return (round(pt[0], 5), round(pt[1], 5))

    point_map = defaultdict(list)
    for line in lines:
        start = round_point((line.dxf.start.x, line.dxf.start.y))
        end = round_point((line.dxf.end.x, line.dxf.end.y))
        point_map[start].append(end)
        point_map[end].append(start)

    polygons = []
    visited_edges = set()

    for line in lines:
        start = round_point((line.dxf.start.x, line.dxf.start.y))
        end = round_point((line.dxf.end.x, line.dxf.end.y))
        edge = tuple(sorted([start, end]))
        if edge in visited_edges:
            continue

        current_polygon = [start, end]
        visited_edges.add(edge)
        current_point = end

        while True:
            neighbors = point_map[current_point]
            next_point = None
            for neighbor in neighbors:
                potential_edge = tuple(sorted([current_point, neighbor]))
                if potential_edge not in visited_edges:
                    next_point = neighbor
                    visited_edges.add(potential_edge)
                    break

            if next_point is None:
                break
            if next_point == current_polygon[0]:
                current_polygon.append(next_point)
                break

            current_polygon.append(next_point)
            current_point = next_point

        if len(current_polygon) > 2:
            if current_polygon[0] != current_polygon[-1]:
                current_polygon.append(current_polygon[0])
            polygons.append(np.array(current_polygon))

    return polygons

def main():
    """
    Main function to extract polygons from a DXF file and output them.
    """
    DXF_FILE_PATH = 'dxf_shapes/rounded_rectangle.dxf'  # Update with your DXF file path

    # Extract polygons from DXF
    polygons = extract_polygons_from_dxf(
        dxf_path=DXF_FILE_PATH,
        circle_approx_points=80,
        layers=None  # Set to specific layers if needed
    )

    if not polygons:
        print("No polygons were extracted from the DXF file.")
        return

    # For simplicity, we'll use the first polygon extracted
    outline_coords = polygons[0]

    # Remove duplicate last point if present
    #if np.array_equal(outline_coords[0], outline_coords[-1]):
    #    outline_coords = outline_coords[:-1]
    
    centroid = np.mean(outline_coords, axis=0)
    print("Calculated centroid:", centroid)
    
    if np.allclose(outline_coords[0], outline_coords[-1]):
        outline_coords = outline_coords[:-1]

    centroid = np.mean(outline_coords, axis=0)
    print("Calculated centroid:", centroid)

    # Option 1: Save the outline_coords to a file in NumPy format
    np.save('outline_coords.npy', outline_coords)
    print("Outline coordinates saved to 'outline_coords.npy'.")

    # Option 2: Print the outline_coords to the console in a format that can be copied
    print("\nCopy the following 'outline_coords' array into your main program if needed:\n")
    print("outline_coords = np.array([")
    for coord in outline_coords:
        print(f"    [{coord[0]}, {coord[1]}],")
    print("])\n")

if __name__ == "__main__":
    main()
