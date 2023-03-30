import datetime
import errno
import os
import time

from math import sqrt
from numba import jit

import numpy as np
from OCC.BOPAlgo import BOPAlgo_Section
from OCC.BRep import BRep_Tool, BRep_Builder
from OCC.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
from OCC.TopAbs import TopAbs_FACE, TopAbs_VERTEX
from OCC.TopExp import TopExp_Explorer
from OCC.TopLoc import TopLoc_Location
from OCC.TopoDS import TopoDS_Shape, topods_Face, TopoDS_Compound, topods_Vertex
from OCC.BRep import BRep_Tool_Pnt
from OCC.gp import gp_Pln, gp_Pnt, gp_Dir

from CColors import CColors
from OCCUtils import get_boundingbox
from OCCUtils.face import Face
from ZGetter import ZGetter
from data_io.ImportType import ImportType
from data_io.LDGCode import LDGCode
from data_io.exporters import generate_mesh
from gcode.gcode_visualizer.GCodeLayerC import GCodeLayerC
from gcode.gcode_visualizer.GCodeLine import GCodeLine
from gcode.gcode_visualizer.VRepRapMove import VRepRapMove
from gcode.gcode_visualizer.VRepRapStates import VRepRapStates
from globals import ANGULAR_DEFLECTION, LINEAR_DEFLECTION
from utils import transform_shape


def update_curr_glob_zmax(line: GCodeLine, prev_glob_zmax: float):
    """
    Update the current highest z value -> used for lifted travel to only lift above global max of last layer
    """
    if line.move.z_e > prev_glob_zmax:
        return line.move.z_e
    else:
        return prev_glob_zmax


def generate_curved_layer_christl(part_shape: TopoDS_Shape, preform_gcode: LDGCode, output_file_path: str,
                                  preform_shape=None, max_lin_deflection=LINEAR_DEFLECTION,
                                  ang_deflection=ANGULAR_DEFLECTION, min_segment_length=0.2, max_extrusion_err=0.5,
                                  lifted_travel_dist=2, low_trav_clearance=0.5, high_trav_clearance=1,
                                  max_len_direct_trav=2, compute_normals=True) -> LDGCode:
    """
    Curved Layer Fused Filament Fabrication (CLFFF) algorithm developed by Wolfgang Christl.

    - Utilizes triangulation to compute locations for z adjustment.
    - Takes length change of adjusted G-Code line into account for extrusion rate calculation
    - Creates collision free traveling moves for the nozzle
    - Uses error function of right Riemann sum to calculate line subdivision
    - Optional preform geometry slicing enables concave CLFFF with discontinuities

    Three types of travels are possible for max speed:

    - Direct travel: No lift of nozzle as curvature between endpoints is not of concern
    - Low travel: Lift to slightly above target point (clearance user defined ~0.5mm) as no higher peak is expected
      between start- & endpoint
    - High travel: Lift above highest point in current layer (clearance user defined ~1mm)

    :param part_shape: Geometry of the original curved part
    :param preform_gcode: Loaded G-Code of the preform
    :param output_file_path: Path where the final G-Code file will be written
    :param preform_shape: Optional preform geometry as TopoDS_Shape to get local layer count and enable concave CLFFF
    :param max_lin_deflection: Maximum linear deflection between final output and original shape (see triangulation)
    :param ang_deflection: Maximal angular deflection between final output and original shape (see triangulation)
    :param max_extrusion_err: Maximum over- or under-extrusion on plane sections with increasing section height
    :param min_segment_length: Minimum allowable segment length. All print paths will at least be that long
    :param lifted_travel_dist: Travels shorter than that will only get a slight lift in z direction (low lifted travel)
    :param high_trav_clearance: Clearance in z direction in case of a high lifted travel (above max height of layer)
    :param low_trav_clearance: Clearance in z direction in case of a low lifted travel
    :param max_len_direct_trav: Max length of a travel move so that it will not be converted to a lifted travel (direct)
    :param compute_normals: Compute the normal vector for each line segment
    :return:
    """
    print(CColors.OKGREEN + "------ Generating curved layers (Wolfgang Christl) ------" + CColors.ENDC)
    projected_edges_comp = get_triang(max_lin_deflection, ang_deflection, part_shape)
    z_getter = ZGetter(None, a_loaded_shape=part_shape)
    preform_getter = None
    if preform_shape is not None:
        print("\tSlicing preform...")
        assert isinstance(preform_shape, TopoDS_Shape)
        _, _, min_z_pref, _, _, _ = get_boundingbox(preform_shape, use_mesh=True)
        preform_shape = transform_shape(preform_shape, [0, 0, -int(min_z_pref+0.5), 0, 0, 0])  # to match the G-Code on z=0
        preform_getter = ZGetter(None, generate_sliced_preform(preform_shape, preform_gcode))

    prev_line = GCodeLine("; Init line")
    prev_line.move = VRepRapMove(0, 0, 0, 0, 0, 0, 0, VRepRapStates.STANDBY)
    prev_line.layer_height = 0
    prev_line.layer_num = 0
    avg_preform_layer_height = preform_gcode.gcode_layers[-1].layer_z / preform_gcode.layer_cnt

    list_curved_layers = []
    curr_glob_zmax = preform_gcode.gcode_layers[0].layer_z  # init with first layer height
    i = 1
    print("\tGenerating curved layers...")
    for layer in preform_gcode.gcode_layers:
        print(f"\r\t\tProcessing layer {i}/{preform_gcode.layer_cnt}", end="", flush=True)
        # assert isinstance(layer, GCodeLayerC)
        layer_compound = gcode_layer_to_edge_compound(layer)
        shape_intersections = find_intersections(projected_edges_comp, layer_compound)
        pnt_array_adjustments = np.array(get_vertices_as_pnts(shape_intersections))
        segmented_gcode_lines = []
        for line in layer.gcode_lines:
            # assert isinstance(line, GCodeLine)
            if line.move is not None and line.move.state is VRepRapStates.PRINTING and pnt_array_adjustments.size > 0:
                pnts_on_segment = find_pnts_on_line_bb(line.move, pnt_array_adjustments)
                if pnts_on_segment:
                    ordered_pnts_on_segment = get_ordered_pnts(pnts_on_segment, (line.move.x_s, line.move.y_s))
                    list_new_moves = split_segment(line.move, ordered_pnts_on_segment)
                    for _move in list_new_moves:
                        new_line = GCodeLine()
                        new_line.set_move(_move)
                        new_line.set_layer(line.layer_num)
                        new_line.set_layer_height(line.layer_height)
                        new_line.set_layer_z(line.layer_z)
                        new_line.comment = line.comment
                        layer_count, z_lowest_layer = get_layer_count(preform_getter, new_line, preform_gcode.layer_cnt)
                        list_displaced_segs, prev_line = displace_z_prev(new_line, prev_line, z_getter, layer_count,
                                                                         z_lowest_layer, avg_preform_layer_height,
                                                                         min_seg_length=min_segment_length,
                                                                         max_abs_extrusion_err=max_extrusion_err,
                                                                         compute_normal=compute_normals)
                        segmented_gcode_lines.extend(list_displaced_segs)
                        if len(list_displaced_segs) > 0:
                            curr_glob_zmax = update_curr_glob_zmax(list_displaced_segs[-1], curr_glob_zmax)
                else:
                    layer_count, z_lowest_layer = get_layer_count(preform_getter, line, preform_gcode.layer_cnt)
                    list_displaced_segs, prev_line = displace_z_prev(line, prev_line, z_getter, layer_count,
                                                                     z_lowest_layer, avg_preform_layer_height,
                                                                     max_abs_extrusion_err=max_extrusion_err,
                                                                     compute_normal=compute_normals)
                    segmented_gcode_lines.extend(list_displaced_segs)
                    if len(list_displaced_segs) > 0:
                        curr_glob_zmax = update_curr_glob_zmax(list_displaced_segs[-1], curr_glob_zmax)
            elif line.move is not None and line.move.state is VRepRapStates.TRAVELING:
                # travel move: Generate offset travel to not collide with part
                layer_count, z_lowest_layer = get_layer_count(preform_getter, line, preform_gcode.layer_cnt)
                if line.move.get_length_xy() > max_len_direct_trav:
                    segmented_gcode_lines.extend(generate_lifted_travel(line, prev_line, z_getter, curr_glob_zmax,
                                                                        layer_count, z_lowest_layer,
                                                                        avg_preform_layer_height,
                                                                        min_distance=lifted_travel_dist,
                                                                        low_travel_clearance=low_trav_clearance,
                                                                        high_travel_clearance=high_trav_clearance))
                else:  # do not generate a lifted travel for simple nozzle lift - just adjust z endpoint - direct travel
                    line = displace_z(line, z_getter, layer_count, z_lowest_layer, avg_preform_layer_height)
                    line.move.z_s = prev_line.move.z_e  # for data integrity also adjust start point
                    segmented_gcode_lines.extend([line])
                prev_line = segmented_gcode_lines[-1]
            else:
                # must be some other G-Code we do not want to adjust -> just pass through
                segmented_gcode_lines.append(line)
        layer.gcode_lines = segmented_gcode_lines  # set the new lines
        list_curved_layers.append(layer)
        i += 1
    list_curved_layers = add_algorithm_notice(list_curved_layers)
    print("\n\tWriting curved layer G-Code to: " + output_file_path)
    write_gcode_to_file(output_file_path, list_curved_layers, digits=4)
    print(CColors.OKGREEN + "\tDone!" + CColors.ENDC)
    curved_layer_gcode = LDGCode(None, [], "CL G-Code", ImportType.gcode, output_file_path)
    curved_layer_gcode.set_list_gcode_layers(list_curved_layers)
    curved_layer_gcode.extrusion_dim = preform_gcode.extrusion_dim
    return curved_layer_gcode


def get_layer_count(preform_getter_instance: ZGetter, curr_line: GCodeLine, fallback_cnt: int) -> tuple:
    """
    Gets the local layer count of the sliced preform at x, y end point of current G-Code line

    :param preform_getter_instance: ZGetter instance with loaded sliced preform
    :param curr_line: Current G-Code line
    :param fallback_cnt: In case no preform geometry was supplied or local layer count is zero we fall back to this
        value. This must be the total layer count stored inside the loaded G-Code
    :return: Number of layers at endpoint x, y position of current line in case a preform geometry was supplied
    """
    if preform_getter_instance is not None:
        return preform_getter_instance.count_layers(curr_line.move.x_e, curr_line.move.y_e)
    else:
        return fallback_cnt, None


def generate_sliced_preform(_preform_shape: TopoDS_Shape, _gcode_preform: LDGCode) -> TopoDS_Shape:
    """
    Creates a geometry consisting of layers/faces that represent the sliced preform shape as described by the G-Code

    :param _preform_shape: TopoDS_Shape of the preform
    :param _gcode_preform: G-Code of the preform
    :return: A shape consisting of faces for each layer in the G-Code
    """
    _compound = TopoDS_Compound()
    a_builder = BRep_Builder()
    a_builder.MakeCompound(_compound)
    for _layer in _gcode_preform.gcode_layers:
        # assert isinstance(_layer, GCodeLayerC)
        plane = gp_Pln(gp_Pnt(0., 0., _layer.layer_z), gp_Dir(0., 0., 1.))
        face = BRepBuilderAPI_MakeFace(plane).Shape()
        a_builder.Add(_compound, face)
    common_shape = BRepAlgoAPI_Common(_preform_shape, _compound)
    common_shape.SetRunParallel(True)
    return common_shape.Shape()


def get_triang(max_lin_deflection, ang_deflection, part_shape, ignore_flat=True) -> TopoDS_Compound:
    """
    Computes the triangulation of a given OCCT geometry. Triangles are converted to lines. Triangles in the xy-plane
    are ignored since no change in curvature. Lines get projected to xy-plane. Duplicates are removed.
    Resulting geometry consists of edges in the xy-plane. Every line represents the points at which intersecting nozzle
    moves need to make a change in vertical direction to approximate the part accordingly

    :param max_lin_deflection: maximal linear deflection of triangulation to geometry
    :param ang_deflection: maximal angular deflection of triangulation to geometry
    :param part_shape: OCCT geometry to be triangulated
    :param ignore_flat: Ignore triangles in xy-plane
    :return: OCCT geometry consisting of the extracted lines from triangulation in xy-plane
    """
    triangulated_mesh = generate_mesh(max_lin_deflection, ang_deflection, part_shape)
    print("\tIterating over mesh... ", end="", flush=True)
    ex = TopExp_Explorer(triangulated_mesh, TopAbs_FACE)
    list_edges = []  # helper list to detect duplicates
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    while ex.More():
        F = topods_Face(ex.Current())
        # ignore faces parallel to or with normal in XY-Layer -> no adjustment of G-Code anyways -> speedup
        if ignore_flat:
            _Face = Face(F)
            _mid_u_v, _ = _Face.mid_point()
            gp_dir_normal = _Face.DiffGeom.normal(_mid_u_v[0], _mid_u_v[1])
            if (gp_dir_normal.X() == 0 and gp_dir_normal.Y() == 0) or gp_dir_normal.Z() == 0:
                ex.Next()
                continue
        L = TopLoc_Location()
        facing = BRep_Tool().Triangulation(F, L).GetObject()
        tab = facing.Nodes()
        tri = facing.Triangles()
        for i in range(1, facing.NbTriangles() + 1):
            trian = tri.Value(i)
            index1, index2, index3 = trian.Get()
            for j in range(1, 4):
                if j == 1:
                    M = index1
                    N = index2
                elif j == 2:
                    N = index3
                elif j == 3:
                    M = index2
                # exclude zero length edges
                if not ((tab.Value(M).X() == tab.Value(N).X() and tab.Value(M).Y() == tab.Value(N).Y()) or
                        (tab.Value(M).X() == tab.Value(N).Y() and tab.Value(M).Y() == tab.Value(N).X())):
                    l1 = (
                    float(tab.Value(M).X()), float(tab.Value(M).Y()), float(tab.Value(N).X()), float(tab.Value(N).Y()))
                    l2 = (
                    float(tab.Value(N).X()), float(tab.Value(N).Y()), float(tab.Value(M).X()), float(tab.Value(M).Y()))
                    # Exclude duplicate edges and their mirrors l2
                    if l1 not in list_edges and l2 not in list_edges:
                        list_edges.append(l1)
                        # project to XY-plane
                        pnt1 = tab.Value(M)
                        pnt2 = tab.Value(N)
                        pnt1.SetZ(0)
                        pnt2.SetZ(0)
                        bb = BRepBuilderAPI_MakeEdge(pnt1, pnt2)
                        if bb.IsDone():
                            builder.Add(comp, bb.Edge())
        ex.Next()
    print(f"{CColors.OKBLUE}Done!{CColors.ENDC}")
    return comp


def gcode_layer_to_edge_compound(n_layer: GCodeLayerC) -> TopoDS_Compound:
    """
    Convert a G-Code layer containing nozzle moves into a OCCT geometry object that consists projected 2D points
    (xy-plane). Every nozzle printing move is converted to a TopoDS_Edge (line). All lines are grouped into a
    TopoDS_Compound

    :param n_layer: The G-Code layer containing the GCodeLines that you want to convert.
    :return: A OCCT geometry object consisting of all nozzle moves as edges. Each move projected into the xy-plane
    """
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    for line in n_layer.gcode_lines:
        # assert isinstance(line, GCodeLine)
        if line.move is not None and line.move.state is VRepRapStates.PRINTING:
            n_edge = line.move.get_topods_edge_projected()
            if n_edge is not None:
                builder.Add(comp, n_edge)
    return comp


def find_intersections(triangulation: TopoDS_Compound, printer_paths: TopoDS_Compound) -> TopoDS_Shape:
    """
    Compute a intersection between two geometries using the OCCT API.
    Used to get the intersection points between the triangulation and the printers extrusion paths

    :param triangulation: Geometry 1
    :param printer_paths: Geometry 2
    :return: Geometry containing the section geometry (vertices in case edges are supplied)
    """
    section_algo = BOPAlgo_Section()
    section_algo.AddArgument(printer_paths)
    section_algo.AddArgument(triangulation)
    section_algo.SetRunParallel(True)
    section_algo.Perform()
    return section_algo.Shape()


def get_vertices_as_pnts(intersection_shape: TopoDS_Shape) -> list:
    """
    Convert all vertices of a given geometry to a list of 2D points (projected on xy-plane)

    :param intersection_shape: Geometry that contains the vertices
    :return: List of 2D points each representing the projection of a vertex onto the xy-plane
    """
    ex = TopExp_Explorer(intersection_shape, TopAbs_VERTEX)
    list_pnts = []
    while ex.More():
        v = ex.Current()
        p = BRep_Tool_Pnt(topods_Vertex(v))
        list_pnts.append((p.X(), p.Y()))
        ex.Next()
    return list_pnts


@jit(nopython=True)
def is_pnt_on_segment(s_x, s_y, e_x, e_y, p_x, p_y, epsilon=1e-6):
    """
    Computes whether a 2D point lies on the given segment. Uses tolerance.

    :param s_x: x coordinate of start point of segment
    :param s_y: y coordinate of start point of segment
    :param e_x: x coordinate of end point of segment
    :param e_y: y coordinate of end point of segment
    :param p_x: x coordinate of point to test
    :param p_y: y coordinate of point to test
    :param epsilon: Tolerance
    :return: True if point is on segment
    """
    # TODO: Vectorize
    cross_product = (p_y - s_y) * (e_x - s_x) - (p_x - s_x) * (e_y - s_y)
    if -epsilon < cross_product < epsilon and \
            min(s_x, e_x) <= p_x <= max(s_x, e_x) and min(s_y, e_y) <= p_y <= max(s_y, e_y):
        return True
    else:
        return False


@jit
def find_pnts_on_line(_move: VRepRapMove, pnts: np.ndarray) -> list:
    """
    Deprecated. Slower than find_pnts_on_line_bb().
    Finds all points within a given set that lay on the given VrepRapMove. Uses brute force search method. Tests all
    points directly.

    :param _move: A line (move of the printer nozzle)
    :param pnts: 2D point set (x, y) with points that you want to test
    :return: A subset of points that lie on the line
    """
    # TODO: Vectorize
    list_pnts_on_segment = []
    for pnt in pnts:
        if is_pnt_on_segment(_move.x_s, _move.y_s, _move.x_e, _move.y_e, pnt[0], pnt[1]):
            list_pnts_on_segment.append((pnt[0], pnt[1]))
    return list(set(list_pnts_on_segment))  # remove duplicates


@jit
def find_pnts_on_line_bb(_move: VRepRapMove, pnts: np.ndarray) -> list:
    """
    Does the same as find_pnts_on_line(). Finds all points within a given set that lay on the given VrepRapMove.
    It uses a bounding box search method. Fist find all points that are inside of bounding box of the given move.
    Check which of these points actually are on the line in next step.
    ~41% faster than find_pnts_on_line() on same data set.

    :param _move: A nozzle movement representing the current extrusion line we want to search for points that are on it
    :param pnts: All points that exist
    :return: List of tuples representing 2d points (x, y)
    """
    _tol = 1e-2
    _list_pnts_on_segment = []
    if _move.x_s < _move.x_e:
        min_x = _move.x_s
        max_x = _move.x_e
    else:
        min_x = _move.x_e
        max_x = _move.x_s
    if _move.y_s < _move.y_e:
        min_y = _move.y_s
        max_y = _move.y_e
    else:
        min_y = _move.y_e
        max_y = _move.y_s
    ll = np.array([min_x - _tol, min_y - _tol])  # lower-left
    ur = np.array([max_x + _tol, max_y + _tol])  # upper-right
    pnts_in_bb = pnts[np.all(np.logical_and(ll <= pnts, pnts <= ur), axis=1)]
    for pnt in pnts_in_bb:
        if is_pnt_on_segment(_move.x_s, _move.y_s, _move.x_e, _move.y_e, pnt[0], pnt[1]):
            _list_pnts_on_segment.append((pnt[0], pnt[1]))
    return list(set(_list_pnts_on_segment))  # remove duplicates


def get_ordered_pnts(pnts: list, start_pnt_segment: tuple) -> list:
    """
    Order a list of points according to their distance to a given point. First point in resulting list is closest to
    supplied point

    :param pnts: List of 2D points to order by distance every point is a tuple (x, y)
    :param start_pnt_segment: 2D point to compute the distance to
    :return: Ordered list of 2D points with first point being closest to supplied start point
    """
    pnts.sort(key=lambda p: sqrt((p[0] - start_pnt_segment[0]) ** 2 + (p[1] - start_pnt_segment[1]) ** 2))
    return pnts


def split_segment(move: VRepRapMove, _ordered_pnts_on_segment: list, list_complete=False) -> list:
    """
    Split a segment/line/nozzle move at a number of positions given by a list of 2D/3D points on the segment.

    :param move: line/nozzle move
    :param _ordered_pnts_on_segment: Ordered list of points (see get_ordered_pnts()) to split the move at.
    :param list_complete: You are sure that _ordered_pnts_on_segment is correct and all necessary points (start, end)
        are inside. Deactivates a check that can cause problems when numerical precision causes minimal deviations within
        split_line_equally
    :return: List of lines/nozzle moves representing the supplied move split at the given locations
    """
    _list_new_segs = []
    is_2d = len(_ordered_pnts_on_segment[0]) < 3  # determine if we got 2D or 3D tuples/points
    if is_2d:
        length_old_move = move.get_length_xy()
        abs_start_pnt = (move.x_s, move.y_s)
        abs_end_pnt = (move.x_e, move.y_e)
    else:
        length_old_move = move.get_length_3d()
        abs_start_pnt = (move.x_s, move.y_s, move.z_s)
        abs_end_pnt = (move.x_e, move.y_e, move.z_e)
    if not list_complete:
        # sometimes the start/end point is already inside the list because the section algo determined it before (unusual)
        # or it is called from split equally where start and endpoint are already inside the list
        if abs_start_pnt not in _ordered_pnts_on_segment:
            _ordered_pnts_on_segment.insert(0, abs_start_pnt)
        if abs_end_pnt not in _ordered_pnts_on_segment:
            _ordered_pnts_on_segment.append(abs_end_pnt)

    num_splits = len(_ordered_pnts_on_segment) - 1
    # i = 0 to (num_splits-1)
    for i in range(num_splits):
        # new move from i to i+1
        start_pnt = _ordered_pnts_on_segment[i]
        end_pnt = _ordered_pnts_on_segment[i + 1]
        if is_2d:
            new_move = VRepRapMove(start_pnt[0], start_pnt[1], move.z_s, end_pnt[0], end_pnt[1], move.z_e, move.speed,
                                   move.state)
            new_move.set_extrusion_rate(move.extrusion_rate * (new_move.get_length_xy() / length_old_move))
        else:
            new_move = VRepRapMove(start_pnt[0], start_pnt[1], start_pnt[2], end_pnt[0], end_pnt[1], end_pnt[2],
                                   move.speed, move.state)
            new_move.set_extrusion_rate(move.extrusion_rate * (new_move.get_length_3d() / length_old_move))
        _list_new_segs.append(new_move)
    return _list_new_segs


def generate_lifted_travel(_line: GCodeLine, _prev_line: GCodeLine, _zgetter_inst: ZGetter, zmin_pass: float,
                           preform_layer_cnt: int, z_lowest_layer_preform, avg_layer_height_preform, min_distance=5,
                           low_travel_clearance=0.5, high_travel_clearance=1) -> list:
    """
    Generates a lifted travel move that is supposed to replace the supplied move/line. New move consists of three
    single movements.

        1. Nozzle up
        2. Traverse
        3. Nozzle down

    This is supposed to stop the print head from crashing into the printed part by (almost) always traveling at a
    height above the printed part. Only works with absolute XYZ coordinates right now!

    :param _line: The current line/travel move that will be replaced
    :param _prev_line: The previous line/travel move
    :param _zgetter_inst: ZGetter instance
    :param zmin_pass: Minimum height the travel has to pass on a long/high travel move
    :param preform_layer_cnt: Number of layers of the Preform G-Code
    :param high_travel_clearance: Clearance between part and extrusion on a lifted high travel
    :param z_lowest_layer_preform: Z coordinate of the lowest layer of the sliced preform
    :param avg_layer_height_preform: Average layer height of the sliced preform gcode
    :param low_travel_clearance: Clearance in z direction in case we go for a low lifted travel
    :param min_distance: If length of travel is <= this value the nozzle will only lift slightly above the destination
        point of the nozzle movement
    :return: List consisting of three GCodeLines representing the lifted travel move
    """
    list_lifted_travel_moves = []
    # Zero values will be overwritten in last step
    lift_move = VRepRapMove(_line.move.x_s, _line.move.y_s, _prev_line.move.z_e, _line.move.x_s, _line.move.y_s,
                            0, _line.move.speed, VRepRapStates.NOZZLE_LIFT)
    lifted_move = VRepRapMove(_line.move.x_s, _line.move.y_s, 0, _line.move.x_e, _line.move.y_e,
                              0, _line.move.speed, VRepRapStates.LIFTED_TRAVELING)
    lower_move = VRepRapMove(_line.move.x_e, _line.move.y_e, 0, _line.move.x_e, _line.move.y_e,
                             _line.move.z_e, _line.move.speed, VRepRapStates.NOZZLE_LOWER)
    lift_line = GCodeLine(None)
    lift_line.set_move(lift_move)
    lift_line.set_layer(_line.layer_num)
    lift_line.comment = "lift nozzle"

    lifted_line = GCodeLine(None)
    lifted_line.set_move(lifted_move)
    lifted_line.set_layer(_line.layer_num)
    lifted_line.comment = "lifted travel"

    lower_line = GCodeLine(None)
    lower_line.set_move(lower_move)
    lower_line.set_layer(_line.layer_num)
    lower_line.set_layer_z(_line.layer_z)  # for correct varying layer count displacement calculation
    lower_line.set_layer_height(_line.layer_height)  # for correct varying layer count displacement calculation
    lower_line.comment = "lower nozzle"
    # adjust z value to match extrusion end and new extrusion start
    lower_line = displace_z(lower_line, _zgetter_inst, preform_layer_cnt, z_lowest_layer_preform,
                            avg_layer_height_preform)

    if _line.move.get_length_xy() > min_distance:
        lift_dest = zmin_pass + high_travel_clearance  # part clearance when high travel
    else:
        # lift nozzle only few mm up in case we got a very short travel
        # depends on uphill or downhill move
        if lower_line.move.z_e > lift_line.move.z_s:
            lift_dest = lower_line.move.z_e + low_travel_clearance
            lifted_line.comment = f"lifted travel (s up)"  # short travel uphill
        else:
            lift_dest = lift_line.move.z_s + low_travel_clearance
            lifted_line.comment = f"lifted travel (s down)"  # short travel downhill

    lift_line.move.z_e = lift_dest
    lifted_line.move.z_s = lift_dest
    lifted_line.move.z_e = lift_dest
    lower_line.move.z_s = lift_dest

    list_lifted_travel_moves.append(lift_line)
    list_lifted_travel_moves.append(lifted_line)
    list_lifted_travel_moves.append(lower_line)
    return list_lifted_travel_moves


def split_line_equally(_a_line: GCodeLine, _num_new_segments: int, extrusion_rate_start: int) -> list:
    """
    Problem: Linear move from point a to b where the extrusion rate needs to increase/decrease from a to b. Not possible
    with a single G-Code line since E-value is linear interpolated along the line/movement
    Solution: Split line into sub-segments with gradually adjusted extrusion rates (interpolation)

    This function splits a line into equidistant segments and interpolates the extrusion rates. Returns a list of
    GCodeLines that replace the input line.

    :param _a_line: GCodeLine to be split
    :param _num_new_segments: Number of resulting split segments/lines
    :param extrusion_rate_start: Extrusion rate that needs to be set at the beginning of the line. Extrusion rate at
        the end of the line is optained from _a_line.move
    :return: List of GCodeLines to replace the input line
    """
    list_new_line_segments = []
    p1 = _a_line.move.start_to_ndarray()
    p2 = _a_line.move.end_to_ndarray()
    l1 = np.linspace(0, 1, _num_new_segments + 1)
    # recompute extrusion: constant linear increase/decrease along the original (non-split) line
    extrusion_rates = np.linspace(extrusion_rate_start, _a_line.move.extrusion_rate, _num_new_segments)
    # create start/end points for new equally long segments (interpolation)
    segments = p1 + (p2 - p1) * l1[:, None]
    segments = list(map(tuple, segments))
    list_split_moves = split_segment(_a_line.move, segments, list_complete=True)
    if _num_new_segments != len(list_split_moves):
        print("Oh shit!")
    i = 0
    for _move in list_split_moves:
        _move.set_extrusion_rate(extrusion_rates[i] / _num_new_segments)
        _move.set_normal(_a_line.move.get_normal())
        _l = GCodeLine()
        _l.comment = f"{_a_line.comment} split"
        _l.set_move(_move)
        _l.set_layer(_a_line.layer_num)
        _l.set_layer_height(_a_line.layer_height)  # set layer height of last segment to all of them, wrong but enough
        list_new_line_segments.append(_l)
        i += 1
    return list_new_line_segments


def displace_z(a_line: GCodeLine, z_getter_instance: ZGetter, local_layer_cnt: int, z_lowest_layer_preform: float,
               avg_layer_height_preform: float) -> GCodeLine:
    """
    Reduced version of the displace_z_prev() method. Only use when you want to displace end point and not require any
    extra stuff like extrusion adjustment, splitting or special movement detection like NOZZLE_LIFT etc.
    Used for generating lifted travel moves.

    When we later convert the move to some actual G-Code we will only need the end point of the move. That is why we
    only calculate the z position of that one (end point).

    :param a_line:
    :param z_getter_instance:
    :param local_layer_cnt:
    :param z_lowest_layer_preform: Z value of the lowest layer at the current position (x, y).
        Can be None if no support for vaying layer height needed
    :param avg_layer_height_preform: Average layer height in preform G-Code e.g. 0.3 - only set if
        z_lowest_layer_preform != None
    :return:
    """
    z_upper, z_lower = z_getter_instance.get_z(a_line.move.x_e, a_line.move.y_e)
    if z_upper is not None:
        extrusion_height = (z_upper - z_lower) / local_layer_cnt
        if z_lowest_layer_preform is None:
            a_line.move.z_e = z_lower + extrusion_height * (a_line.layer_num + 1)  # layer num starts at 0
        else:
            # support for varying local layer count in preform geometry
            a_line.move.z_e = z_lower + extrusion_height * calculate_local_layer_index(a_line.layer_z,
                                                                                       z_lowest_layer_preform,
                                                                                       avg_layer_height_preform)
        a_line.set_layer_height(extrusion_height)
        # if a_line.move.state is VRepRapStates.PRINTING:
        #     # Change of extrusion rate due to change of layer thickness
        #     a_line.move.extrusion_rate *= extrusion_height / a_line.layer_height  # FH version
        #     # Change of extrusion rate due to z adjustment
        #     # Change of segment length is ~linear in case of delta_z >= max(delta_x, delta_y)
        #     # Christl requires start and endpoint of extrusion to be set
        #     a_line.move.extrusion_rate *= a_line.move.get_length_3d() / a_line.move.get_length_xy()
    return a_line


def calculate_local_layer_index(z_preform_current_layer: int, z_lowest_local_layer: float,
                                const_layer_height_preform: float) -> int:
    """
    Calculates the local layer index of the preform G-Code

    :param z_preform_current_layer: Z coordinate of current layer in preform G-Code
    :param z_lowest_local_layer: Z coordinate of lowest layer in preform G-Code
    :param const_layer_height_preform: Average individual layer height in preform e.g. 0.3
    :return: Local layer count form 1 to total layer count
    """
    return int(((z_preform_current_layer - z_lowest_local_layer)/const_layer_height_preform) + 0.5) + 1  # round to +int


def displace_z_prev(curr_line: GCodeLine, prev_line: GCodeLine, z_getter_instance: ZGetter, local_layer_cnt: int,
                    z_lowest_layer_preform: float, avg_layer_height_preform: float, min_seg_length=0.2,
                    max_abs_extrusion_err=0.5, compute_normal=False) -> (list, GCodeLine):
    """
    Z-coordinate of start and endpoint of VRepRapMove are displaced.
    Recalculation of extrusion rate according to:

        - Layer height change due to z-displacement
        - Length change due to z-displacement
        - Extrusion ramp from start to end point required: Split of input line into multiple segments
        - Support for varying layer count in preform G-Code using local layer index

    Splitting of the line in to n segments: n=(_delta_h * _length2d * _length3d)/(2*max_abs_extrusion_err) with n>1
    _delta_h is the layer height change from start to end point. Formula is derived from error estimation formula of
    right Riemann sum.

    Current position (x, y) is the end point of the move

    :param curr_line: Input line containing the current VRepRapMove and layer number
    :param prev_line: Input line containing the previous VRepRapMove and layer number
    :param z_getter_instance: Instance of ZGetter
    :param local_layer_cnt: Total number of layers at the current position (x, y)
    :param z_lowest_layer_preform: Z value of the lowest layer at the current position (x, y).
        Can be None if no support for vaying layer height needed
    :param avg_layer_height_preform: Average layer height in preform G-Code e.g. 0.3 - only set if
        z_lowest_layer_preform != None
    :param min_seg_length: Minimal segment length when segment is split
    :param max_abs_extrusion_err: Maximum extrusion error in mm²/Segment according to right Riemann sum error
    :param compute_normal: Compute normal vectors for every move based on its end point
    :return: List of GCodeLines where the z coordinate is displaced to follow the shape of the object loaded with
     ZGetter, Updated previous move (in most cases the current move)
    """
    if prev_line.move.state is not VRepRapStates.PRINTING:
        # In case the previous line is a travel we need to compute the layer height at the start point of the curr_line
        z_upper_s, z_lower_s = z_getter_instance.get_z(curr_line.move.x_s, curr_line.move.y_s)
        if z_upper_s is not None:
            extrusion_height_s = (z_upper_s - z_lower_s) / local_layer_cnt  # new layer height at start point
        else:
            extrusion_height_s = curr_line.layer_height  # error fallback, might generate shit further down
    else:
        extrusion_height_s = prev_line.layer_height

    # Overwrite in case actual prev line was ignored
    curr_line.move.x_s = prev_line.move.x_e
    curr_line.move.y_s = prev_line.move.y_e

    if compute_normal:
        z_upper, z_lower, n_upper, n_lower = z_getter_instance.get_z_normals(curr_line.move.x_e, curr_line.move.y_e)
    else:
        z_upper, z_lower = z_getter_instance.get_z(curr_line.move.x_e, curr_line.move.y_e)
    if z_upper is not None and z_upper != z_lower:
        extrusion_height = (z_upper - z_lower) / local_layer_cnt  # new layer height at end point
        if curr_line.move.state is not VRepRapStates.NOZZLE_LOWER:
            curr_line.move.z_s = prev_line.move.z_e
        if curr_line.move.state is not VRepRapStates.NOZZLE_LIFT:
            if z_lowest_layer_preform is None:
                curr_line.move.z_e = z_lower + extrusion_height * (curr_line.layer_num + 1)  # layer num starts at 0
            else:
                # support for varying local layer count in preform geometry
                lli = calculate_local_layer_index(curr_line.layer_z, z_lowest_layer_preform, avg_layer_height_preform)
                curr_line.move.z_e = z_lower + extrusion_height * lli
                curr_line.local_layer_cnt = local_layer_cnt
                curr_line.local_layer_indx = lli
                curr_line.comment = f"{curr_line.comment} llc: {local_layer_cnt} lli: {lli}"
        # curr_line.set_layer_height(extrusion_height)
        if curr_line.move.state is VRepRapStates.PRINTING:
            # Calculate before overwrite
            extrusion_rate_start = curr_line.move.extrusion_rate * extrusion_height_s / extrusion_height
            _length3d = curr_line.move.get_length_3d()
            if _length3d < min_seg_length:
                # print("Deleted some")
                return [], prev_line  # ignore the current line/movement
            _length2d = curr_line.move.get_length_xy()

            # Change of extrusion rate due to change of layer thickness (new/old) extrusion height
            curr_line.move.extrusion_rate *= extrusion_height / curr_line.layer_height  # FH/Hackaday version
            # Change of extrusion rate due to z adjustment
            # Change of segment length is ~linear in case of delta_z >= max(delta_x, delta_y)
            curr_line.move.extrusion_rate *= _length3d / _length2d  # Christl
            curr_line.set_layer_height(extrusion_height)  # overwrite layer height with new one
            if compute_normal:
                _normal = interpolate_normal(dir_to_numpy(n_lower), dir_to_numpy(n_upper), z_upper, z_lower,
                                             curr_line.move.z_e)
                curr_line.move.set_normal((_normal[0], _normal[1], _normal[2]))

            # Check if we got a linear layer_height change and split line so that extrusion rate can be adjusted
            _delta_h = abs(extrusion_height - extrusion_height_s)
            if _delta_h > 0:
                # calculate/est. the number of segments we should split the current move to get a linear-like extrusion
                # over height change [Assumption: Extrusions are rectangular]
                pref_seg_count = int(sqrt((_length2d / (2 * max_abs_extrusion_err)) * _delta_h) + 1.5)
                if pref_seg_count > 1:
                    while pref_seg_count > 0 and (_length3d / pref_seg_count) < min_seg_length:
                        pref_seg_count -= 1
                    if pref_seg_count > 1:
                        _list_split_line = split_line_equally(curr_line, pref_seg_count, extrusion_rate_start)
                        return _list_split_line, _list_split_line[-1]
    return [curr_line], curr_line


def dir_to_numpy(direction: gp_Dir) -> np.ndarray:
    return np.array([direction.X(), direction.Y(), direction.Z()])


def interpolate_normal(lower_normal: np.ndarray, upper_normal: np.ndarray, z_upper: float, z_lower: float,
                       z_layer: float) -> np.ndarray:
    """
    Linear interpolation from lower normal vector to upper normal vector. Calculates normal vector for current
    layer index.

    :param lower_normal: 3D Normal vector as numpy array of upper surface normal
    :param upper_normal:  3D normal vector as numpy array of lower surface normal
    :param z_upper: Upper z coordinate of final form hull at XY
    :param z_lower: Lower z coordinate of final form hull at XY
    :param z_layer: Z coordinate of current layer
    :return: Normal vector at XY position of upper/lower normal and Z position of current layer
    """
    t = (z_layer - z_lower) / (z_upper - z_lower)
    return t * upper_normal + (1-t) * lower_normal


def write_gcode_to_file(output_file_path: str, curved_layers: list, digits=3, debug_out=False):
    if not os.path.exists(os.path.dirname(output_file_path)):
        try:
            os.makedirs(os.path.dirname(output_file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    x_changed = True
    y_changed = True
    z_changed = True
    prev_x = None
    prev_y = None
    prev_z = None
    with open(output_file_path, 'w+', encoding='utf-8') as new_file:
        for _cl in curved_layers:
            # assert isinstance(_cl, GCodeLayerC)
            for _line in _cl.gcode_lines:
                # assert isinstance(_line, GCodeLine)
                if _line.move is not None:
                    x_changed = prev_x is None or prev_x != _line.move.x_e or debug_out
                    y_changed = prev_y is None or prev_y != _line.move.y_e or debug_out
                    z_changed = prev_z is None or prev_z != _line.move.z_e or debug_out
                    prev_x = _line.move.x_e
                    prev_y = _line.move.y_e
                    prev_z = _line.move.z_e
                new_file.write(_line.return_regenerated_gcode(digits, x_changed=x_changed, y_changed=y_changed,
                                                              z_changed=z_changed))


def add_algorithm_notice(all_layers: list) -> list:
    ts = time.time()
    timestamp_str = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S %d.%m.%Y')
    all_layers[0].gcode_lines. \
        insert(0, GCodeLine("; Created by Real3DFFF on " + timestamp_str + " using CLFFF (Christl)\n\n"))
    return all_layers
