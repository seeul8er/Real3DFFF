import multiprocessing as mp
from multiprocessing import Queue
from queue import Empty

import numpy
from OCC.BRep import BRep_Tool_Pnt
from OCC.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Fuse
from OCC.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_Transform, \
    BRepBuilderAPI_MakePolygon
from OCC.BRepGProp import brepgprop_LinearProperties
from OCC.BRepIntCurveSurface import BRepIntCurveSurface_Inter
from OCC.BRepMesh import BRepMesh_IncrementalMesh
from OCC.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.GProp import GProp_GProps
from OCC.Geom import Geom_Line
from OCC.GeomAdaptor import GeomAdaptor_Curve
from OCC.HLRAlgo import HLRAlgo_Projector
from OCC.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCC.ShapeFix import ShapeFix_Shape, ShapeFix_Wire
from OCC.TopAbs import TopAbs_FORWARD
from OCC.TopTools import TopTools_ListOfShape
from OCC.TopoDS import TopoDS_Shape
from OCC.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Lin, gp_Ax2, gp_Trsf
from scipy.spatial import distance, ConvexHull

from CColors import CColors
from ZGetter import ZGetter
from data_io.ImportType import ImportType
from OCCUtils import Topo, get_boundingbox
from OCCUtils.core_geometry_utils import assert_isdone
from OCCUtils.edge import Edge
from OCCUtils.face import Face
from Zone import Zone
from utils import make_extrusion, get_edges, draw_list_of_edges, list_to_compound_shape, edge_vertices_to_numpy_array, \
    transform_shape


def algo_preform(import_filetype: ImportType, _original_shape: TopoDS_Shape, running_os='w',
                 multicore_enabled=False, use_convex_hull=True, display_viewer=None):
    """
    Creates a preform that can be processed by a slicing software to generate G-Code. The preform can be described as a
    extruded shadow/projection on the X/Y layer. The preform has a constant thickness. No curves or concave edges.

    :param display_viewer: Supply the 3D display viewer in case you want to debug the code and display generation stages
    :param multicore_enabled: Enable multi core support (buggy with windows)
    :param running_os: w for windows, l for linux OS
    :param import_filetype: See enum ImportType: The filetype we imported from CAD
    :param _original_shape: The shape we want to create a projection and preform from. (The imported geometry)
    :param use_convex_hull: Use a convex hull algorithm to detect overall outline (not holes etc.; the shape surrounding
     everything) of preform with STL imports. If set to False the find_edge_loop(...) implementation will be used to
     detect all loops/wires/the overall outline
    :return: A TopoDS_Shape representing the preform with a fixed height. Return False if failed
    """
    print("")
    print(CColors.OKGREEN + "--- Generate preform v1" + CColors.ENDC)
    shape_projection = create_projection(_original_shape)
    sfs = ShapeFix_Shape()
    sfs.Init(shape_projection)
    sfs.Perform()
    fused_edge = sfs.Shape()

    print("Detecting shadow outline (cleaning projected geometry) ...", end="")
    _all_edges_of_projection = get_edges(fused_edge)
    _list_edges_shadow_outline = []
    # start = time.time()
    # multi core: find optimal number of parallel processes
    _process_count = mp.cpu_count()
    _chunksize = int(len(_all_edges_of_projection) / _process_count)
    _min_chunksize = 200  # Linux
    if running_os == 'w':
        _min_chunksize = 6000  # Windows is much slower in starting processes --> more overhead! -> more edges/process
    while _chunksize <= _min_chunksize and _process_count > 1:
        _process_count -= 1
        _chunksize = int(len(_all_edges_of_projection) / _process_count)
    if multicore_enabled and len(_all_edges_of_projection) > _chunksize:  # sometimes multi core is not the best option
        print(" using " + str(_process_count) + " processes")
        _my_processes_list = []
        _my_split_edge_list = []  # Split the edges in chucks of data and feed them to processes
        _progress_queue = Queue()  # We find the (newly) done number of edges in here
        _solution_queue = Queue()  # We find the calculation results in here
        # Split the work into equally sized parts and feed to processes
        for i in range(_process_count):
            if i == (_process_count - 1):
                _my_split_edge_list.append(_all_edges_of_projection[(i * _chunksize):])
            else:
                _my_split_edge_list.append(_all_edges_of_projection[(i * _chunksize):((i * _chunksize) + _chunksize)])
            _myp = mp.Process(name=str(i), target=find_shadow_outline, args=(_my_split_edge_list[i], _original_shape,
                                                                             _solution_queue, _progress_queue, i))
            _my_processes_list.append(_myp)
            _myp.start()  # takes so long on windows; super fast on linux --> chunksize on win must be higher
        # Update console
        a_process_seems_to_be_alive = True
        _num_done_all_processes = 0
        while a_process_seems_to_be_alive:
            for _a_process in _my_processes_list:
                try:
                    _num_newly_done_by_process = _progress_queue.get(True, 0.1)
                    _num_done_all_processes += _num_newly_done_by_process
                    print("\r\tDone: " + str(_num_done_all_processes) + "/" + str(len(_all_edges_of_projection))
                          + " edges", end="")
                except Empty:
                    if not _a_process.is_alive():  # enough with linux to kill processes
                        a_process_seems_to_be_alive = False
                        break
                    else:
                        if _num_done_all_processes == len(_all_edges_of_projection):  # needed with windows
                            a_process_seems_to_be_alive = False
                            break
        print("\r\tDone: " + str(len(_all_edges_of_projection)) + "/" + str(len(_all_edges_of_projection)) + " edges")
        # Retain the results form the different processes in correct order otherwise strange bugs may happen with API
        for _ in _my_processes_list:
            _processresult = _solution_queue.get()
            _list_edges_shadow_outline.insert(_processresult[1], _processresult[0])
        _list_edges_shadow_outline = list(numpy.itertools.chain.from_iterable(_list_edges_shadow_outline))
        # Block until all processes have finished
        for job in _my_processes_list:
            job.join()
    else:
        # single core
        print(" using 1 process")
        _list_edges_shadow_outline = find_shadow_outline(_all_edges_of_projection, _original_shape)
    # end = time.time()
    print(CColors.OKBLUE + "Shadow outline generated!" + CColors.ENDC)
    _list_closed_wires = []
    _list_edges_unsued = _list_edges_shadow_outline

    if use_convex_hull and (import_filetype is ImportType.stl):
        # https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
        print("Detecting linked edges: Searching for overall outline (convex hull)...")
        convex_hull = ConvexHull(edge_vertices_to_numpy_array(_list_edges_unsued))
        # Do some shape healing with the API - should not be necessary but why not :)
        sfs = ShapeFix_Wire()
        sfs.Load(convex_hull_to_wire(convex_hull))
        sfs.Perform()
        sfs.FixReorder()
        sfs.SetClosedWireMode(True)
        _list_closed_wires.append(sfs.Wire())
        # remove all edges form list which have a vertex that is already part of the convex hull/overall preform outline
        # THIS STEP DOES WORK BY 50% OR SO. Errors in the next step are very likely with convex hull. But we filter them
        # returns a list of edges that are/should not be part of convex hull. These may are the edges forming holes etc.
        _list_edges_unsued = find_edges_used_in_convex_hull(convex_hull, _list_edges_unsued)

    print("Detecting linked edges ...")
    while len(_list_edges_unsued) > 0:
        # Set the starting edge
        wire_outline, _list_edges_unsued = find_edge_loop(_list_edges_unsued[0], shape_projection, _list_edges_unsued)
        if wire_outline is not None:
            # fix wire
            sfs = ShapeFix_Wire()
            sfs.Load(wire_outline)
            sfs.Perform()
            sfs.FixReorder()
            sfs.SetClosedWireMode(True)
            wire_outline = sfs.Wire()
            # append wire to list
            _list_closed_wires.append(wire_outline)
            print("\tFound a closed loop (wire), " + str(len(_list_edges_unsued)) + " edges are left over")
        else:
            if import_filetype is not ImportType.stl:
                # With .stp|.igs always fall back to .stl if there is an error -> .stp & .igs have no messy geometry
                return False
            # Very likely error with convex hull algorithms and STL in general! Ignore them and just keep on going
            print(CColors.FAIL + "\tError: Could not find a closed edge loop (wire)." + CColors.ENDC)
            # if all edges were processed and we found at least one loop we try to make the best out of it and go on
            if len(_list_closed_wires) > 0 and len(_list_edges_unsued) == 0:
                # .stl import has no fall back option. Try to make the best out of the results we gathered
                print(
                    CColors.WARNING + "\tAll edges tested. Mesh might be messy or very detailed. Trying to build "
                                      "preform from detected loops." + CColors.ENDC)
                _list_edges_unsued = []
            else:
                # remove the starting edge from the list of available edges so it does not get tested again
                # it might be an edge that has no connection to any other edge
                del _list_edges_unsued[0]
                # Go on with the loop till we processed all edges in _list_edges_unsued
            #     print(CColors.FAIL + "Drawing left over edges." + CColors.ENDC)
            #     draw_list_of_edges(_list_edges_unsued, display_Viewer3d)
            #     return False

    print(CColors.OKBLUE + "Edges are assigned to loops. Found " + str(len(_list_closed_wires)) + " loop(s)!"
          + CColors.ENDC)

    print("Generating final geometry ...")
    # BUG: brepgprop_LinearProperties() only works with *.stl imports --> *.stp crashes!
    # BUG: ShapeFix_Shape() only works with .stp and does nothing with .stl!
    # Solution: Recognize the import_filetype -->   STL: brepgprop_LinearProperties() --> find outer wire
    #                                         -->   STP: ShapeFix_Shape() --> auto fix outer/inner wires
    print("\tAnalyzing wires/loops ...")
    _index_longest_wire = 0
    _longest_wire_length = 0
    if import_filetype is ImportType.stl:
        for wire_index, _wire in enumerate(_list_closed_wires):
            props = GProp_GProps()  # API
            brepgprop_LinearProperties(_wire, props)
            if props.Mass() > _longest_wire_length:
                _index_longest_wire = wire_index
                _longest_wire_length = props.Mass()

    if len(_list_closed_wires) > 0:
        print("\tBuilding face ...")
        _Face = BRepBuilderAPI_MakeFace(_list_closed_wires[_index_longest_wire], True)  # API
        if len(_list_closed_wires) > 1:
            for _a_wire in _list_closed_wires:
                _Face.Add(_a_wire)
        sfs = ShapeFix_Shape()  # API
        sfs.Init(_Face.Shape())
        sfs.Perform()
        _shape_face_preform = sfs.Shape()
        print("\tEstimating extrusion height ... ")
        _extrusion_height = estimate_max_cross_section(_original_shape)  # utils.py
        print("\tHeight estimated to " + str(_extrusion_height) + " [units]")
        print("\tExtruding face ...")
        shape_preform = make_extrusion(_shape_face_preform, _extrusion_height)  # utils.py
        print("\tTranslating preform to match original z_min position...")
        xmin, ymin, zmin, _, _, _ = get_boundingbox(_original_shape, use_mesh=True)
        shape_preform = transform_shape(shape_preform, [0.0, 0.0, zmin, 0.0, 0.0, 0.0])
        print(CColors.OKGREEN + "Finished generating preform!" + CColors.ENDC)
        return shape_preform
    else:
        print(CColors.FAIL + "Failed generating preform with algorithm v1" + CColors.ENDC)
        return False


def find_edge_loop(starting_edge, shape_projection, _list_edges_available_from_outline):
    """
    Finds a loop of connected edges in a list of edges

    :param starting_edge: The starting edge of a loop that we want to find
    :param shape_projection: The shape containing all edges
    :param _list_edges_available_from_outline: A list of edges that shall be connected (not all edges may be used)
    :return: A wire that contains of a closed loop of edges, a list of unused edges that where in
        _list_edges_available_from_outline
    """
    still_gotta_shit_to_do = True
    topo_ex = Topo(shape_projection)
    matched_vert_past_edge = None
    _list_edges_a_loop = []
    current_edge = starting_edge
    _list_edges_a_loop.append(starting_edge)
    num_connected = 1
    print("\t\tProcessing " + str(len(_list_edges_available_from_outline)) + " edges")
    while still_gotta_shit_to_do:
        next_edge, matched_vert_past_edge, _index_next_edge = find_next_edge(current_edge,
                                                                             _list_edges_available_from_outline,
                                                                             topo_ex, matched_vert_past_edge)
        if next_edge is None:
            print(CColors.FAIL + "\n\t\tError: Found no next edge! Aborting loop!"
                  + CColors.ENDC)
            return None, _list_edges_available_from_outline
        else:
            if next_edge not in _list_edges_a_loop:
                _list_edges_a_loop.append(next_edge)  # Has common vertex. Add to list if not already in it
                num_connected = num_connected + 1
                print("\r\t\tConnected " + str(num_connected) + " edges to current loop", end="")
            else:
                if next_edge is starting_edge:
                    still_gotta_shit_to_do = False  # We reached the end!
                    print("")
            current_edge = next_edge
            del _list_edges_available_from_outline[_index_next_edge]  # Makes algorithm 50% faster and a lot easier

    # draw_list_of_edges(_list_edges_a_loop)
    _wire_builder = BRepBuilderAPI_MakeWire()  # Build a Wire from ouline edges
    # With multi core we need to use TopTools_ListOfShape() for some reason?!
    _list_of_shapes = TopTools_ListOfShape()
    for edge_outline in _list_edges_a_loop:
        _list_of_shapes.Append(edge_outline)
    _wire_builder.Add(_list_of_shapes)
    # left in _list_edges_available_from_outline is unused and not assigned to a loop!
    return _wire_builder.Wire(), _list_edges_available_from_outline


def find_next_edge(current_edge, _list_edges_outline, topo_ex, old_vertice):
    """
    Call this function to find the next edge to current edge!

    :param current_edge: The current edge for that we want to find the next (neighbouring) edge
    :param _list_edges_outline: A list of all possible next edges
    :param topo_ex: topo_ex = Topo(shape_projection) to init it
    :param old_vertice: Set to None on first iteration. Else it is the vertex that was matched in the prev.
        loop with a vertex of the current edge
    :return: The_next_edge_we_found, the_vertex_of_next_edge_that_matched_vertex_of_current_edge
    """
    _vertex_distance_tolerance = 0.000000001
    _list_vertices_current_edge = list(topo_ex.vertices_from_edge(current_edge))
    _list_possible_next_edges = []

    for index_next_edge, _possible_next_edge in enumerate(_list_edges_outline):
        if current_edge is not _possible_next_edge:
            _list_vertices_next_edge = topo_ex.vertices_from_edge(_possible_next_edge)

            for vertex_current in _list_vertices_current_edge:  # Iteration über vertices der aktuellen Kante
                if (old_vertice is None) or (BRep_Tool_Pnt(vertex_current).SquareDistance(BRep_Tool_Pnt(old_vertice)) >
                                             _vertex_distance_tolerance):
                    for vert_next_edge in _list_vertices_next_edge:  # Iteration durch alle vertices des untersuchten edges

                        distance = BRep_Tool_Pnt(vertex_current).SquareDistance(BRep_Tool_Pnt(vert_next_edge))
                        if distance < _vertex_distance_tolerance:
                            _list_possible_next_edges.append(
                                [distance, _possible_next_edge, vert_next_edge, index_next_edge])
                            # return _possible_next_edge, vert_next_edge, index_next_edge
    if len(_list_possible_next_edges) == 1:
        return _list_possible_next_edges[0][1], _list_possible_next_edges[0][2], _list_possible_next_edges[0][3]
    elif len(_list_possible_next_edges) > 1:
        print(CColors.WARNING + "\n\t\tFound " + str(
            len(_list_possible_next_edges)) + " possible next edges" + CColors.ENDC)
        _list_possible_next_edges.sort(key=lambda _list_possible_next_edges: _list_possible_next_edges[0])
        return _list_possible_next_edges[0][1], _list_possible_next_edges[0][2], _list_possible_next_edges[0][3]
    else:
        return None, None, None


def find_shadow_outline(edges_shape_projection, _loaded_shape, _solution_queue=None, _progress_queue=None,
                        process_index=0):
    """
    Finds the outline of a 2D projection in Z direction. Removes the inner edges (cleanup for algorithm).

    :param edges_shape_projection: The edges that should create the shadow outline
    :param _loaded_shape: The 3D shape that we test each intersection on
    :param _solution_queue: A Queue object. If multi core is enabled we write the result to that
    :param _progress_queue: A Queue object. Writes the done number of edges into it (num edges since last write)
    :param process_index: Only used with multi core enabled. Needed to format the console out correctly
    :return: A list of edges that make up the outline of the shadow of the shape
    """
    _intersection_tol = 0.0000001
    _list_edges_to_remove = []
    _list_edges_to_keep = []

    # start pre-init (speedup)
    _intersection = BRepIntCurveSurface_Inter()
    _intersection.Load(_loaded_shape, _intersection_tol)
    # end pre-init
    _done_counter = 0
    _done_pref = 0
    for edge_an_edge in edges_shape_projection:
        _Edge_an_edge = Edge(edge_an_edge)
        int_parameter, _gp_Pnt_edge_midpoint = _Edge_an_edge.mid_point()
        _gp_Pnt_1, _gp_Pnt_2 = create_measure_points(edge_an_edge, int_parameter, _gp_Pnt_edge_midpoint)

        _list_gp_Pnt_1_intersection_points = []  # intersection points with _gp_Pnt_1 and shape_loaded
        _list_gp_Pnt_2_intersection_points = []  # intersection points with _gp_Pnt_2 and shape_loaded

        _line = gp_Lin(_gp_Pnt_1, gp_Dir(0, 0, 1))
        # look for intersection between loaded shape and line through _gp_Pnt_1
        _intersection.Init(GeomAdaptor_Curve(Geom_Line(_line).GetHandle()))
        while _intersection.More():
            _list_gp_Pnt_1_intersection_points.append(_intersection.Pnt())
            _intersection.Next()
        _line = gp_Lin(_gp_Pnt_2, gp_Dir(0, 0, 1))
        # look for intersection between loaded shape and line through _gp_Pnt_2
        _intersection.Init(GeomAdaptor_Curve(Geom_Line(_line).GetHandle()))
        while _intersection.More():
            _list_gp_Pnt_2_intersection_points.append(_intersection.Pnt())
            _intersection.Next()

        # Determine if the edge is an inner edge or an outer
        _num_intersections_p1 = len(_list_gp_Pnt_1_intersection_points)
        _num_intersections_p2 = len(_list_gp_Pnt_2_intersection_points)
        if _num_intersections_p1 != 0 and _num_intersections_p2 != 0:
            _list_edges_to_remove.append(edge_an_edge)
            # Debug: show points that indicate a inner edge
            # draw_point_as_vertex(_gp_Pnt_1)
            # draw_point_as_vertex(_gp_Pnt_2)
        else:
            _list_edges_to_keep.append(edge_an_edge)
        _done_counter += 1
        if _solution_queue is None:
            print("\r\tDone " + str(_done_counter) + "/" + str(len(edges_shape_projection)) + " edges", end="")
        else:
            _progress_queue.put(_done_counter - _done_pref)
            _done_pref = _done_counter

    if _solution_queue is not None:
        # multi core: push solution to Queue plus its index to create the complete list otherwise strange bugs happen
        _solution_queue.put([_list_edges_to_keep, process_index])
    else:
        print("\n\tDetected and removed " + str(len(_list_edges_to_remove)) + " inner edges.")
    return _list_edges_to_keep


def create_measure_points(_edge_an_edge, _int_parameter, _gp_Pnt_edge_midpoint):
    """
    Creates points perpendicular to a point on a given edge with a defined distance to the curve/edge

    :param _edge_an_edge: A edge to what the points should be created
    :param _int_parameter: The parameter of the curve where we find the middle point (e.g. length of curve/2)
    :param _gp_Pnt_edge_midpoint: Left over. Not used... exists because of debugging purposes
    :return: point1, point2 (class: gp_Pnt)
    """
    ee = Edge(_edge_an_edge)
    bCurve = ee.adaptor  # makes multi core working with windows
    # bCurve = BRepAdaptor_Curve(_edge_an_edge)  # only works with linux
    if bCurve is not None:
        _t = gp_Vec()  # This is our tangent vector
        _mymid = gp_Pnt()
        bCurve.D1(_int_parameter, _mymid, _t)  # Note: _mymid == _gp_Pnt_edge_midpoint!
        _b = _t.Crossed(gp_Vec(0, 0, 1))  # Compute cross vector b = _t x _z
        _b.Normalize()  # Normalize it
        _distance_to_curve = 0.000001  # How far de we want the test points to be away from the curve
        _gp_Pnt_1 = gp_Pnt(gp_Vec(_mymid.X(), _mymid.Y(), _mymid.Z()).Added(
            _b.Multiplied(_distance_to_curve)).XYZ())  # Compute a point with _distance_to_curve
        _gp_Pnt_2 = gp_Pnt(gp_Vec(_mymid.X(), _mymid.Y(), _mymid.Z()).Added(
            _b.Multiplied(-_distance_to_curve)).XYZ())  # Compute a point with _distance_to_curve
        # draw_point_as_vertex(_gp_Pnt_1)
        # draw_point_as_vertex(_gp_Pnt_2)
        return _gp_Pnt_1, _gp_Pnt_2


def clean_edges_shadow_outline(_list_edges_shadow_outline, projected_shape):
    """
    NOT USED ANYWHERE - WIP
    Supposed to make STL geometry with messy outline working
    Cleans the projected outline

    :param _list_edges_shadow_outline:
    :return: Outline without overlapping edges etc.
    """
    print("Cleaning shadow outline ...")

    def get_edge_index_from_row_index(index_row, num_edges):
        if index_row < num_edges:
            return index_row
        else:
            return index_row - num_edges

    def get_second_vertex_index_from_first_row_index(index_row, num_edges):
        if index_row < num_edges:
            return index_row + num_edges
        else:
            return index_row - num_edges

    def get_number_of_zero_distances(row, tolerance):
        _num_zeros = 0
        for i_col in range(len(row)):
            if row[i_col] <= tolerance:
                _num_zeros += 1
        return _num_zeros

    def find_shortest_edge_length(matrix, number_edges):
        _shortest = 1000000
        for _i_row in range(number_edges):
            for _i_col in range(number_edges, number_edges * 2):
                if (_i_col - _i_row) == number_edges:  # only diagonal elements
                    if matrix[_i_row][_i_col] < _shortest:
                        _shortest = matrix[_i_row][_i_col]
        print("\tShortest edge has length: ", _shortest)
        # TODO remove comment?!
        # return 0
        return _shortest

    _point_first = []
    _point_last = []
    _indices_to_be_removed = []  # A list that contains all indices of _list_edges_shadow_outline that shall be deleted
    for _edge in _list_edges_shadow_outline:
        _point = BRep_Tool_Pnt(Edge(_edge).first_vertex())
        _point_first.append([_point.X(), _point.Y()])
        _point = BRep_Tool_Pnt(Edge(_edge).last_vertex())
        _point_last.append([_point.X(), _point.Y()])
    _point_all = numpy.concatenate((_point_first, _point_last))
    _num_edges = len(_list_edges_shadow_outline)
    print("\tNumber of edges: ", _num_edges)
    _dist_matrix = distance.cdist(_point_all, _point_all, 'euclidean')  # distance of all vertices to each other
    _tol = find_shortest_edge_length(_dist_matrix, _num_edges)
    for i_row, row in enumerate(_dist_matrix):  # Rows with index
        _num_zeros = get_number_of_zero_distances(row, _tol)
        if _num_zeros == 1:
            pass
            # Case 2 & 3 -->
            print("\tFound no neighbouring vertex to row " + str(i_row) + " removing edge: " + str(
                get_edge_index_from_row_index(i_row, _num_edges)))
            # TODO: removes too much -> holes :(
            _indices_to_be_removed.append(get_edge_index_from_row_index(i_row, _num_edges))
        elif _num_zeros > 2:
            # TODO remove comment to enable all features - far away form robust :(
            pass
            # # also test the other vertex of edge
            # print("\tFound "+str(_num_zeros-2)+" possible duplicates at row " + str(i_row) + " checking row " + str(get_second_vertex_index_from_first_row_index(i_row, _num_edges)) + " as well")
            # _num_zeros = get_number_of_zero_distances(_dist_matrix[get_second_vertex_index_from_first_row_index(i_row, _num_edges)], _tol)
            # if _num_zeros > 2:
            #     # Case 1 (edge directly on top of another one) --> remove
            #     print(CColors.WARNING + "\tFound " + str(_num_zeros-2) + " duplicate edges in row "+str(i_row)+". Removing edge "+str(get_edge_index_from_row_index(i_row, _num_edges))+"!" + CColors.ENDC)
            #     _indices_to_be_removed.append(get_edge_index_from_row_index(i_row, _num_edges))
            #     # Matrix does not get recalculated. Need to remove detected duplicate from algos scope/eyes
            #     # Otherwise algo will remove all edges that "once" were duplicates --> would result in holes
            #     _ow_dist = 1000  # overwrite distance > _tol
            #     for new_row_index in range(i_row + 1, _num_edges*2):
            #         if new_row_index < _num_edges:
            #             _dist_matrix[new_row_index][i_row] = _ow_dist
            #         elif new_row_index == _num_edges:
            #             _dist_matrix[new_row_index] = _ow_dist  # whole row -> will result in _num_zeros != 2
            #         elif new_row_index > _num_edges:
            #             _dist_matrix[new_row_index, get_second_vertex_index_from_first_row_index(i_row, _num_edges)] = _ow_dist
        elif _num_zeros != 2:
            pass
            # print("\tSome strange: Number of <_tol entries is " + str(_num_zeros) + " maybe tolerance is too big")
    print(_indices_to_be_removed)
    _indices_to_be_removed = set(_indices_to_be_removed)  # remove duplicate indices
    for i in sorted(_indices_to_be_removed, reverse=True):  # remove indices, sort so highest index is removed first
        del _list_edges_shadow_outline[i]
    print(CColors.OKBLUE + "Done! Removed " + str(len(_indices_to_be_removed)) + " edges" + CColors.ENDC)
    return _list_edges_shadow_outline


def estimate_max_cross_section(_new_shape, _samples_per_axis=100):
    """
    Estimates the maximum thickness in z-direction of a shape.
    Samples the _new_shape_loaded at a resolution of bounding_box_size/100 [units]

    :param _samples_per_axis: Number of sample points per axis. Default: 100
    :param _new_shape: The shape that should be analyzed
    :return: The maximum thickness found in [units] of the shapes coordinate system
    """
    xmin, ymin, _, xmax, ymax, _ = get_boundingbox(_new_shape)
    _done = 0
    _total_samples = _samples_per_axis*_samples_per_axis
    _x_sep_size = (xmax-xmin) / _samples_per_axis
    _y_sep_size = (ymax - ymin) / _samples_per_axis
    _max_height = 0
    z_getter = ZGetter(None, _new_shape)
    import numpy
    for _x_coord in numpy.arange(xmin, xmax, _x_sep_size):
        for _y_coord in numpy.arange(ymin, ymax, _y_sep_size):
            # TODO: implement a get_z_orthogonal() to get more exact thickness on curved faces
            _z_upper, _z_lower = z_getter.get_z(_x_coord, _y_coord)
            _done += 1
            print("\r\tDone " + str(_done) + "/" + str(_total_samples) + " sample points", end="")
            if _z_upper is not None and _z_lower is not None:
                _height = _z_upper-_z_lower
                if _height > _max_height:
                    _max_height = _height
    print("\n\tEstimated thickness to ", _max_height)
    return _max_height


def find_edges_used_in_convex_hull(_convex_hull, _list_all_edges):
    """
    BUGGY - REMOVES SOME OF THEM BUT BY FAR NOT ALL AS HULL CONSISTS OF LESS EDGES THAN PROJECTION OUTLINE
    See what vertices are used to create the convex hull. Remove corresponding edges from the list.
    It can not be guaranteed that the returned list is 100% correct as convex hull may mix vertices form different edges

    :param _convex_hull: The convex hull
    :type _convex_hull: scipy.spatial.ConvexHull
    :param _list_all_edges: A list of Edges that needs to be cleaned
    :return: List of remaining edges that are not part of the convex hull
    """
    i = 0
    for _vertex_index_in_hull in _convex_hull.vertices:
        for _a_edge in _list_all_edges:
            topo_exp = Topo(_a_edge)
            vertices_of_edge = topo_exp.vertices()
            for vertex in vertices_of_edge:
                v_x = BRep_Tool_Pnt(vertex).X()  # That little trick gets us the coords of a vertex
                v_y = BRep_Tool_Pnt(vertex).Y()
                if v_x == _convex_hull.points[_vertex_index_in_hull][0] \
                        and v_y == _convex_hull.points[_vertex_index_in_hull][1]:
                    _list_all_edges.remove(_a_edge)
                    i += 1
                    # Problem: need to move on to next edge, python=pain_in_the_ass when you want to break nested loops
                    break
            else:
                continue  # only executed if the inner loop did NOT break - Many thanks to stackoverflow.com
            break  # only executed if the inner loop DID break
        else:
            continue
    print("Removed " + str(i) + " edges that are already part of convex hull from list")
    return _list_all_edges


def convex_hull_to_wire(_convex_hull):
    """
    Converts a convex hull object form scipy.spatial.ConvexHull algo to a TopoDS_Wire

    :param _convex_hull: Object obtained by algo
    :type _convex_hull: scipy.spatial.ConvexHull
    :return: a closed TopoDS_Wire of the outline/convex hull, all the vertices not part of the convex hull
    """
    make_poly = BRepBuilderAPI_MakePolygon()  # points/vertices to a polygonal wire -> TopoDS_Wire with straight edges
    for _point_index_in_hull in _convex_hull.vertices:
        # adding every second point should be enough as two points make for an edge and may be overlapping anyways
        make_poly.Add(gp_Pnt(_convex_hull.points[_point_index_in_hull][0], _convex_hull.points[_point_index_in_hull][1], 0))
    make_poly.Build()
    make_poly.Close()
    with assert_isdone(make_poly, 'failed to produce wire from convex hull :('):
        return make_poly.Wire()


def create_projection(TopoDS_Shape_shape):
    """
    Creates a projection in Z direction onto the ground plane. Projection might not be perfect. Needs cleaning!

    :param TopoDS_Shape_shape: The shape that should be projected to X/Y layer
    :return: TopoDS_Shape of the projected outline
    """
    print("Creating projection ...", end="")
    myProj = HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)))
    myAlgo = HLRBRep_Algo()
    myAlgo.Add(TopoDS_Shape_shape)
    myAlgo.Projector(myProj)
    myAlgo.Update()
    myAlgo.Hide()
    aHLRToShape = HLRBRep_HLRToShape(myAlgo.GetHandle())
    shape_projection = TopoDS_Shape(aHLRToShape.VCompound())
    print(CColors.OKBLUE + " Done!" + CColors.ENDC)
    return shape_projection


def algo_preform_v2(list_lower_layer: list, list_upper_layer: list, cad_shape: TopoDS_Shape, ignore_steps=False,
                    import_filetype=None, display_Viewer3d=None):
    """
    Creates a preform by extrusion and boolean operations. Keeps concave edges (hard angles)
    Of there is only one upper and lower zone we will generate the preform only be using the lower zones + an estimate
    of the thickness of the CAD part. Otherwise a preform with a height equal to the max. z value of the CAD-part would
    be created. That would make life a lot harder for the G-Code modding script as preform height != CAD part height.

    :param ignore_steps: Set to True if you want to create a simple "shadow-projection" of the shape ignoring concave
        steps. Results will be the same as with Preform algo v1. Just a lot more stable.
    :param cad_shape: The shape containing the original CAD geometry
    :param import_filetype: Unused!
    :param display_Viewer3d: Unused! - might be needed for debugging to call rendering methods
    :param list_lower_layer: A list composed of faces that represent the lower half of the imported geometry
    :param list_upper_layer: A list composed of faces that represent the upper half of the imported geometry
    :return:
    """
    print("")
    print(CColors.OKGREEN + "--- Generate preform v2" + CColors.ENDC)
    if ignore_steps:
        print(CColors.BOLD + "Ignoring concave edges/steps. Generating preform without them" + CColors.ENDC)

    xmin, ymin, zmin, xmax, ymax, zmax = get_boundingbox(cad_shape, use_mesh=True)
    if xmin < 0.0 or ymin < 0.0 or zmin < 0.0:
        print(CColors.FAIL + "Parts of the geometry are outside the build volume! Preform-Algo v2 might fail.\nMake"
                             " sure imported geometry is in positive 3D space (x, y, z)! Currently: (" + str(xmin)
              + ", " + str(ymin) + ", " + str(zmin) + ")" + CColors.ENDC)
        # print(CColors.WARNING + "Moving/Mirroring geometry into positive space!" + CColors.ENDC)
        # cad_shape = transform_shape(cad_shape, [abs(xmin)+1, abs(ymin)+1, abs(zmin)+1, 0.0, 0.0, 0.0])

    print(CColors.OKBLUE + "Zoning lower layer..." + CColors.ENDC)
    _lower_zones = zone_layer(cad_shape, list_lower_layer)
    print(CColors.OKBLUE + "Zoning upper layer..." + CColors.ENDC)
    _upper_zones = zone_layer(cad_shape, list_upper_layer)
    xmin, ymin, zmin, xmax, ymax, zmax = get_boundingbox(cad_shape, use_mesh=True)  # Bounding Box/Hüllkörper around CAD-geometry

    _only_one_zone = False
    if len(_upper_zones) == 1 and len(_lower_zones) == 1:
        print(CColors.BOLD + "Detected a shape with gradual thickness changes. Using shortcut!" + CColors.ENDC)
        _only_one_zone = True

    if not _only_one_zone:
        print(CColors.OKBLUE + "Starting upper preform..." + CColors.ENDC)
        _current_num = 0
        _shapes_from_zones = []
        for _a_zone in _upper_zones:
            _current_num += 1
            # https://dev.opencascade.org/index.php?q=node/1179
            print(CColors.OKBLUE + "\tProcessing zone " + str(_current_num) + "..." + CColors.ENDC)
            _zone_shape = list_to_compound_shape(_a_zone.list_faces)

            print(CColors.ENDC + "\t\tGenerating positive extrusion..." + CColors.ENDC)
            z_xmin, z_ymin, z_zmin, z_xmax, z_ymax, z_zmax = get_boundingbox(_zone_shape)
            extruded_shape = make_extrusion(_zone_shape, zmax * 2.1, vector=gp_Vec(0., 0., 1.))

            # Section/Common with a shape --> shape
            # print(CColors.ENDC + "\tEstimating zone thickness (z-direction)..." + CColors.ENDC)
            # est_height = estimate_max_zone_height(cad_shape, _zone_shape, _zone_is_curved=_a_zone.is_curved())
            # containing_box = BRepPrimAPI_MakeBox(z_xmax+1, z_ymax+1, -1 * est_height)  # (0, 0) to (x, y, z)
            # (0, 0, zmax) to (x, y, z)
            containing_box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, zmax), z_xmax + 1, z_ymax + 1, z_zmax)
            containing_box.Build()

            print(CColors.ENDC + "\t\tGenerating common solid..." + CColors.ENDC)
            section = BRepAlgoAPI_Common(extruded_shape, containing_box.Shape())
            _new_shape = section.Shape()

            print(CColors.ENDC + "\t\tRepositioning zone..." + CColors.ENDC)
            _translation = gp_Trsf()
            # _translation.SetTranslation(gp_Pnt(0, 0, 0), gp_Pnt(0, 0, est_height + z_zmin))
            _translation.SetTranslation(gp_Pnt(0, 0, 0), gp_Pnt(0, 0, -zmax))
            _transformator = BRepBuilderAPI_Transform(_translation)
            _transformator.Perform(_new_shape)
            _new_shape = _transformator.Shape()
            _shapes_from_zones.append(_new_shape)
        # https://www.opencascade.com/doc/occt-7.2.0/overview/html/occt_user_guides__boolean_operations.html#occt_algorithms_11b_1
        # There are faster and less dirty ways but this works!
        print(CColors.OKBLUE + "\tFusing/Fixing geometry..." + CColors.ENDC)
        if len(_shapes_from_zones) > 1:
            upper_preform_shape = _shapes_from_zones[0]
            for _i in range(len(_shapes_from_zones) - 1):
                boolean_algo = BRepAlgoAPI_Fuse(upper_preform_shape, _shapes_from_zones[_i + 1])
                upper_preform_shape = boolean_algo.Shape()
        else:
            boolean_algo = BRepAlgoAPI_Fuse(_shapes_from_zones[0], _shapes_from_zones[0])
            upper_preform_shape = boolean_algo.Shape()
        print(CColors.OKGREEN + "Finished upper preform!" + CColors.ENDC)

    print(CColors.OKBLUE + "Starting lower preform..." + CColors.ENDC)
    _current_num = 0
    _shapes_from_zones = []
    est_height = -1
    for _a_zone in _lower_zones:
        _current_num += 1
        # https://dev.opencascade.org/index.php?q=node/1179
        print(CColors.OKBLUE + "\tProcessing zone " + str(_current_num) + "..." + CColors.ENDC)
        _zone_shape = list_to_compound_shape(_a_zone.list_faces)

        print(CColors.ENDC + "\t\tGenerating negative extrusion..." + CColors.ENDC)
        z_xmin, z_ymin, z_zmin, z_xmax, z_ymax, z_zmax = get_boundingbox(_zone_shape)
        # extrude twice the height
        extruded_shape = make_extrusion(_zone_shape, zmax * 2.1, vector=gp_Vec(0., 0., -1.))

        # Section/Common with a shape --> shape
        if _only_one_zone:
            print(CColors.ENDC + "\tEstimating zone thickness (z-direction)..." + CColors.ENDC)
            est_height = estimate_max_zone_height(cad_shape, _zone_shape, _zone_is_curved=_a_zone.is_curved())
            containing_box = BRepPrimAPI_MakeBox(z_xmax + 1, z_ymax + 1, -1 * est_height)  # (0, 0) to (x, y, z)
        elif ignore_steps:
            print(CColors.ENDC + "\tEstimating part thickness (z-direction)..." + CColors.ENDC)
            if est_height == -1:  # only estimate complete part thickness once
                est_height = estimate_max_zone_height(cad_shape, cad_shape, _zone_is_curved=True)
            containing_box = BRepPrimAPI_MakeBox(z_xmax + 1, z_ymax + 1, -1 * est_height)  # (0, 0) to (x, y, z)
        else:
            containing_box = BRepPrimAPI_MakeBox(z_xmax + 1, z_ymax + 1, -1 * (zmax - z_zmin))  # (0, 0) to (x, y, z)
        containing_box.Build()

        print(CColors.ENDC + "\t\tGenerating common solid..." + CColors.ENDC)
        section = BRepAlgoAPI_Common(extruded_shape, containing_box.Shape())
        _new_shape = section.Shape()

        print(CColors.ENDC + "\t\tRepositioning zone..." + CColors.ENDC)
        _translation = gp_Trsf()
        if _only_one_zone:
            _translation.SetTranslation(gp_Pnt(0, 0, 0), gp_Pnt(0, 0, est_height + z_zmin))
        elif ignore_steps:
            _translation.SetTranslation(gp_Pnt(0, 0, 0), gp_Pnt(0, 0, est_height + zmin))
        else:
            _translation.SetTranslation(gp_Pnt(0, 0, 0), gp_Pnt(0, 0, zmax))
        _transformator = BRepBuilderAPI_Transform(_translation)
        _transformator.Perform(_new_shape)
        _new_shape = _transformator.Shape()
        _shapes_from_zones.append(_new_shape)

    # https://www.opencascade.com/doc/occt-7.2.0/overview/html/occt_user_guides__boolean_operations.html#occt_algorithms_11b_1
    # There are faster and less dirty ways but this works!
    print(CColors.OKBLUE + "\tFusing/Fixing geometry..." + CColors.ENDC)
    if len(_shapes_from_zones) > 1:
        lower_preform_shape = _shapes_from_zones[0]
        for _i in range(len(_shapes_from_zones) - 1):
            boolean_algo = BRepAlgoAPI_Fuse(lower_preform_shape, _shapes_from_zones[_i + 1])
            lower_preform_shape = boolean_algo.Shape()
    else:
        boolean_algo = BRepAlgoAPI_Fuse(_shapes_from_zones[0], _shapes_from_zones[0])
        lower_preform_shape = boolean_algo.Shape()
    print(CColors.OKGREEN + "Finished lower preform!" + CColors.ENDC)
    if _only_one_zone or ignore_steps:
        final_preform = lower_preform_shape
    else:
        print(CColors.OKBLUE + "Building final preform..." + CColors.ENDC)
        common_operation = BRepAlgoAPI_Common(upper_preform_shape, lower_preform_shape)
        final_preform = common_operation.Shape()
    print(CColors.OKGREEN + "Finished generating preform!" + CColors.ENDC)
    # https://www.opencascade.com/content/modify-shape
    # Can change the coordinates of vertices
    # Might be the solution to get better thickness approximation with some parts. Transform upper layer to est. height
    return final_preform


def zone_layer(cad_shape, list_faces_ul):
    """
    Determine groups/lists of connected faces

    :type list_faces_ul: List of relevant faces (upper or lower part/layer of CAD-Geometry)
    :type cad_shape: Original CAD import as TopoDS_Shape
    :returns: A list of Zone objects
    """

    def is_zone_curved(_new_zone, _tol=1):
        """
        Check angle of face normal to z-axis
        :param zone: the zone to check
        :param _tol: [°] +-180°
        :return: True/False
        """
        # TODO: check by face normal may be fast an accurate with STL files but may fail with STEP
        _is_curved = False
        _pi = numpy.pi
        for _face in _new_zone.list_faces:
            _Face = Face(_face)
            _mid_u_v, _ = _Face.mid_point()
            gp_dir_normal = _Face.DiffGeom.normal(_mid_u_v[0], _mid_u_v[1])
            _angle_z_to_normal = gp_dir_normal.Angle(gp_Dir(0, 0, -1))
            _angle_z_to_normal = (_angle_z_to_normal * 180) / _pi  # to °
            if _angle_z_to_normal != 0 and _angle_z_to_normal != 180:
                _is_curved = True
        return _is_curved

    list_zones = []
    num_zones = 0
    topo = Topo(cad_shape)  # Use a Topo_Explorer object to get access to faces with their edges and vertices
    while len(list_faces_ul) > 0:
        _a_face = list_faces_ul[0]
        # Find neighbouring faces via common edges
        _list_edges_from_face = list(topo.edges_from_face(_a_face))
        zone = Zone()
        while len(_list_edges_from_face) > 0:
            for _an_edge_from_face in _list_edges_from_face:
                _list_edges_from_face.remove(
                    _an_edge_from_face)  # remove edge from list because we will have analyzed it
                _list_faces_possible_neighbours = topo.faces_from_edge(_an_edge_from_face)  # all possible neighbours
                for _face_poss_neighbour in _list_faces_possible_neighbours:
                    if _face_poss_neighbour in list_faces_ul:
                        _list_edges_from_face.extend(list(topo.edges_from_face(_face_poss_neighbour)))
                        zone.list_faces.append(_face_poss_neighbour)
                        list_faces_ul.remove(_face_poss_neighbour)
        zone._curved = is_zone_curved(zone)
        list_zones.append(zone)
        num_zones += 1
        print("\r\tDetected " + str(num_zones) + " zones", end="")
    print("")
    return list_zones


def detect_concave_edges_in_zone(_new_list_zones_ul: list):
    """
    Detect convex edges from upper zone and lower zone list of faces

    :type _new_list_zones_ul: List of TopoDS_Face objects in each zone
        [[[lower_zone_0], [lower_zone_1], [lower_zone_n], ...], [[upper_zone_0], [upper_zone_1], [upper_zone_n], ...]]
    """
    for i in range(2):
        if i == 0:
            print(CColors.OKBLUE + "Detecting lower layer convex edges..." + CColors.ENDC)
        else:
            print(CColors.OKBLUE + "Detecting upper layer convex edges..." + CColors.ENDC)
        _zone_lists = _new_list_zones_ul[i]
        for a in range(len(_zone_lists)):
            # TODO: do we really need those edges?!
            pass


def find_concave_edges(shape_cad_geometry, display_Viewer3d):
    # Tip: http://www.grasshopper3d.com/forum/topics/convex-or-concave-angle-between-faces
    list_concave_edges = []
    print(CColors.OKBLUE + "Looking for convex edges..." + CColors.ENDC)
    topo = Topo(shape_cad_geometry)
    all_edges = topo.edges()
    for an_edge in all_edges:
        faces_from_edge = topo.faces_from_edge(an_edge)
        for a_face in faces_from_edge:
            _first_face = Face(a_face)
            _mid_u_v, _ = _first_face.mid_point()
            _midpoint_first = _first_face.parameter_to_point(_mid_u_v[0], _mid_u_v[1])
            _gp_dir_normal = _first_face.DiffGeom.normal(_mid_u_v[0], _mid_u_v[1])
            for a_second_face in faces_from_edge:
                if a_second_face is not a_face:
                    _secondFace = Face(a_second_face)
                    _mid_u_v_second, _ = _secondFace.mid_point()

                    _gp_dir_normal_second = _secondFace.DiffGeom.normal(_mid_u_v_second[0], _mid_u_v_second[1])
                    _angle_between_faces = _gp_dir_normal_second.Angle(_gp_dir_normal)
                    _angle_between_faces = (_angle_between_faces * 180) / numpy.pi  # to degree

                    _midpoint_second = _secondFace.parameter_to_point(_mid_u_v_second[0], _mid_u_v_second[1])
                    _vector_between_midpoints = gp_Dir(gp_Vec(_midpoint_first, _midpoint_second))
                    _angle_midpointvector_normal = _vector_between_midpoints.Angle(_gp_dir_normal)  # to normal of face1
                    _angle_midpointvector_normal = (_angle_midpointvector_normal * 180) / numpy.pi  # to degree

                    print(_angle_between_faces, _angle_midpointvector_normal)
                    if _first_face.Orientation() == TopAbs_FORWARD:
                        if _angle_between_faces == 90 and _angle_midpointvector_normal < 90:
                            list_concave_edges.append(an_edge)
                    else:
                        if _angle_between_faces == 90 and _angle_midpointvector_normal > 90:
                            list_concave_edges.append(an_edge)
    draw_list_of_edges(list_concave_edges, display_Viewer3d)


def estimate_max_zone_height(_cad_shape, _a_zone_shape, _zone_is_curved=True):
    """
    Returns max. thickness of the CAD geometry in the area of a given shape/zone/geometry (_a_zone_shape)

    :param _cad_shape:
    :param _a_zone_shape:
    :param _zone_is_curved:
    :return:
    """
    xmin, ymin, _, xmax, ymax, _ = get_boundingbox(_a_zone_shape)
    _done = 0
    if _zone_is_curved:
        _samples_per_axis = 70
    else:
        _samples_per_axis = 3
    _total_samples = _samples_per_axis*_samples_per_axis
    _x_sep_size = (xmax-xmin) / _samples_per_axis
    _y_sep_size = (ymax - ymin) / _samples_per_axis
    _max_height = 0
    z_getter = ZGetter(None, _cad_shape)
    import numpy
    for _x_coord in numpy.arange(xmin+0.1, xmax-0.1, _x_sep_size):
        for _y_coord in numpy.arange(ymin+0.1, ymax-0.1, _y_sep_size):
            # TODO: implement a get_z_orthogonal() to get more exact thickness on curved faces
            _z_upper, _z_lower = z_getter.get_z(_x_coord, _y_coord)
            _done += 1
            if _done%100:
                print("\r\t\tDone " + str(_done) + "/" + str(_total_samples) + " sample points", end="")
            if _z_upper is not None and _z_lower is not None:
                _height = _z_upper-_z_lower
                if _height > _max_height:
                    _max_height = _height
    print("\n\tEstimated thickness to ", _max_height)
    return _max_height
