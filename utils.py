import numpy
from OCC.BRep import BRep_Builder, BRep_Tool_Pnt
from OCC.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeWire, BRepBuilderAPI_Sewing, \
    BRepBuilderAPI_Transform, BRepBuilderAPI_GTransform
from OCC.BRepIntCurveSurface import BRepIntCurveSurface_Inter
from OCC.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeSphere
from OCC.SMESH import SMESH_Mesh
from OCC.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE
from OCC.TopExp import TopExp_Explorer
from OCC.TopoDS import topods_Edge, topods_Face, topods_Wire, TopoDS_Compound, TopoDS_Shape
from OCC.gp import gp_Vec, gp_Pnt, gp_Dir, gp_Lin, gp_Trsf, gp_Quaternion, gp_Intrinsic_XYZ, gp_GTrsf

from OCCUtils import Topo
from OCCUtils.Construct import make_vertex, make_edge


def make_extrusion(face, length, vector=gp_Vec(0., 0., 1.)):
    """Creates an extrusion from a :param face, along the vector :param vector with a distance :param length.
    By default, the extrusion is along the z axis.

    :param face: A TopoDS_Face that should be extruded
    :param length: The height of the extrusion
    :param vector: A gp_Vec() Object. Gives the direction of the extrusion
    :return: The extruded TopoDS_Shape
    """
    vector.Normalize()
    vector.Scale(length)
    return BRepPrimAPI_MakePrism(face, vector).Shape()


def make_closed_polygon(*args):
    mypoly = BRepBuilderAPI_MakePolygon()
    for pt in args:
        if isinstance(pt, list) or isinstance(pt, tuple):
            for i in pt:
                mypoly.Add(i)
        else:
            mypoly.Add(pt)
    mypoly.Build()
    mypoly.Close()
    result = mypoly.Wire()
    return result


def make_wire(*args):
    # if we get an iterable, than add all edges to wire builder
    if isinstance(args[0], list) or isinstance(args[0], tuple):
        wire = BRepBuilderAPI_MakeWire()
        for i in args[0]:
            wire.Add(i)
        # wire.Build()
        return wire.Wire()
    wire = BRepBuilderAPI_MakeWire(*args)
    return wire.Wire()


def get_single_intersection_points(shape, global_x, global_y):
    _list_gp_Pnt_intersection_points = []
    _intersection = BRepIntCurveSurface_Inter()
    _line = gp_Lin(gp_Pnt(global_x, global_y, 0), gp_Dir(0, 0, 1))
    _intersection.Init(shape, _line, 0.0000001)
    while _intersection.More():
        _list_gp_Pnt_intersection_points.append(_intersection.Pnt())
        print(_intersection.Pnt())
        _intersection.Next()
    return _list_gp_Pnt_intersection_points


def get_faces(shape):
    """ return the faces from `shape`

    :param shape: TopoDS_Shape, or a subclass like TopoDS_Solid
    :return: a list of faces found in `shape`
    """
    topexp = TopExp_Explorer()
    topexp.Init(shape, TopAbs_FACE)
    _faces = []

    while topexp.More():
        fc = topods_Face(topexp.Current())
        _faces.append(fc)
        topexp.Next()
    return _faces


def get_wires(shape):
    topexp = TopExp_Explorer()
    topexp.Init(shape, TopAbs_WIRE)
    _wires = []

    while topexp.More():
        fc = topods_Wire(topexp.Current())
        _wires.append(fc)
        topexp.Next()
    return _wires


def get_edges(shape):
    topexp = TopExp_Explorer()
    topexp.Init(shape, TopAbs_EDGE)
    _edges = []

    while topexp.More():
        fc = topods_Edge(topexp.Current())
        _edges.append(fc)
        topexp.Next()
    return _edges


def draw_point_as_sphere(_gp_pnt, display_viewer3d, _radius=1):
    if display_viewer3d is not None:
        display_viewer3d.DisplayShape(BRepPrimAPI_MakeSphere(_gp_pnt, _radius).Shape())  # Origin


def draw_point_as_vertex(_gp_pnt, display_viewer3d):
    if display_viewer3d is not None:
        display_viewer3d.DisplayShape(make_vertex(_gp_pnt))


def draw_vector_as_edge(_gp_vec, _gp_pnt, display_viewer3d):
    if display_viewer3d is not None:
        display_viewer3d.DisplayShape(make_edge(gp_Lin(_gp_pnt, gp_Dir(_gp_vec))))


def draw_list_of_edges(_list_outer_outline, display_viewer3d):
    if display_viewer3d is not None:
        for _a_edge in _list_outer_outline:
            display_viewer3d.DisplayShape(_a_edge)


def draw_list_of_faces(_list_faces, display_viewer3d):
    if display_viewer3d is not None:
        for _a_face in _list_faces:
            display_viewer3d.DisplayShape(_a_face)


def multicore_update_ui_find_shadow_outline(my_update_queue, my_process_list):
    processes_still_running = True
    while processes_still_running:
        print("\tProgress: ", end="")
        _list_process_is_dead = []
        for process in my_process_list:
            if process.is_alive():
                _list_process_is_dead.append(False)
            else:
                _list_process_is_dead.append(True)
        if False in _list_process_is_dead:
            processes_still_running = False


def list_to_compound_shape(new_list) -> TopoDS_Compound:
    """
    Create a TopoDS_Compound from a list of e.g. faces

    :param new_list: a list of TopoDS_Shape elements
    :return: new_list as TopoDS_Compound object
    """
    _compound = TopoDS_Compound()
    a_builder = BRep_Builder()
    a_builder.MakeCompound(_compound)
    for _a_face in new_list:
        a_builder.Add(_compound, _a_face)
    return _compound


def sew_list(new_list):
    _sewer = BRepBuilderAPI_Sewing()
    first_run = True
    for _a_face in new_list:
        if first_run:
            _sewer.Load(new_list[0])
            first_run = False
        else:
            _sewer.Add(_a_face)
    return _sewer.SewedShape()


def edge_vertices_to_numpy_array(_list_of_edges):
    """
    Takes a list of TopoDS_Edges. Converts the vertices of all edges to a 2D numpy array. This array can be used by the
    scipy.spatial.ConvexHull algo to create a convex hull from the vertices.

    :param _list_of_edges: List of TopoDS_Edges
    :return: A 2darray (numpy) with all koordinates of vertices
    """
    _list_vertex_coords = []  # temp list to store all coords of the vertices
    for _TopoEdge in _list_of_edges:
        topo_explorer = Topo(_TopoEdge)
        vertices_of_edge = topo_explorer.vertices()
        for vertex in vertices_of_edge:
            _list_vertex_coords.append([BRep_Tool_Pnt(vertex).X(), BRep_Tool_Pnt(vertex).Y()])
    return numpy.array(_list_vertex_coords)


def find_unused_edges_in_convex_hull_old(_convex_hull, _list_all_edges):
    """
    DEPRECATED: NOT USED!
    See what vertices are used to create the convex hull. Remove corresponding edges from the list.
    It can not be guaranteed that the returned list is 100% correct as convex hull may mix vertices form different edges

    :param _convex_hull: The convex hull
    :type _convex_hull: scipy.spatial.ConvexHull
    :param _list_all_edges: A list of Edges that needs to be cleaned
    :return: List of remaining edges that are not part of the convex hull
    """
    i = 0
    for _a_edge in _list_all_edges:
        topo_exp = Topo(_a_edge)
        vertices_of_edge = topo_exp.vertices()
        for vertex in vertices_of_edge:
            v_x = BRep_Tool_Pnt(vertex).X()  # That little trick gets us the coords of a vertex
            v_y = BRep_Tool_Pnt(vertex).Y()
            for _vertex_index_in_hull in _convex_hull.vertices:
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


def get_shape_positon(new_shape: TopoDS_Shape) -> list:
    """
    Returns the position of the shapes coordinate system relative to the origin of the scene (0,0,0)
    When imported into the scene all geometry will sit at (0,0,0); (0,0,0)

    :param new_shape: A TopoDS_Shape that you want to get the info from
    :return: [translateX, translateY, translateZ, rotateX, rotateY, rotateZ] with rotation angles in rad
    """
    loc = new_shape.Location() # get_shape_pos()
    transfo = loc.Transformation()
    trnasl_part = transfo.TranslationPart()
    rotation = transfo.GetRotation()
    x, y, z = rotation.GetEulerAngles(gp_Intrinsic_XYZ)
    return [trnasl_part.X(), trnasl_part.Y(), trnasl_part.Z(), x, y, z]


def transform_shape(new_shape: TopoDS_Shape, trans_rot) -> TopoDS_Shape:
    """
    Transform a TopoDS_Shape relative according to a vector and a rotation given by trans_rot list. The list can be obtained by
    get_shape_positon(). It has the following structure [translateX, translateY, translateZ, rotateX, rotateY, rotateZ].
    This function also accepts a homogeneous transformation matrix as parameter trans_rot
    Use with caution. Might generate unexpected results when triangulation is accessed:
    https://opencascade.blogspot.com/2009/02/continued.html#Triangulation
    Use of BRepBuilderAPI_GTransform() should fix this but will not allow absolute transformations since the position
    is overwritten

    :param new_shape: The TopoDS_Shape you want to transform
    :param trans_rot: [translateX, translateY, translateZ, rotateX, rotateY, rotateZ] in mm and rad OR a homogeneous
        transformation matrix in form of a numpy.ndarray (4x4 or 3x3)
    :return: The transformed TopoDS_Shape
    """
    _translation = gp_Trsf()
    if type(trans_rot) is list:
        new_gp_quaternion = gp_Quaternion()
        new_gp_quaternion.SetEulerAngles(gp_Intrinsic_XYZ, trans_rot[3], trans_rot[4],
                                         trans_rot[5])
        new_gp_vector_translation = gp_Vec(trans_rot[0], trans_rot[1], trans_rot[2])
        _translation.SetTransformation(new_gp_quaternion, new_gp_vector_translation)
    elif type(trans_rot) is numpy.ndarray:
        # We got a homogeneous transformation matrix
        _translation.SetValues(trans_rot[0][0], trans_rot[0][1], trans_rot[0][2], trans_rot[0][3],
                               trans_rot[1][0], trans_rot[1][1], trans_rot[1][2], trans_rot[1][3],
                               trans_rot[2][0], trans_rot[2][1], trans_rot[2][2], trans_rot[2][3])
    # https://www.opencascade.com/content/changing-shape-transformation-and-performing-boolean-operations
    _transformator = BRepBuilderAPI_GTransform(gp_GTrsf(_translation))
    # This might cause problems as it does not transform underlying mesh from triangulation?
    # _transformator = BRepBuilderAPI_Transform(_translation)
    _transformator.Perform(new_shape)
    return _transformator.Shape()


def apply_transformation(points_matrix: numpy.ndarray, homo_transfo_matrix: numpy.ndarray) -> numpy.ndarray:
    """
    Does the same as cv2.ppf_match_3d.transformPCPose()

    :param points_matrix: The points you want to transform in form of a matrix (3, 1) OR (x, 3) -> one point per row
    :param homo_transfo_matrix: homogeneous transformation matrix
    :return: the transformed points_matrix
    """
    if points_matrix.shape == (3, 1):
        # retruns (3, 1) numpy array of transformed 3D point
        return numpy.matmul(homo_transfo_matrix[0:3, 0:3], points_matrix)
    else:
        # Allows to transform an entire array of points in one step
        # (matrix with one point per row)
        return numpy.matmul(homo_transfo_matrix[0:3, 0:3], points_matrix.T).T + homo_transfo_matrix[:3, 3].T

def set_axes_equal(ax):
    def set_axes_radius(_ax, _origin, _radius):
        _ax.set_xlim3d([_origin[0] - _radius, _origin[0] + _radius])
        _ax.set_ylim3d([_origin[1] - _radius, _origin[1] + _radius])
        _ax.set_zlim3d([_origin[2] - _radius, _origin[2] + _radius])
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    :param _ax: a matplotlib axis, e.g., as output from plt.gca().
      
    Source: https://stackoverflow.com/a/50664367
    """
    limits = numpy.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = numpy.mean(limits, axis=1)
    radius = 0.5 * numpy.max(numpy.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

