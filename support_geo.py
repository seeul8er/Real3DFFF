import numpy
from OCC.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
from OCC.TopoDS import TopoDS_Shape
from OCC.gp import gp_Dir, gp_Pnt

from CColors import CColors
from OCCUtils import Topo, get_boundingbox
from OCCUtils.face import Face
from utils import get_faces, make_extrusion, list_to_compound_shape, transform_shape


def extract_support_geometry_algo(new_shape: TopoDS_Shape, normal_thresh=90, extrude=True, solid_geo=False,
                                  solid_geo_offset=-0.2) -> tuple:
    """
    Gets all faces of shape_loaded which have an angle < normal_thresh between their normal in the midpoint of the
    face and the vector in negative z-direction (0, 0, -1). Faces are stored inside a list.

    :param extrude: Extrude the extracted faces by an amount of 0.0001 so that slicers detect it as geometry and
        create support for it. Pure faces (no closed 3D geometry) might be ignored by some slicers.
    :param new_shape: The TopoDS_Shape from which we want to extract the support
    :type normal_thresh: [°] detect the face as support if angle of the face normal to z-axis is lower than this value
    :param solid_geo: Create closed solid as support structure
    :param solid_geo_offset: in case of solid_geo: Distance in z between support structure and part for easier
        separation after print (z-bump/support_material_contact_distance)
    :return: A merged TopoDS_Compound (all support faces) and a list composed of TopoDS_Face elements that fulfill the
        requirement ang(normal, z) < normal_thresh
    """
    _list_faces = get_faces(new_shape)
    _list_faces_support = []
    _thresh_rad = (normal_thresh*numpy.math.pi)/180
    for _face in _list_faces:
        _Face = Face(_face)
        _mid_u_v, _ = _Face.mid_point()
        gp_dir_normal = _Face.DiffGeom.normal(_mid_u_v[0], _mid_u_v[1])
        _angle_to_normal = gp_dir_normal.Angle(gp_Dir(0, 0, -1))
        if _face.Orientation() == TopAbs_FORWARD and _angle_to_normal < _thresh_rad:
            _list_faces_support.append(_face)
        elif _face.Orientation() == TopAbs_REVERSED and _angle_to_normal > _thresh_rad:
            _list_faces_support.append(_face)
    if extrude or solid_geo:
        if solid_geo:
            comp = list_to_compound_shape(_list_faces_support)
            comp = transform_shape(comp, [0, 0, solid_geo_offset, 0, 0, 0])
            _, _, _, _, _, zmax = get_boundingbox(comp)
            sxmin, symin, szmin, sxmax, symax, szmax = get_boundingbox(new_shape)
            final_shape = make_extrusion(comp, -1*(zmax+10))
            box = BRepPrimAPI_MakeBox(gp_Pnt(sxmin-10, symin-10, -1*(zmax+15)), gp_Pnt(sxmax+10, symax+10, 0))
            final_shape = BRepAlgoAPI_Cut(final_shape, box.Shape()).Shape()
        else:
            final_shape = make_extrusion(list_to_compound_shape(_list_faces_support), 0.0001)
    else:
        final_shape = list_to_compound_shape(_list_faces_support)
    # print("Done!")
    return _list_faces_support, final_shape


def extract_faces_by_threshold(new_shape, normal_thresh=90, upper_lower="lower") -> list:
    """
    Not much differenct than extract_support_geometry_algo() but will not extrude the faces and can get upper or lower
    faces.
    Extracts the faces of a new_shape which have:
    an [ANGLE < (upper_lower="lower") normal_thresh] OR an [ANGLE > (upper_lower="upper") normal_thresh] between their
    normal in the midpoint of the face and the vector in negative z-direction (0, 0, -1).

    :param upper_lower: "lower" returns geometry that makes up the bottom plate (needed for support structure) "upper"
        returns geometry that makes up the top plate
    :type upper_lower: str
    :param new_shape: The TopoDS_Shape from which we want to extract the support (usually the imported CAD-File)
    :type normal_thresh: [°] detect the face as support if angle of the face normal to z-axis is <> than this
        value
    :return: A list composed of TopoDS_Face elements that fulfill the requirement (ang(normal, z) <> normal_thresh)
    """
    # TODO: Be able to handle STL/Mesh imports using SMESH
    # face_iter = mesh_ds.facesIterator()
    # for i in range(mesh_ds.NbFaces() - 1):
    #     face = face_iter.next()
    #     print('Face %i, type %i' % (i, face.GetType()))
    # http://docs.salome-platform.org/7/tui/SMESH/classSMDS__MeshFace.html
    # http://docs.salome-platform.org/7/tui/SMESH/namespaceSMESH__MeshAlgos.html#a5801b2b09361af8f078aa84226534c03
    print("Extracting " + upper_lower + " geometry ... ", end="")
    topo = Topo(new_shape)
    _list_faces = topo.faces()
    _list_faces_support = []
    _thresh_rad = (normal_thresh*numpy.math.pi)/180
    for _face in _list_faces:
        _Face = Face(_face)
        _mid_u_v, _ = _Face.mid_point()
        gp_dir_normal = _Face.DiffGeom.normal(_mid_u_v[0], _mid_u_v[1])
        _angle_to_normal = gp_dir_normal.Angle(gp_Dir(0, 0, -1))
        if upper_lower == "lower":
            if _face.Orientation() == TopAbs_FORWARD and _angle_to_normal < _thresh_rad:
                _list_faces_support.append(_face)
            elif _face.Orientation() == TopAbs_REVERSED and _angle_to_normal > _thresh_rad:
                _list_faces_support.append(_face)
        else:
            if _face.Orientation() == TopAbs_FORWARD and _angle_to_normal > _thresh_rad:
                _list_faces_support.append(_face)
            elif _face.Orientation() == TopAbs_REVERSED and _angle_to_normal < _thresh_rad:
                _list_faces_support.append(_face)
    print(CColors.OKBLUE + "Done!" + CColors.ENDC)
    return _list_faces_support
