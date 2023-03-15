from enum import Enum


class ImportType(Enum):
    stl = 1
    stp = 2
    igs = 3
    shape = 4  # A TopoDS_Shape from OCC - not really an import but used for in app generated geometry
    # When using SMESH. File is not converted to a TopoDS_Shape (OCC API geo. format) but directly displayed
    # With this import type you can not use the OCE API! Load the file first via load_stl() or convert manually
    # For display and Blender modules only
    mesh = 5  # Used with STL files since importing/conv large STLs to OCC internal format (TopoDS_Shape) takes too long
    gcode = 6
    ref_pnt = 7
    pnt_cloud = 8
    ply = 9
    dummy = 10
