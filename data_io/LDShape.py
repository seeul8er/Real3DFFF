import os

import numpy as np
from OCC import AIS
from OCC.Graphic3d import Graphic3d_NOM_ALUMINIUM
from OCC.TopoDS import TopoDS_Shape

from OCCUtils import get_boundingbox
from data_io.ImportType import ImportType
from data_io.LoadedData import LoadedData
from data_io.exporters import export_stl, generate_mesh
from data_io.loaders import load_stl
from globals import LINEAR_DEFLECTION, ANGULAR_DEFLECTION, TMP_FOLDER_PATH
from utils import get_shape_positon, transform_shape


class LDShape(LoadedData):
    """
    Stores data that can be represented as TopoDS_Shape (parametric CAD data like STP or IGES).
    """

    def __init__(self, new_data, name: str, import_type: ImportType, filepath=None):
        super().__init__(new_data, name, import_type, filepath)
        self.material = Graphic3d_NOM_ALUMINIUM  # Graphic3d_NameOfMaterial
        self._ref_pnts = []  # list of LDAtosRefPnt

    def get_stl_filepath(self):
        """
        Write the current geometry data to a file so that it can be imported by an external application or converted to
        a TopoDS_Shape via reimporting/reloading. Used for program internal conversions. Do not use for exporting.
        After transformations in 3D space the original loaded file does not represent the current state of the geometry

        :return: File path to the exported shape
        """
        return export_stl(self.data, os.path.join(TMP_FOLDER_PATH, "_temp_shape_getSTL.stl"))  # Write to temp file

    def get_position(self):
        return get_shape_positon(self.get_shape())

    def get_shape(self) -> TopoDS_Shape:
        return self.data

    def get_part_height(self, use_mesh=True):
        xmin, ymin, zmin, xmax, ymax, zmax = get_boundingbox(self.data, use_mesh=use_mesh)
        return zmax - zmin

    def generate_triangulated_temp_shape(self, lin_deflection=LINEAR_DEFLECTION, ang_deflection=ANGULAR_DEFLECTION):
        _temp_filename = os.path.join(TMP_FOLDER_PATH, "_temp.stl")
        print("Generating and loading temporary file: " + _temp_filename)
        newshape_mesh = generate_mesh(lin_deflection, ang_deflection, self.get_shape())
        export_stl(newshape_mesh, filename=_temp_filename, bool_asciimode=False)
        return load_stl(_temp_filename)
