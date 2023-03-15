import os
from random import randint

import numpy as np
from OCC.MeshVS import MeshVS_Mesh, MeshVS_DMF_Shading, MeshVS_DA_EdgeColor, MeshVS_DA_ShowEdges, MeshVS_BP_Mesh, \
    MeshVS_DA_InteriorColor, MeshVS_DA_DisplayNodes, MeshVS_MeshPrsBuilder
from OCC.Quantity import Quantity_NOC_GRAY90, Quantity_Color
from OCC.SMESH import SMESH_MeshVSLink, SMESH_Mesh
from OCC.TopoDS import TopoDS_Shape

from OCCUtils import get_boundingbox
from data_io.ImportType import ImportType
from data_io.LoadedData import LoadedData
from data_io.exporters import export_stl
from data_io.loaders import load_stl
from globals import TMP_FOLDER_PATH
from utils import transform_mesh, transform_shape


class LDMesh(LoadedData):

    def __init__(self, new_data: SMESH_Mesh, name: str, import_type: ImportType, filepath=None):
        super().__init__(new_data, name, import_type, filepath)
        self.mesh_color = randint(240, 500)  # Quantity_NOC_ORANGERED2
        self.show_wireframe = False
        self._cached_topods_shape = None  # In case we already converted the mesh to a TopoDS_Shape store shape here
        self._ref_pnts = []  # list of LDAtosRefPnt

    def get_stl_filepath(self, save_filepath=None):
        """
        Write the current geometry data to a file so that it can be imported by an external application or converted to
        a TopoDS_Shape via reimporting/reloading. Used for program internal conversions. Do not use for exporting.
        After transformations in 3D space the original loaded file does not represent the current state of the geometry

        :return: File path to the exported shape
        """
        if save_filepath is None:
            save_filepath = os.path.join(TMP_FOLDER_PATH, "_temp_shape_getstl.stl")
        return export_stl(self.data, save_filepath)  # Write to temp file

    def get_shape(self, ignore_cache=False) -> TopoDS_Shape:
        if self._cached_topods_shape is None or ignore_cache:
            # Write to temp file (in case it was transformed)
            self._cached_topods_shape = load_stl(self.get_stl_filepath())
        return self._cached_topods_shape

    def get_part_height(self, use_mesh=True):
        xmin, ymin, zmin, xmax, ymax, zmax = get_boundingbox(self.data, use_mesh=use_mesh)
        return zmax - zmin

    def has_ref_pnts(self) -> bool:
        return len(self._ref_pnts) > 0

    def set_mesh_color(self, new_color: int):
        """
        Must be integer or enum of type Quantity_NameOfColor

        :param new_color: Integer from 1~500
        """
        self.mesh_color = new_color
