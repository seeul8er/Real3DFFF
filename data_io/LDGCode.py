import os
import time
from importlib import util

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from OCC import AIS
from OCC.Graphic3d import Graphic3d_MaterialAspect, Graphic3d_NOM_SHINY_PLASTIC
from OCC.MeshVS import MeshVS_Mesh, MeshVS_DA_InteriorColor, MeshVS_DA_DisplayNodes, MeshVS_DA_EdgeColor, \
    MeshVS_DA_ShowEdges, MeshVS_DMF_Shading, MeshVS_DA_FrontMaterial, \
    MeshVS_DA_Reflection, MeshVS_DA_SmoothShading, MeshVS_DA_ColorReflection, MeshVS_ElementalColorPrsBuilder, \
    MeshVS_BP_ElemColor
from OCC.Quantity import Quantity_Color, Quantity_NOC_GRAY90, Quantity_NOC_GREENYELLOW, Quantity_NOC_SPRINGGREEN, \
    Quantity_TOC_HLS
from OCC.SMESH import SMESH_MeshVSLink, SMESH_Mesh, SMESH_Gen, SMESH_MeshEditor
from OCC.SMESHDS import SMESHDS_Mesh
from OCC.TopoDS import TopoDS_Shape
from OCC.gp import gp_Pnt

from OCCUtils.Construct import make_edge
from data_io.ImportType import ImportType
from data_io.LoadedData import LoadedData
from gcode.gcode_visualizer.VRepRapStates import VRepRapStates
from globals import ROOT_FOLDER_PATH
from utils import list_to_compound_shape


class LDGCode(LoadedData):
    """
    Takes a list of TopoDS_Edges as new_data input
    """

    def __init__(self, new_data: SMESH_Mesh or None, list_travels: list, name: str, import_type: ImportType,
                 filepath=None):
        super().__init__(new_data, name, import_type, filepath)
        self.travels = list_travels
        self.ais_object_travels = None
        self._list_speeds = []  # list of print head speeds per extrusion
        self.gcode_lines = []  # NOTUSED: all data of the GCode file in form of a list of GCodeLine objects. Not by layers
        self.gcode_layers = []
        self.speed_min = 0
        self.speed_max = 0
        self.layer_cnt = 0  # total layer count
        self._list_speeds_extrusions = []  # only used internally for 3D representation as temp storage
        self.extrusion_dim = np.array([0.4, 0.2], dtype=np.double)  # (nozzle_diameter, layer_height) For visualisation
        self.extrusion_color = Quantity_NOC_SPRINGGREEN
        self.travels_color = Quantity_NOC_GREENYELLOW

    def get_shape(self) -> TopoDS_Shape:
        raise NotImplementedError("Can not get TopoDS_Shape of loaded G-Code object!")
        # return list_to_compound_shape(self.data)

    def set_list_speeds(self, new_speed_list: list):
        self._list_speeds = new_speed_list
        if len(new_speed_list) > 0:
            self.speed_min = min(self._list_speeds)
            self.speed_max = max(self._list_speeds)

    def get_speeds(self) -> list:
        return self._list_speeds

    def transform(self, trans_rot):
        raise NotImplementedError("Can not transform loaded G-Code object!")

    def get_position(self):
        raise NotImplementedError("Can not get position of loaded G-Code object!")

    def speed_to_hue(self, speed_f):
        """
        max min speed and max speed to gradient between red-blue in HSL color space

        :param speed_f:
        :return:
        """
        return int(240.0 + ((0.0 - 240.0) / (self.speed_max - self.speed_min)) * (speed_f - self.speed_min))

    def plot_layer_cut_2d(self, from_layer: int, to_layer: int, from_line: int, to_line: int, xz_section=True,
                          linewidth_multiplier=20):
        """
        Plot a XZ or YZ cut through the G-Code layers displaying some fancy internal parameters

        :param linewidth_multiplier: Multiplied with _layer_height value to adjust line width
        :param xz_section: Make section in XZ-Plane. If False make section in YZ-Plane
        :param from_layer: Start layer index
        :param to_layer: End layer index
        :param from_line: Start G-Code extrusion line inside layer
        :param to_line: End G-Code extrusion line inside layer
        :return: Some fancy graphics
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('equal')

        _x = []
        _y = []
        _z = []
        _speed = []
        _layer_height = []
        _local_layer_indx = []
        _normals = []
        _extrusion_per_height = []
        for layer_index in range(from_layer, to_layer + 1):
            layer_c = self.gcode_layers[layer_index]
            for line_index in range(from_line, to_line + 1):
                line = layer_c.gcode_lines[line_index]
                # assert isinstance(line, GCodeLine)
                if line.move is not None and line.move.state is VRepRapStates.PRINTING:
                    # assert isinstance(line.move, VRepRapMove)
                    _x.append([line.move.x_s, line.move.x_e])
                    _y.append([line.move.y_s, line.move.y_e])
                    _z.append([line.move.z_s, line.move.z_e])
                    _speed.append(line.move.speed)
                    _layer_height.append(line.layer_height)
                    _local_layer_indx.append(line.local_layer_indx)
                    _normals.append(line.move.get_normal())
                    _extrusion_per_height.append(
                        line.move.extrusion_rate / line.move.get_length_3d())

        assert len(_x) == len(_y) == len(_z)

        color_param = _extrusion_per_height
        # cmap = plt.get_cmap('rainbow')
        cmap = plt.get_cmap('viridis')
        norm = matplotlib.colors.Normalize(vmin=min(color_param), vmax=max(color_param))

        _second_axis = _x
        _label = "X [mm]"
        if not xz_section:
            _second_axis = _y
            _label = "Y [mm]"
        plt.xlabel(_label)
        plt.ylabel("Z [mm]")
        for i in range(len(_second_axis)):
            ax.plot(_second_axis[i], _z[i], color=cmap(norm(color_param[i])),
                    linewidth=linewidth_multiplier * _layer_height[i])
            if _normals[i] is not None:
                normal = _normals[i]
                x_n = (_second_axis[i][0] + _second_axis[i][1]) / 2
                z_n = (_z[i][0] + _z[i][1]) / 2
                _n_dir = normal[0]
                if not xz_section:
                    _n_dir = normal[1]
                ax.quiver(x_n, z_n, _n_dir, normal[2], color=(0, 0, 0, 0.5))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, label="Extrusion rate/Millimeter")
        # ax.legend()
        plt.show()
        # Axes3D.plot()

    def set_list_gcode_lines(self, raw_gcode: list):
        self.gcode_lines = raw_gcode

    def set_list_gcode_layers(self, ngcode_layer: list):
        self.gcode_layers = ngcode_layer

    def set_total_layer_count(self, layer_cnt):
        self.layer_cnt = layer_cnt
