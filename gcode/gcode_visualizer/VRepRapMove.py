from math import sqrt

import numpy as np
from OCC.TopoDS import TopoDS_Edge
from OCC.gp import gp_Pnt

from OCCUtils.Construct import make_edge
from gcode.gcode_visualizer.VRepRapStates import VRepRapStates


class VRepRapMove:
    """
    Represents a movement of the printhead from one point to another
    """
    def __init__(self, x_start, y_start, z_start, x_end, y_end, z_end, speed, state: VRepRapStates):
        self.x_s = x_start
        self.y_s = y_start
        self.z_s = z_start

        self.x_e = x_end
        self.y_e = y_end
        self.z_e = z_end

        self.state = state
        self.speed = speed
        self.extrusion_rate = 0

        self._normal = None  # is a triple(x, y, z) that represents the normal vector

    def set_extrusion_rate(self, new_extrusion_rate: float):
        self.extrusion_rate = new_extrusion_rate

    def set_start_pnt(self, x: float, y: float, z: float):
        self.x_s = x
        self.y_s = y
        self.z_s = z

    def set_end_pnt(self, x: float, y: float, z: float):
        self.x_e = x
        self.y_e = y
        self.z_e = z

    def set_speed(self, new_speed: float):
        self.speed = new_speed

    def get_topods_edge(self) -> TopoDS_Edge:
        return make_edge(gp_Pnt(self.x_s, self.y_s, self.z_s), gp_Pnt(self.x_e, self.y_e, self.z_e))

    def get_topods_edge_projected(self) -> TopoDS_Edge or None:
        """
        Create an edge from the move that lies on the XY-Plane

        :return: A TopoDS_Edge representing the move
        """
        if not (self.x_s == self.x_e and self.y_s == self.y_e):
            return make_edge(gp_Pnt(self.x_s, self.y_s, 0), gp_Pnt(self.x_e, self.y_e, 0))
        else:
            # Edge would have length 0
            return None

    def get_length_xy(self) -> float:
        return sqrt((self.x_s - self.x_e)**2 + (self.y_s - self.y_e)**2)

    def get_length_3d(self) -> float:
        return sqrt((self.x_s - self.x_e)**2 + (self.y_s - self.y_e)**2 + (self.z_s - self.z_e)**2)

    def start_to_ndarray(self) -> np.ndarray:
        return np.array([self.x_s, self.y_s, self.z_s])

    def end_to_ndarray(self) -> np.ndarray:
        return np.array([self.x_e, self.y_e, self.z_e])

    def set_normal(self, _new_normal: (float, float, float)):
        self._normal = _new_normal

    def get_normal(self) -> (float, float, float):
        return self._normal

    def to_gcode(self, _comment="", _digits=4, x_changed=True, y_changed=True, z_changed=True) -> str:
        """
        Generates the actual gcode for writing to the output file. Change the syntax for how to output normals here if
        required.

        :param _comment:
        :param _digits:
        :param x_changed:
        :param y_changed:
        :param z_changed:
        :return:
        """
        if _comment is None:
            formatted_comment = ""
        else:
            formatted_comment = f" ; {_comment}"
        x_out = f" X{self.x_e:{_digits}f}"
        y_out = f" Y{self.y_e:{_digits}f}"
        z_out = f" Z{self.z_e:{_digits}f}"
        if not x_changed:
            x_out = ""
        if not y_changed:
            y_out = ""
        if not z_changed:
            z_out = ""

        if self.state is VRepRapStates.PRINTING:
            if self._normal is None:
                # No normal vector computed - only output x, y, z
                return f"G1{x_out}{y_out}{z_out} " \
                    f"E{self.extrusion_rate:{_digits}f}{formatted_comment}\n"
            else:
                # Output x, y, z and the normal vector as N O R
                return f"G1{x_out}{y_out}{z_out} " \
                    f"E{self.extrusion_rate:{_digits}f} N{self._normal[0]:{_digits}f} O{self._normal[1]:{_digits}f} " \
                    f"R{self._normal[2]:{_digits}f}{formatted_comment}\n"
        else:
            return f"G1{x_out}{y_out}{z_out} " \
                f"F{self.speed:{_digits}f}{formatted_comment}\n"  # traveling
