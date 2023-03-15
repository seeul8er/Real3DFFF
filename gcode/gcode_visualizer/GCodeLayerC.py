class GCodeLayerC:
    """
    G-Code layer class used by Christl implementation of CLFFF algorithm
    """

    def __init__(self, gcode_lines: list, layer_indx: int, layer_z: int):
        self.gcode_lines = gcode_lines  # List of GCodeLine objects
        self.layer_indx = layer_indx  # Index starting from 0
        self.layer_z = layer_z  # total z coordinate of the layer e.g. 123 mm
