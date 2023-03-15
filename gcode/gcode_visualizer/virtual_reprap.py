import os
import re
import time

import numpy as np
from OCC.SMESH import SMESH_Gen, SMESH_MeshEditor
from OCC.SMESHDS import SMESHDS_Mesh
from OCC.gp import gp_Pnt

from OCCUtils.core_geometry_utils import make_edge
from data_io.ImportType import ImportType
from data_io.LDGCode import LDGCode
from gcode.gcode_visualizer.GCodeLayerC import GCodeLayerC
from gcode.gcode_visualizer.GCodeLine import GCodeLine
from gcode.gcode_visualizer.VRepRapMove import VRepRapMove
from gcode.gcode_visualizer.VRepRapStates import VRepRapStates

# TODO: Make this a VirtualRepRap class with variables etc. globals are shit!

use_relative_extrusion_parameters = True
old_x = 0
old_y = 0
old_z = 0
old_e = 0
x = 0
y = 0
z = 0
e = 0
f = 0
x_home = 0
y_home = 0
z_home = 0
extrusion_length_filament = 0
n_x = None
n_y = None
n_z = None

layer_num_regex = re.compile(r'.*layer_num=(\d*)')
layer_z_regex = re.compile(r'.*layer_z=(\d+\.\d+)')
simplify3d_allinone_regex = re.compile(r'.*layer (\d*), Z = (\d+\.\d+)')
ideamaker_z_regex = re.compile(r'Z:(\d+\.\d+)')
ideamaker_layer_regex = re.compile(r'LAYER:(\d*)')


def cross_3d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Custom implementation of the 3D cross product that is way faster than numpy (~88% faster)

    :return: Result of cross product of two 3D vectors
    """
    return np.array([v1[1] * v2[2] - v1[2] * v2[1],
                     v1[2] * v2[0] - v1[0] * v2[2],
                     v1[0] * v2[1] - v1[1] * v2[0]])


def home_reset(list_words):
    global x, y, z, e, x_home, y_home, z_home
    for a_word in list_words:
        if a_word.letter == 'E':
            e = a_word.value
        if a_word.letter == 'X':
            x_home = x - a_word.value
        elif a_word.letter == 'Y':
            y_home = y - a_word.value
        elif a_word.letter == 'Z':
            z_home = z - a_word.value


def process_lin_move(list_words, prev_state: VRepRapStates) -> VRepRapStates:
    """
    Sets current state of printer - checks if we would print something

    :param list_words: list of words/commands that where extracted by G-Code parser
    :param prev_state: previous state of the printer
    :return: A VRepRapState
    """
    global x, y, z, e, f, old_x, old_y, old_z, old_e, x_home, y_home, z_home, extrusion_length_filament, \
        use_relative_extrusion_parameters, n_x, n_y, n_z
    foundx = False
    foundy = False
    foundz = False
    founde = False
    old_x = x
    old_y = y
    old_z = z
    old_e = e
    for a_word in list_words:
        if a_word.letter == 'X':
            foundx = True
            x = x_home + a_word.value
        elif a_word.letter == 'Y':
            foundy = True
            y = y_home + a_word.value
        elif a_word.letter == 'Z':
            foundz = True
            z = z_home + a_word.value
        elif a_word.letter == 'E':
            e = a_word.value
            founde = True
            # if use_relative_extrusion_parameters:
            #     extrusion_length_filament = old_e + e
            # else:
            #     extrusion_length_filament = e - old_e
            # if extrusion_length_filament > 0:
            #     isextruding = True
            # else:
            #     isextruding = False
        elif a_word.letter == 'F':
            f = a_word.value
        elif a_word.letter == 'N':
            n_x = a_word.value
        elif a_word.letter == 'O':
            n_y = a_word.value
        elif a_word.letter == 'R':
            n_z = a_word.value
    # decide what kind of movement/operation it was
    if (foundx or foundy) and founde and e > 0:
        # RepRap is only extruding if G0 command is supplied with an extrusion rate that gets interpolated along line
        return VRepRapStates.PRINTING
    elif founde and e < 0:
        return VRepRapStates.RETRACTING
    elif founde and e > 0:
        return VRepRapStates.EXTRUDING
    elif foundz and not foundx and not foundy and prev_state is VRepRapStates.LIFTED_TRAVELING:
        return VRepRapStates.NOZZLE_LOWER
    # elif (foundx or foundy) and prev_state is VRepRapStates.NOZZLE_LIFT:
    #     return VRepRapStates.LIFTED_TRAVELING
    # elif foundz and not foundx and not foundy:
    #     return VRepRapStates.NOZZLE_LIFT
    elif foundx or foundy or foundz:
        return VRepRapStates.TRAVELING
    else:
        return VRepRapStates.CHANGING_PARAM


def create_move(x_start, x_end, y_start, y_end, z_start, z_end, speed, state: VRepRapStates):
    """
    Redundant to create line but only creates new move

    :return:
    """
    if not (x_start == x_end and y_start == y_end and z_start == z_end):
        return VRepRapMove(x_start, y_start, z_start, x_end, y_end, z_end, speed, state)
    else:
        return None


def create_line(x_start, x_end, y_start, y_end, z_start, z_end, speed, state: VRepRapStates) -> tuple:
    """
    Creates a edge/line with OCC-API at location of the extrusion. Is later displayed as G-Code

    :param speed:
    :param state:
    :param x_start:
    :param x_end:
    :param y_start:
    :param y_end:
    :param z_start:
    :param z_end:
    :return:
    """
    # Sometimes slicers create "extrusions" that start and end at the same point. This would make OCCT crash
    if not (x_start == x_end and y_start == y_end and z_start == z_end):
        return VRepRapMove(x_start, y_start, z_start, x_end, y_end, z_end, speed, state), \
               make_edge(gp_Pnt(x_start, y_start, z_start), gp_Pnt(x_end, y_end, z_end))
    else:
        return None, None


def add_mesh_from_line(_start: np.ndarray, _end: np.ndarray, _size_params: np.ndarray, _mesh_desc: SMESHDS_Mesh):
    """
    NOT USED ANYMORE: MESH NOW GENERATED INSIDE LDGCODE ON DISPLAY
    Creates a 3D mesh-pipe from two points. The start and end point of an extrusion.

    :param _start: Start point of extrusion [x, y, z] as double
    :param _end: End point of extrusion [x, y, z] as double
    :param _size_params: [width, height] of extrusion in x/y direction
    :param _mesh_desc: SMESHDS_Mesh of the mesh
        You can get it eg. when creating a new and empty mesh
        aMeshGen = SMESH_Gen()
        aMesh = aMeshGen.CreateMesh(0, True)
        e = SMESH_MeshEditor(aMesh)
        mesh_desc = e.GetMeshDS()
    """
    direction = _end - _start
    direction_norm = direction / np.linalg.norm(direction)
    _start = -_size_params[0] / 2 * direction_norm + _start
    _end = _size_params[0] / 2 * direction_norm + _end
    vertical_vec = np.array([0, 0, -1])  # vector of the extrusion direction in z (downwards)
    start_center = np.array([_start[0], _start[1], _start[2] - (_size_params[1] / 2)])
    end_center = np.array([_end[0], _end[1], _end[2] - (_size_params[1] / 2)])
    # numpy cross product is relatively slow for simple 3D vectors. We use our implementation to speed things up.
    # horizontal_vec = np.cross(direction, vertical_vec)  # vector that contains s2/e2 & s4/e4 in direction of broadness
    horizontal_vec = cross_3d(direction, vertical_vec)
    horizontal_vec_norm = horizontal_vec / np.linalg.norm(horizontal_vec)

    s2 = -_size_params[0] / 2 * horizontal_vec_norm + start_center
    s4 = _size_params[0] / 2 * horizontal_vec_norm + start_center

    s1 = _mesh_desc.AddNode(_start[0], _start[1], _start[2])
    s2 = _mesh_desc.AddNode(s2[0], s2[1], s2[2])
    s3 = _mesh_desc.AddNode(_start[0], _start[1], _start[2] - _size_params[1])
    s4 = _mesh_desc.AddNode(s4[0], s4[1], s4[2])

    e2 = -_size_params[0] / 2 * horizontal_vec_norm + end_center
    e4 = _size_params[0] / 2 * horizontal_vec_norm + end_center

    e1 = _mesh_desc.AddNode(_end[0], _end[1], _end[2])
    e2 = _mesh_desc.AddNode(e2[0], e2[1], e2[2])
    e3 = _mesh_desc.AddNode(_end[0], _end[1], _end[2] - _size_params[1])
    e4 = _mesh_desc.AddNode(e4[0], e4[1], e4[2])

    _mesh_desc.AddFace(s1, s2, e2, e1)  # face 12
    _mesh_desc.AddFace(s2, s3, e3, e2)  # face 23
    _mesh_desc.AddFace(s3, s4, e4, e3)  # face 34
    _mesh_desc.AddFace(s4, s1, e1, e4)  # face 41


def set_current_layer_params(_aline: GCodeLine, _current_layer, _current_layer_z, _old_layer_z) -> (GCodeLine, int, float, float):
    """
    Extracts layer number and current Z height. Needs specific G-Code comments. Simpily3D, ideaMaker and Sli3r supported for now.
    Simplify3D G-Code comment style:  ; layer 125, Z = 45.360
    Slic3r G-Code comment style:  ; layer_num=125, layer_z=45.360
    ideaMaker G-Code comment style: ;LAYER:XX ; Z:XX.XX
    SuperSlicer G-Code comment style: mix of ideaMaker and Slic3r

    :param _aline:  The G-Code line to search the comment section
    :param _current_layer:  The current layer number. If nothing found this will be returned unchanged
    :param _current_layer_z: The current layer z height. If nothing found this will be returned unchanged
    :param _old_layer_z: The old layer z height. If nothing found this will be returned unchanged
    :return: Tuple(aline, current layer index, current layer z, old layer z)
    """
    simplify_regex_already_matched = False
    if _aline.comment:
        # Check for layer number
        matches = simplify3d_allinone_regex.match(_aline.comment)
        if matches and matches.group(1) and matches.group(2):
            _current_layer = int(matches.group(1))
            _old_layer_z = _current_layer_z
            _current_layer_z = float(matches.group(1))
            simplify_regex_already_matched = True   # we already got the layer z height, do not re-regex in next part
        else:
            matches = ideamaker_layer_regex.match(_aline.comment)
            if matches and matches.group(1):
                _current_layer = int(matches.group(1))
            else:
                matches = layer_num_regex.match(_aline.comment)
                if matches and matches.group(1):
                    _current_layer = int(matches.group(1))

        # Check for layer z-height
        matches = ideamaker_z_regex.match(_aline.comment)
        if matches and matches.group(1):
            _old_layer_z = _current_layer_z
            _current_layer_z = float(matches.group(1))
        else:
            matches = layer_z_regex.match(_aline.comment)
            if matches and matches.group(1):
                _old_layer_z = _current_layer_z
                _current_layer_z = float(matches.group(1))
            else:
                if not simplify_regex_already_matched:
                    matches = simplify3d_allinone_regex.match(_aline.comment)
                    if matches and matches.group(1) and matches.group(2):
                        _current_layer = int(matches.group(1))
                        _old_layer_z = _current_layer_z
                        _current_layer_z = float(matches.group(1))

        # # Check for Simpily3D layer information first
        # matches = simplify3d_allinone_regex.match(_aline.comment)
        # if matches and matches.group(1) and matches.group(2):
        #     _current_layer = int(matches.group(1))
        #     _old_layer_z = _current_layer_z
        #     _current_layer_z = float(matches.group(1))
        # else:
        #     # Check for IdeaMaker syntax
        #     matches = ideamaker_layer_regex.match(_aline.comment)
        #     if matches and matches.group(1):
        #         _current_layer = int(matches.group(1))
        #     matches = ideamaker_z_regex.match(_aline.comment)
        #     if matches and matches.group(1):
        #         _old_layer_z = _current_layer_z
        #         _current_layer_z = float(matches.group(1))
        #     else:
        #         # Check for -> Slic3r/generic syntax
        #         matches = layer_num_regex.match(_aline.comment)
        #         if matches and matches.group(1):
        #             _current_layer = int(matches.group(1))
        #         matches = layer_z_regex.match(_aline.comment)
        #         if matches and matches.group(1):
        #             _old_layer_z = _current_layer_z
        #             _current_layer_z = float(matches.group(1))
    _aline.set_layer(_current_layer)
    _aline.set_layer_z(_current_layer_z)
    _aline.set_layer_height(_current_layer_z - _old_layer_z)    # TODO: Check why 0 or not right at all -> only one idication per layer change is allowed
    return _aline, _current_layer, _current_layer_z, _old_layer_z


def process_gcode(file_path, layer_height, nozzle_diameter, start=None, end=None) -> LDGCode:
    """
    Processes a gcode file. Returns a list of TopoDS_Shapes representing the extrusions. Chooseable start & endpoint
    inside gcode file

    :param file_path: Path to the g-code file
    :param layer_height: Height of each layer
    :param nozzle_diameter: The diameter of the nozzle. Defines the width of the extrusion
    :param start: Start line for the parser in g-code file (use to display portion of g-code)
    :param end: End line for the parser in g-code file (use to display portion of g-code)
    :return: (extrusion mesh, travel moves, list_speeds, list_moves)
        extrusion mesh: type SMESH_MESH
        travel moves: list containing TopoDS_Shapes
    """
    global x, y, z, e, old_e, f, old_x, old_y, old_z, extrusion_length_filament, use_relative_extrusion_parameters, \
        n_x, n_y, n_z
    old_x = 0
    old_y = 0
    old_z = 0
    old_e = 0
    x = 0
    y = 0
    z = 0
    e = 0
    f = 0
    state = VRepRapStates.STANDBY
    current_layer_num = 0
    layer_z = 0
    old_layer_z = 0
    prev_layer_num = current_layer_num
    list_travels = []
    list_speeds = []
    list_layers = []  # all layers of GCode file - consists of lists of GCodeLines (list_lines)
    list_lines = []  # all data of the GCode file in form of a list of GCodeLine objects
    # Create a new an empty mesh object that represents the extrusions
    # a_mesh_gen = SMESH_Gen()
    # a_mesh = a_mesh_gen.CreateMesh(0, True)
    # _mesh_editor = SMESH_MeshEditor(a_mesh)
    # mesh_desc = _mesh_editor.GetMeshDS()
    extrusion_dimensions = np.array([nozzle_diameter, layer_height], dtype=np.double)
    last_update = time.time()

    print("Loading gcode file...")
    ignore_first = False
    with open(file_path, 'r', encoding="latin-1") as file:
        i = 0
        for i, l in enumerate(file):
            pass
        i = i + 1  # counted the number of lines in the file
        file.seek(0)
        totaltodo = i
        for cnt, line in enumerate(file):
            if cnt <= totaltodo:
                aline = GCodeLine(line)
                aline, current_layer_num, layer_z, old_layer_z = set_current_layer_params(aline, prev_layer_num,
                                                                                          layer_z, old_layer_z)
                if aline.block.words:
                    for _word in aline.block.words:
                        if _word.letter == 'G' and (_word.value == 1 or _word.value == 0):
                            state = process_lin_move(aline.block.words, state)
                            if state is VRepRapStates.PRINTING:
                                # add_mesh_from_line(np.array([old_x, old_y, old_z], dtype=np.double),
                                #                    np.array([x, y, z], dtype=np.double), extrusion_dimensions,
                                #                    mesh_desc)
                                list_speeds.append(f)
                                _move = create_move(old_x, x, old_y, y, old_z, z, f, state)
                                if _move is not None:
                                    _move.set_extrusion_rate(e)
                                    if state is VRepRapStates.PRINTING and n_x is not None and n_y is not None \
                                            and n_z is not None:
                                        _move.set_normal((n_x, n_y, n_z))
                                    aline.set_move(_move)
                            elif state is VRepRapStates.TRAVELING or state is VRepRapStates.NOZZLE_LIFT \
                                    or state is VRepRapStates.NOZZLE_LOWER or state is VRepRapStates.LIFTED_TRAVELING:
                                _move, new_line = create_line(old_x, x, old_y, y, old_z, z, f, state)
                                list_speeds.append(f)
                                if _move is not None:
                                    aline.set_move(_move)
                                    list_travels.append(new_line)
                        elif _word.letter == 'G' and _word.value == 92:
                            home_reset(aline.block.words)
                        elif _word.letter == 'M' and _word.value == 82:
                            use_relative_extrusion_parameters = False
                            # print("Absolute extrusion parameters")
                        elif _word.letter == 'M' and _word.value == 83:
                            use_relative_extrusion_parameters = True
                            # print("Relative extrusion parameters")
                if prev_layer_num == current_layer_num:
                    list_lines.append(aline)
                else:
                    # add all prev lines to a new layer
                    list_layers.append(GCodeLayerC(list_lines, prev_layer_num, layer_z))
                    # clear list of lines to be ready to take up lines for new layer, but do not delete the references
                    list_lines = [aline]
                prev_layer_num = current_layer_num
                if (time.time() - last_update) > 0.250:  # Updating the console takes a long time. Only to at some point
                    print(f"\r\tProcessed line {cnt}/{totaltodo} layer: {current_layer_num}", end="")
                    last_update = time.time()
    print(f"\r\tProcessed line {cnt}/{totaltodo} layer: {current_layer_num} layer_z: {layer_z}")
    list_layers.append(GCodeLayerC(list_lines, prev_layer_num, layer_z))
    if ignore_first and len(list_travels) > 1:
        del list_travels[0]
    _new_gcode = LDGCode(None, list_travels, os.path.basename(file_path), ImportType.gcode, file_path)
    _new_gcode.set_list_speeds(list_speeds)
    _new_gcode.set_list_gcode_layers(list_layers)
    _new_gcode.set_list_gcode_lines(list_lines)
    _new_gcode.set_total_layer_count(current_layer_num + 1)
    _new_gcode.extrusion_dim = extrusion_dimensions
    return _new_gcode
