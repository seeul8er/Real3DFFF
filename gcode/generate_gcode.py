import os
import subprocess

from data_io.ImportType import ImportType
from data_io.LDMesh import LDMesh
from data_io.LDShape import LDShape
from data_io.LoadedData import LoadedData
from data_io.exporters import generate_mesh, export_stl
from gcode.SlicingEngine import SlicingEngine
from globals import LINEAR_DEFLECTION, ANGULAR_DEFLECTION, CURA_APP_PATH, CURA_CONFIG_FILE, SLICER_APP_PATH, \
    SLICER_CONFIG_FILE, SLICER_PE_APP_PATH, SLICER_PE_CONFIG_FILE, TMP_FOLDER_PATH


def _stl_export_necessary(geometry: LoadedData) -> bool:
    """
    Checks whether a STL file for the geometry already exits on disk

    :return: True if we need to create an STL file first
    """
    if geometry.import_type is ImportType.mesh or geometry.import_type is ImportType.stl:
        # Check if we know and can use the location of the geometry on disk
        if not geometry.filepath == "" and \
                (geometry.filepath.endswith(".stl") or geometry.filepath.endswith(".STL")):
            return False
    return True


def generate_gcode(geometry: LoadedData, engine: SlicingEngine, overwrites: dict, slicer_path="",
                   slicer_conf="") -> str:
    """
    1. Exports the geometry to STL
    2. With :param slice_in_place set to True: Adds a build plate offset in z to the predefined settings file
    3. Calls a slicing engine with a predefined settings file to generate the gcode for the geometry
    4. Reloads the G-Code and displays it in the UI.

    :param slicer_conf: Path to a valid config file for the specified slicing software
    :param slicer_path: The to the slicers executable. If not set path stored in globals will be used
    :param overwrites: Dictionary of command line arguments with its values that will overwrite the default config
    :param geometry: The geometry as LoadedData that you want to generate to G-Code for
        place geometry on built plate
    :param engine: Enum defining the desired slicing engine
    :return: The path to the gcode file
    """
    gcode_file_path = "real3d_fff_" + geometry.name[:-4] + ".gcode"
    gcode_file_path = os.path.join(TMP_FOLDER_PATH, gcode_file_path)
    if isinstance(geometry, LDShape):
        _mesh = generate_mesh(LINEAR_DEFLECTION, ANGULAR_DEFLECTION, geometry.get_shape())
        stl_file_path = export_stl(_mesh, filename=os.path.join(TMP_FOLDER_PATH, "temp_stl_for_gcode.stl"))
    elif isinstance(geometry, LDMesh):
        stl_file_path = geometry.get_stl_filepath(save_filepath=os.path.join(TMP_FOLDER_PATH, "temp_stl_for_gcode.stl"))
    else:
        raise TypeError

    if engine is SlicingEngine.SLIC3R:
        if slicer_path == "":
            slicer_path = SLICER_APP_PATH
        if slicer_conf == "":
            slicer_conf = SLICER_CONFIG_FILE
        overwrite_params = []
        for key, value in overwrites.items():
            overwrite_params.append("--" + str(key))
            overwrite_params.append(str(value))
        command_list = [slicer_path, '--dont-arrange', '--use-relative-e-distances', '--gcode-comments',
                        '--load', slicer_conf] + overwrite_params + ['--output', gcode_file_path, stl_file_path]
        try:
            command_list.remove('')  # otherwise it might not execute, sometimes there is no '' so we need to try:...
        except ValueError:
            pass
        print(' '.join(str(e) for e in command_list))
        subprocess.run(command_list, stderr=subprocess.STDOUT)
    elif engine is SlicingEngine.SLIC3R_PE:
        if slicer_path == "":
            slicer_path = SLICER_PE_APP_PATH
        if slicer_conf == "":
            slicer_conf = SLICER_PE_CONFIG_FILE
        overwrite_params = []
        for key, value in overwrites.items():
            overwrite_params.append("--" + str(key))
            overwrite_params.append(str(value))
        command_list = [slicer_path, '--dont-arrange', '--use-relative-e-distances', '--slice', '--no-gui',
                        '--gcode-comments', '--load', slicer_conf] + overwrite_params + ['--output', gcode_file_path,
                                                                                         stl_file_path]
        command_list.remove('')
        print(' '.join(str(e) for e in command_list))
        subprocess.run(command_list, close_fds=True, stderr=subprocess.STDOUT)
    elif engine is SlicingEngine.CURA:
        z_offset_override = '17'
        if slicer_path == "":
            slicer_path = CURA_APP_PATH
        if slicer_conf == "":
            slicer_conf = CURA_CONFIG_FILE
        subprocess.run([slicer_path, 'slice', '-j', slicer_conf, '-s', z_offset_override, '-o',
                        gcode_file_path, '-l', stl_file_path])
        raise NotImplementedError
    return gcode_file_path
