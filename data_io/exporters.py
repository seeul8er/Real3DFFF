import colorsys
import os
import shutil
import time
from importlib import util

from OCC.BRepMesh import BRepMesh_IncrementalMesh
from OCC.IFSelect import IFSelect_RetDone
from OCC.Interface import Interface_Static_SetCVal
from OCC.SMESH import SMESH_Mesh
from OCC.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.StlAPI import StlAPI_Writer
from OCC.TopoDS import TopoDS_Shape
from PyQt5.QtWidgets import QFileDialog
from math import pi

from CColors import CColors
from gcode.gcode_visualizer.VRepRapStates import VRepRapStates
from globals import show_ui, LINEAR_DEFLECTION, ANGULAR_DEFLECTION, ROOT_FOLDER_PATH


def export_shape_to_step(_new_shape=None, new_export_path="export.step", running_os='w'):
    """
    Helper to export any TopoDS_Shape to STEP format. Opens a file saving dialog where the user can specify a location.
    If "show_ui" is False: Uses :param new_export_path as the location for saving
    This function is not required by regular users. Used manly for debugging and verification purposes

    :param _new_shape: A TopoDS_Shape that you want to save on disk
    :param new_export_path: A path to a location where the file should be saved if "show_ui" is False
    :param running_os: For cross platform. Set to 'l' if Linux or 'w' if running on Windows
    """
    if _new_shape is not None:
        _mesh_preform = generate_mesh(LINEAR_DEFLECTION, ANGULAR_DEFLECTION, _new_shape)
        if show_ui:
            save_url = QFileDialog.getSaveFileUrl(caption="Save as *.STEP", filter="STEP files (*.stp *.step)")
            thepath = save_url[0].toString()
            if not thepath.endswith(".step"):
                if not thepath.endswith(".stp"):
                    thepath = thepath + ".step"
            if thepath.startswith("file:///") and running_os == 'w':  # windows - use of os package might be cleverer
                thepath = thepath[8:]
            elif thepath.startswith("file://") and running_os == 'l':  # linux - use of os package might be cleverer
                thepath = thepath[7:]
            export_step(_mesh_preform, filename=thepath)
        else:
            export_step(_mesh_preform, filename=new_export_path)
    else:
        print(CColors.FAIL + "Error: Can not export to STEP. No shape given!" + CColors.ENDC)


def export_shape_to_stl(_new_shape=None, new_export_path="export.stl", saveascii=False, running_os='w'):
    """
    Helper to export any TopoDS_Shape or SMESH_Mesh to STL format.
    Opens a file saving dialog where the user can specify a location.
    If "show_ui" is False: Uses :param new_export_path as the location for saving

    :param _new_shape: A TopoDS_Shape or SMESH_Mesh that you want to save on disk
    :param new_export_path: A path to a location where the file should be saved if "show_ui" is False
    :param saveascii: Use ASCII file format (bigger file size but readable by humans)
    :param running_os: For cross platform. Set to 'l' if Linux or 'w' if running on Windows
    """
    if _new_shape is not None:
        if isinstance(_new_shape, TopoDS_Shape):
            # before exporting to STL the shape needs to be converted to a mesh (triangles)
            _new_shape = generate_mesh(LINEAR_DEFLECTION, ANGULAR_DEFLECTION, _new_shape)
        if show_ui:
            save_url = QFileDialog.getSaveFileUrl(caption="Save as *.STL", filter="*.stl")
            thepath = save_url[0].toString()
            if not thepath.endswith(".stl"):
                thepath = thepath + ".stl"
            if thepath.startswith("file:///") and running_os == 'w':  # windows
                thepath = thepath[8:]
            elif thepath.startswith("file://") and running_os == 'l':  # linux
                thepath = thepath[7:]
            export_stl(_new_shape, filename=thepath, bool_asciimode=saveascii)
        else:
            export_stl(_new_shape, filename=new_export_path, bool_asciimode=saveascii)
    else:
        print(CColors.FAIL + "Error: Can not export STL. No shape given!" + CColors.ENDC)


def save_as(existing_file_on_disk: str, qfiledialog_filter="G-Code (*.gcode)", running_os='w'):
    """
    Shows save dialog UI to let user select a location for the file he wants to save.
    Copies an existing file on disk to a new location. Used to save generated gcode files etc. to a user defined
    location. These files usually already exist on disk as temporary files and just need to be saved somewhere else.

    :param existing_file_on_disk: Path to file that you want to save somewhere else
    :param qfiledialog_filter: A filter that is passed to the QFileDialog class.
    :param running_os: For cross platform. Set to 'l' if Linux or 'w' if running on Windows
    """
    save_url = QFileDialog.getSaveFileUrl(caption="Select new folder & name", filter=qfiledialog_filter)
    new_file_path = save_url[0].toString()
    if new_file_path.startswith("file:///") and running_os == 'w':  # windows - use of os package might be cleverer
        new_file_path = new_file_path[8:]
    elif new_file_path.startswith("file://") and running_os == 'l':  # linux - use of os package might be cleverer
        new_file_path = new_file_path[7:]
    if new_file_path:
        print("Coping to: " + new_file_path + "...", end="", flush=True)
        shutil.copy2(existing_file_on_disk, new_file_path)
        print("Done!")


def generate_mesh(lin_deflection, ang_deflection, shape_shape, display_mesh=False, display_viewer3d=None):
    """
    Generates a triangulated mesh from given shape

    :param display_mesh: Draw mesh on screen after finished generating
    :param lin_deflection: linear deflection [mm]
    :param ang_deflection: angular deflection [rad]?!
    :param shape_shape: The shape that should be triangulated
    :param display_viewer3d: OCE 3d display instance
    :return: The triangulated mesh
    """
    print(f"Generating triangulation - linear: {lin_deflection} angular: +{ang_deflection}... ", end="", flush=True)
    mesh = BRepMesh_IncrementalMesh(shape_shape, lin_deflection, False, ang_deflection, True)
    mesh.SetParallel(True)
    mesh.Perform()
    assert mesh.IsDone()
    if display_mesh and display_viewer3d is not None:
        display_viewer3d.DisplayShape(mesh.Shape())
    print(CColors.OKBLUE + "Done!" + CColors.ENDC)
    return mesh.Shape()


def export_stl(shape_export, filename="exported.stl", bool_asciimode=False):
    """
    Export a TopoDS_Shape or SMESH_Mesh to STL using the STL exporter libs provided by OCE/SMESH

    :param shape_export: The TopoDS_Shape that should be exported. Must be meshed!
    :param filename: The file name of the exported geometry
    :param bool_asciimode: Save STL in ASCII mode (readable, bigger file)
    :return: On success: the relative export file path | On failure: None
    """
    print("Exporting STL: " + filename + "... ", end="", flush=True)
    if isinstance(shape_export, TopoDS_Shape):
        stl_writer = StlAPI_Writer()
        stl_writer.SetASCIIMode(bool_asciimode)
        stl_writer.Write(shape_export, filename)
    elif isinstance(shape_export, SMESH_Mesh):
        shape_export.ExportSTL(filename, bool_asciimode)
    else:
        print("Error: Can not export to STL. Specified shape is not a supported geometry format/object!")
        return None
    print("Done!")
    return filename


def export_step(a_shape, filename, application_protocol="AP203"):
    """ exports a shape to a STEP file

    :param a_shape: the topods_shape to export (a compound, a solid etc.)
    :param filename: the filename
    :param application_protocol: "AP203" or "AP214"
    """
    #  Copyright 2018 Thomas Paviot (tpaviot@gmail.com)
    #  This function is part of pythonOCC.
    #
    # pythonOCC is free software: you can redistribute it and/or modify
    # it under the terms of the GNU Lesser General Public License as published by
    # the Free Software Foundation, either version 3 of the License, or
    # (at your option) any later version.
    #
    # pythonOCC is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU Lesser General Public License for more details.
    ##
    # You should have received a copy of the GNU Lesser General Public License
    # along with pythonOCC.  If not, see <http://www.gnu.org/licenses/>

    # a few checks
    assert not a_shape.IsNull()
    assert application_protocol in ["AP203", "AP214IS"]
    if os.path.isfile(filename):
        print("Warning: %s file already exists and will be replaced" % filename)
    # creates and initialise the step exporter
    step_writer = STEPControl_Writer()
    Interface_Static_SetCVal("write.step.schema", "AP203")

    # transfer shapes and write file
    step_writer.Transfer(a_shape, STEPControl_AsIs)
    status = step_writer.Write(filename)

    assert status == IFSelect_RetDone
    assert os.path.isfile(filename)


def export_to_blend(gcode, export_path=os.path.join(ROOT_FOLDER_PATH, "real3dfff_gcode.blend"), _resolution=2,
                    default_extrusion_color=(0.01, 0.124, 0.556), default_travel_color=(1.0, 0.0, 0.0),
                    default_infill_color=(0.253, 0.420, 0.015), nozzle_diameter=0.4, start_layer=None, end_layer=None,
                    draw_travels=True, detect_infill=True, animated=True, scale_factor=0.01, material_by_speed=True):
    """
    To visualize the G-Code and render high quality images in Blender. Material can be based on speed or infill/perimeter
    Layers can be animated: 1 layer/frame

    :param gcode: LDCode file
    :param nozzle_diameter: Determines width and height of extrusion if no layer height is set
    :param scale_factor: Scale all coords and lengths by this value to better fit inside the blender scene
    :param detect_infill: Create a separate material for G-Code lines with "infill" as part of the comment
    :param draw_travels: Create geometry to represent travel moves in blender file
    :param default_travel_color: Default diffuse travel color in Blender (red)
    :param default_extrusion_color: Default diffuse extrusion color in Blender (light blue)
    :param end_layer: End layer index
    :param start_layer: Start layer index
    :param animated: Animate: Make one layer per frame visible
    :param _resolution: 0: 4 Sided extrusions, Might not set higher than 4; (0, 2, 4) for best results
    :param material_by_speed: Material color by speed. Overwrites draw_travels parameter
    :param export_path: Path to .blend out file
    :return:
    """
    print(f"Saving to {export_path}")
    blender_avail = util.find_spec('bpy') is not None
    if blender_avail:
        import bpy
        bpy.ops.wm.read_homefile()  # reset scene
        prev_line = VRepRapStates.STANDBY
        list_sequence = []
        speed_materials = []
        speeds = list(set(gcode.get_speeds()))
        if material_by_speed:
            for i in range(len(speeds)):
                speed_mat = bpy.data.materials.new(f"SpeedMaterial{i}")
                speed_mat.diffuse_color = colorsys.hsv_to_rgb(gcode.speed_to_hue(speeds[i]) / 360, 0.87, 0.4)
                speed_mat.use_shadeless = False
                speed_materials.append(speed_mat)

        travel_mat = bpy.data.materials.new('TravelMaterial')
        travel_mat.diffuse_color = default_travel_color
        travel_mat.use_shadeless = True

        extrusion_mat = bpy.data.materials.new('ExtrusionMaterial')
        extrusion_mat.diffuse_color = default_extrusion_color
        extrusion_mat.use_shadeless = False

        infill_mat = bpy.data.materials.new('InfillMaterial')
        infill_mat.diffuse_color = default_infill_color
        infill_mat.use_shadeless = False

        def make_polyline(_list_sequence: list, _line, fallback_layer_height=(nozzle_diameter/2), travel_thickness=0.05,
                          _detect_infill=True):
            name = f"Movement{move_cnt}"
            the_line_data = bpy.data.curves.new(name=name, type='CURVE')
            the_line_data.dimensions = '3D'
            the_line_data.fill_mode = 'FULL'
            the_line_data.twist_mode = 'Z_UP'
            the_line_data.bevel_resolution = _resolution
            if _line.move.state is VRepRapStates.TRAVELING:
                the_line_data.bevel_depth = travel_thickness * scale_factor
            else:
                the_line_data.bevel_depth = fallback_layer_height * scale_factor
            polyline = the_line_data.splines.new('POLY')
            polyline.points.add(len(_list_sequence) - 1)
            for idx in range(len(_list_sequence)):
                polyline.points[idx].co = (_list_sequence[idx]) + (1.0,)

            blender_line = bpy.data.objects.new(name, the_line_data)
            if animated:
                blender_line.hide_render = True
                blender_line.hide = True
                blender_line.keyframe_insert(data_path="hide_render", frame=_line.layer_num)
                blender_line.keyframe_insert(data_path="hide", frame=_line.layer_num)
                blender_line.hide_render = False
                blender_line.hide = False
                blender_line.keyframe_insert(data_path="hide_render", frame=_line.layer_num + 1)
                blender_line.keyframe_insert(data_path="hide", frame=_line.layer_num + 1)
            blender_line['layer_height'] = _line.layer_height  # add custom properties that can be used for texture
            # blender_line['local_layer_indx'] = _line.local_layer_indx
            blender_line['layer_index'] = _line.layer_num
            blender_line['speed'] = _line.move.speed
            bpy.context.scene.objects.link(blender_line)
            blender_line.location = (0.0, 0.0, 0.0)
            if material_by_speed:
                if _line.move.state is VRepRapStates.TRAVELING:
                    blender_line.data.materials.append(travel_mat)
                else:
                    blender_line.data.materials.append(speed_materials[speeds.index(_line.move.speed)])
            else:
                if _line.move.state is VRepRapStates.PRINTING:
                    if _detect_infill and "infill" in _line.comment.lower():
                        blender_line.data.materials.append(infill_mat)
                    else:
                        blender_line.data.materials.append(extrusion_mat)
                elif _line.move.state is VRepRapStates.TRAVELING:
                    blender_line.data.materials.append(travel_mat)

        print("Creating 3D Representation")
        move_cnt = 0
        line_cnt = 0
        last_update = time.time()
        if start_layer is None or end_layer is None:
            start_layer = 0
            end_layer = gcode.gcode_layers[-1].layer_indx
        if end_layer > gcode.gcode_layers[-1].layer_indx:
            end_layer = gcode.gcode_layers[-1].layer_indx
        if animated:
            scn = bpy.context.scene
            scn.frame_start = 0
            scn.frame_end = end_layer-start_layer
        for layer_index in range(start_layer, end_layer + 1):
            layer = gcode.gcode_layers[layer_index]
            for line in layer.gcode_lines:
                if (time.time() - last_update) > 0.250:
                    print(f"\r\tProcessed {line_cnt}", end="", flush=True)
                    last_update = time.time()
                if line.move is not None:
                    if line.move.state is not VRepRapStates.TRAVELING or (
                            line.move.state is VRepRapStates.TRAVELING and draw_travels):
                        if material_by_speed:
                            if prev_line is VRepRapStates.STANDBY:
                                list_sequence.append((line.move.x_s * scale_factor, line.move.y_s * scale_factor,
                                                      line.move.z_s * scale_factor))
                                list_sequence.append((line.move.x_e * scale_factor, line.move.y_e * scale_factor,
                                                      line.move.z_e * scale_factor))
                            elif prev_line.move.speed is line.move.speed:
                                list_sequence.append((line.move.x_e * scale_factor, line.move.y_e * scale_factor,
                                                      line.move.z_e * scale_factor))
                            else:
                                make_polyline(list_sequence, prev_line, _detect_infill=detect_infill)
                                move_cnt += 1
                                list_sequence = [(line.move.x_s * scale_factor, line.move.y_s * scale_factor,
                                                  line.move.z_s * scale_factor),
                                                 (line.move.x_e * scale_factor, line.move.y_e * scale_factor,
                                                  line.move.z_e * scale_factor)]
                        else:
                            if prev_line is VRepRapStates.STANDBY:
                                list_sequence.append((line.move.x_s * scale_factor, line.move.y_s * scale_factor,
                                                      line.move.z_s * scale_factor))
                                list_sequence.append((line.move.x_e * scale_factor, line.move.y_e * scale_factor,
                                                      line.move.z_e * scale_factor))
                            elif prev_line.move.state is line.move.state:
                                list_sequence.append((line.move.x_e * scale_factor, line.move.y_e * scale_factor,
                                                      line.move.z_e * scale_factor))
                            else:
                                # Create line with same properties
                                make_polyline(list_sequence, prev_line, _detect_infill=detect_infill)
                                move_cnt += 1
                                list_sequence = [(line.move.x_s * scale_factor, line.move.y_s * scale_factor,
                                                  line.move.z_s * scale_factor),
                                                 (line.move.x_e * scale_factor, line.move.y_e * scale_factor,
                                                  line.move.z_e * scale_factor)]
                        prev_line = line
                line_cnt += 1
        if len(list_sequence) > 0:
            make_polyline(list_sequence, prev_line)
        print(f"\r\tProcessed {line_cnt}", flush=True)
        # update camera position to look into the build volume
        camera = bpy.data.objects['Camera']
        camera.rotation_euler[0] = 55 * pi / 180
        camera.rotation_euler[1] = 0 * pi / 180
        camera.rotation_euler[2] = -45 * pi / 180
        camera.location.x = 0
        camera.location.y = 0
        camera.location.z = 100 * scale_factor
        objs = bpy.data.objects
        objs.remove(objs["Cube"], True)  # get rid of default geometry
        print("Saving...")
        bpy.ops.wm.save_as_mainfile(filepath=export_path)
        print(f"Done! Saved to: {export_path}")
    else:
        print("FAIL: Blender package bpy is not available! Can not export to file")
