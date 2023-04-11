# <img src="real3dfff_logo.png" width="100" height="100" /> Real3DFFF

Traditional vs Curved-Layer Fused Filament Fabrication (CLFFF) using Real3DFFF

https://user-images.githubusercontent.com/24637325/230965579-060c6ff2-75d5-4293-90fa-5e9db6e54b03.mp4

# Features

*  Generate curved tool paths for FFF-Printers including normal vectors for 5-axis printing
*  Generate Preform geometries
*  Extract geometries for support only generation
*  Import/Export of G-Code, STL, STEP, IGES

# How does it work

*This section is WIP*

![part_to_clfff_gcode](https://user-images.githubusercontent.com/24637325/231271483-11810f2f-6010-46c9-97e7-523010124045.jpg)

## Abstract

The developed algorithm uses a Preform Geometry that represents an extruded shadow of the actual part to be printed. That Preform is sliced with a regular slicing software that creates annotations about current layer and z-height inside the generated G-Code for the Preform. That G-Code is then adjusted on a per-layer basis to match the final geometry of the to be printed part.  
The adjustment of a G-Code movement inside a layer includes the re-calculation of the z-coordinate based on the upper and lower surface coordinates in that specific area of the to be printed part. This requires a segmentation/splitting of G-Code segments into smaller segments to properly approximate the final geometry. In addition, the extrusion amount per segment and travel moves are recalculated to compensate for the change due to the z-adjustment of the layer.   
Since the total number of layers in the sliced Preform is identical to the number in the final output, the layer height in the final G-Code is variable (adaptive) based on the z-cross section of the to be printed geometry.  
During all following steps we use a coordinate system where the Z-axis points into the build direction while the XY-plane represents the build plate of the printer.

## The Preform

In this work, a preform refers to an auxiliary body. A preform is created from the original geometry. The preform results from an orthogonal projection of the original geometry into the XY plane. The projection, which is a like a shadow of the component, is extruded in the Z-direction. The amount of extrusion should be chosen to correspond to the largest amount of the cross-section in the Z-direction of the original geometry. Together with the chosen layer thickness during slicing, the amount of extrusion determines the number of layers in the final part.  
An important property of the preform is that it has a constant cross-section in Z and all information about overhangs and undercuts of the original part is lost.  
Preform geometries can be created in many ways. If the CAD data for the part to be printed is available, the preform can be created directly by projecting the outer edges in CAD. Other methods exist for tessellated geometries (STL). The application "Autodesk Netfabb", for example, provides such a function. The bottom side of the to be printed geometry is extruded in the positive Z-direction. A Boolean “common” intersection with a box will result in the desired preform geometry. The XY-dimension of the box must be at least equal to the XY-dimension of the part. The height of the box in Z determines the height of the preform. The box must be positioned above the highest point of the bottom surface of the part.

## Surface Coordinates

The algorithm ZGetter is used to obtain the Z-coordinates of the surface of a component at an arbitrary point (X, Y). Usually, two values are returned for a requested position. One for the bottom and one for the top of the component. In the following also called Z<sub>lower</sub> and Z<sub>upper</sub>. If the part contains undercuts or faulty geometry (e.g. in STL files), more than two intersection points may result. In this case, the two points with the lowest and the highest amount of Z are returned.  
The algorithm determines the intersection points by intersecting a straight line passing through the point (X, Y) and parallel to the Z-axis with the geometry to be printed. OCCT provides corresponding implementations for the generation of straight lines and the determination of intersection points with a 3D geometry.  
In addition to the Z coordinates, the normal vector on the top and bottom of the geometry can be determined at the sampled point (X, Y). For this purpose, the partial surface of the component on which, for example, the point (X, Y, Z<sub>lower</sub>) is located is determined. The normal vector of the surface at this point can then be calculated. Here, too, OCCT provides implementations for the determination of the surface and the normal vector at the point (X, Y). As a result, two additional normal vectors are obtained at the points (X, Y, Z<sub>lower</sub>) and (X, Y, Z<sub>upper</sub>). The vectors are normalised and it is ensured that they have the same orientation. By definition, this is in the negative Z direction.


![part_to_clfff_gcode_2d](https://user-images.githubusercontent.com/24637325/231271511-3bdea28e-3cfc-4a35-bfa9-0e63d22a08e0.jpg)


# Install
Real3DFFF requires a big set of dependencies, some of them being a bit dated.
To make things easier, a complete python environment is provided in addition to the code to get you started in no time.

1. Download the package from the GitHub release section.
2. Extract the files. 
3. Run `main.py` using the provided python interpreter in the root folder of the extracted files.

# Usage

```python
import os

from data_io.loaders import load_step
from gcode.gcode_visualizer.virtual_reprap import VirtualRepRap
from gcode.real_3d.generate_curved_layer import generate_curved_layer_christl
from globals import ANGULAR_DEFLECTION

if __name__ == "__main__":
    """
    Generate curved layer Fused Filament Fabrication paths using the algorithm according to Christl
    """
    path_geo = "test_geometry/wave_rounded/wave_round.stp"
    # path to preform G-Code
    path_gcode = "test_geometry/wave_rounded/wave_round_preform_IdeaMaker.gcode"
    # Path to output file that will be created with the final curved layer gcode inside
    path_out_file = "test_geometry/wave_rounded/wave_round_curved.gcode"
    # supply preform geometry if you want to use the local layer index -> This feature does not work yet use None
    path_preform = None

    if os.path.exists(path_out_file):
        os.remove(path_out_file)

    part_shape = load_step(path_geo)
    preform_shape = load_step(path_preform)
    vreprap = VirtualRepRap()
    preform_gcode = vreprap.readin_gcode(path_gcode, 0.2, 0.4)

    curved_layer_gcode = generate_curved_layer_christl(part_shape, preform_gcode, path_out_file,
                                                       preform_shape=preform_shape,
                                                       max_lin_deflection=0.5,
                                                       ang_deflection=ANGULAR_DEFLECTION,
                                                       min_segment_length=0.2,
                                                       max_extrusion_err=0.5,
                                                       lifted_travel_dist=2,
                                                       low_trav_clearance=0.5,
                                                       high_trav_clearance=1,
                                                       max_len_direct_trav=2,
                                                       compute_normals=False
                                                       )
```

# Examples

![img0030](https://user-images.githubusercontent.com/24637325/230966753-27a66a8c-9369-49ab-ace8-eaf8485738eb.png)

![extrusion](https://user-images.githubusercontent.com/24637325/230968475-04670609-b0a2-437b-87cc-b3edd10c1a9c.png)

Extrusion rate and computed normal vector.

# Notes
Author: Wolfgang Christl - Extract of my 2019 Master-Thesis

All code licensed under GNU LESSER GENERAL PUBLIC LICENSE V3
