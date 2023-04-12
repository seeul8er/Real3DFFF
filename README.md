# <img src="real3dfff_logo.png" width="100" height="100" /> Real3DFFF

Traditional vs Curved-Layer Fused Filament Fabrication (CLFFF) using Real3DFFF

https://user-images.githubusercontent.com/24637325/230965579-060c6ff2-75d5-4293-90fa-5e9db6e54b03.mp4

# Features

*  Generate curved tool paths for FFF-Printers including normal vectors for 5-axis printing
*  Generate Preform geometries
*  Extract geometries for support only generation
*  Import/Export of G-Code, STL, STEP, IGES

# How does it work

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

## Adaptive Curved Layer Fused Filament Fabrication Process
The algorithm for generating curved paths is described below. The first step is to create a preform from the original geometry. As already described, this can be done directly in CAD and on the basis of the component geometry to be printed. Both the preform and the component geometry to be printed should have the same coordinate system and be congruent in the XY plane. This pre-positioning ensures correct execution of the subsequent steps.  
For the preform geometry, the G-code must now be generated using a slicer. Care must be taken that the slicer does not realign the preform geometry on the print bed. In principle, any slicer is suitable as long as it is able to output the layer index and the Z-coordinate of the layer at the beginning of each layer. A defined notation must be adhered to so that subsequent steps can extract this information. The Real3DFFF application supports the notation formats of the slicers "Slic3r", "Slic3r PE", “SuperSlicer” and IdeaMaker. For the former three, the notation can be freely adapted via script.  

### Triangulation of the print geometry
In the next step, the geometry to be printed is triangulated i.e., its surface is approximated with triangles. This step is not necessary for parts in the STL or similar formats, as their surface already consists of triangles. The CAD formats STEP and IGES can be converted to STL via the OCCT API. The maximum allowed deviation of the resulting geometry from the component geometry is defined by the parameters "linear deflection" and "angular deflection".

### Projection of the triangulation into the X/Y plane
![grafik](https://user-images.githubusercontent.com/24637325/231286843-59625d16-b346-4f43-b316-5fedb7ca2262.png)

All triangular surfaces with a surface normal parallel to the Z-axis are now removed from the resulting triangular grid. The same applies to surfaces whose normal has a Z-component of zero. In these areas the curvature of the component surface is constant or irrelevant. As a result we receive all surfaces of the approximated component geometry where the component has a curvature. All edges of these surfaces are projected into the XY-plane. In the following, these edges are called the "adjustment layer".

### Preform G-code segmentation
![grafik](https://user-images.githubusercontent.com/24637325/231286930-cdc024f9-2234-4b85-a0a4-a3b8fcd228ae.png)

Now the G-code of the preform can be adapted to the final geometry. In the next step, the G-code of the preform is loaded. The individual layers are identified with their layer index and the corresponding Z-coordinate. The slicer has stored this information in the G-code via the defined notation. The following step only considers the G-code segments in which an extrusion actually takes place. Travel movements are ignored for now. Each G-Code layer is projected into the XY-plane.  
Now all extrusions of a G-Code layer are intersected with the edges of the adjustment layer that were previously projected into the XY-plane. This step is conducted for each layer of the Preform G-code. The resulting intersections give the points in the Preform G-code where a G-Code move (extrusion segment) must be split and the Z-coordinates adjusted to fit the final geometry. Each start and end point of a movement from the preform G-code must also be adjusted in the Z-direction. This ensures that the final G-code exactly fills the parts volume that is defined by triangulation. OCCT helps to calculate the intersection points between the G-code layer and the adjustment layer.

![part_to_clfff_gcode_2d](https://user-images.githubusercontent.com/24637325/231271511-3bdea28e-3cfc-4a35-bfa9-0e63d22a08e0.jpg)

### Removal of short G-code segments
![grafik](https://user-images.githubusercontent.com/24637325/231287256-3a42ce15-52e8-4187-b714-33e5418d8e0b.png)

The segmentation of Preform G-code layers can result in very short extrusion segments. The minimum allowed segment length can be defined by a parameter in the Real3DFFF application. Segments that do not meet the criterion are deleted and the resulting gap is filled by adjusting the start point of the following segment. This step is performed in the 2D and in the XY-plane.

### Adjustment of the Z-coordinate of each segment
![grafik](https://user-images.githubusercontent.com/24637325/231278412-b8a50052-d5f2-432f-85cf-e4a34f75615c.png)

[7-1]

$$h_{layer}=\frac{z_{upper}-z_{lower}}{Layer Count Preform}$$
$$z_{segment}=z_{lower}+h_{layer}i_{layer}$$

In this step, the segmented Preform G-code and the respective extrusion rate of each segment is adjusted. Each end point of a segment is now moved using the ZGetter class. It is not necessary to move the start points, as only the end point of a movement is specified in the G-code format.  
The new Z-coordinate of each segment end point at (X, Y) is calculated according to [7-1] from the number of layers in the preform G-code, the layer height, the current layer index and the Z-coordinates of the component surface at point (X, Y) returned by the ZGetter implementation. The number of layers in the preform is constant, as only preform geometries with constant cross-section in the Z-direction are considered.

### Recalculation of the extrusion rate per G-Code segment
Since the cross-section of the part can change but the number of layers remains constant through the cross-section, the extrusion rate must be scaled according to the increasing/decreasing local layer thickness (adaptive layer thickness).  

[7-2]

$$E_{Segment,new}=E_{Segment,Preform}\frac{h_{layer}}{h_{layer,Preform}}$$

Where $E_{Segment,new}$ is the new extrusion rate on the segment. It results from the extrusion rate of the segment in the preform G-code $E_{Segment,Preform}$, its layer thickness $h_{layer,Preform}$ and the newly calculated layer thickness $h_{layer}$.
By shifting the end point of the segment in Z-direction, an elongation of the segment takes place. To compensate for this, the extrusion rate is multiplied by another factor.

[7-3]

$$E_{Segment,new,\Delta l}=E_{Segment,new}\frac{l_{3D}}{l_{2D}}$$

Here $l_{3D}$ is the length of the segment adjusted in the Z-direction and $l_{2D}$ its projection in the XY-plane, i.e. the length of the segment in the preform G-code.

### Approximation of the normal vector

![grafik](https://user-images.githubusercontent.com/24637325/231286350-f0a96018-a301-42e2-9de9-d57a4d58e1d7.png)

As already described, ZGetter also provides the normal vector at the top and bottom ($\vec{n_{upper}}$ and $\vec{n_{lower}}$) of the component geometry at the location (X, Y).
By means of linear interpolation, "intermediate vectors" can be calculated for the individual levels.

[7-4]

$$w=\frac{\vec{n_{Segment}}-\vec{n_{lower}}}{\vec{n_{upper}}-\vec{n_{lower}}}$$

$$\vec{n_{Segment}}=w\vec{n_{upper}}+(1-w)\vec{n_{lower}}$$

The interpolated normal vector $\vec{n_{Segment}}$ is the normal vector for the currently considered segment [7-5]. The parameter $\vec{z_{Segment}}$ specifies the Z-coordinate of the layer for which the normal vector is to be calculated. The normal vector is outputted in the final G-code. To ensure compatibility with 3-axis FFF printers and common slicer software, the normal vector is written to the G-code file as follows:  
In addition to the X, Y, Z and E coordinates of an extrusion instruction, the parameters N, O and R are added to the instruction. The three coordinates of the normal vector are stored as additional parameters. The parameter N stores the X-component, O the Y-component and R the Z-value of the normal. The previously defined printer coordinate system is used.

### Further G-code adjustments
The following are part of the adaptive CLFFF algorithm to ensure printability and quality of the final G-Code. 

#### Adjustment of the travel movements
In addition to the extrusions, the travel movements must also be adjusted. Otherwise, there will be collisions between the print head and the printed geometry. Depending on the length of the travel movement, the movement is replaced by three individual movements. Raising the nozzle by a defined value, moving the print head in the XY plane to the target point and lowering it to the desired Z coordinate. To optimise the printing time, the process distinguishes between three types. Long, short and direct travel movements. The criterion in each case is the length of the travel movement in 2D. It can be set via parameters in Real3DFFF.  
With long travel movements, the print head is raised by a defined value above the highest point of the already printed part.  
With short travel movements, the printhead is raised by a defined value before it moves in the XY-plane to the target coordinate and is lowered again. With such movements, it is assumed that the travel distance is too short for a collision to occur.  
In the case of direct travel movements, the printhead is not raised, but travels directly in a straight line to the target point. 

#### Segmentation for linear cross-section changes

![grafik](https://user-images.githubusercontent.com/24637325/231286210-e29d017c-825d-4f80-a70f-b1147f28593a.png)
![grafik](https://user-images.githubusercontent.com/24637325/231287371-9fb53c1d-2f00-4a30-aaae-c0c713be00c4.png)

The method described so far for segmenting the preform G-code uses the triangulation of the part surface. In areas with linear cross-section changes of the part, there are only very few edges available from the tessellation. As a result, an extrusion is not split up very often and only a few G-code segments are created in such an area. This is problematic since the extrusion rate on a G-Code segment can only be constant. Due to the linear increase of the final parts cross section, a constant change in layer height of the final G-Code is required. The required increase in layer height and thereby also in extrusion rate can only be achieved by splitting the G-Code segment into multiple segments to approximate that linear increase.  
The developed algorithm therefore subdivides a G-Code segment of the preform into $n_{seg}$ segments, so that the extrusion rate on each sub-segment can be adapted to the cross-section change of the part.

[7-5]

$$\vec{n_{seg}} = \lfloor{ \sqrt{\frac{l_{2D}}{2E_S}\vert{h_e-h_s}\vert}+1}\rfloor, \vec{n_{seg}}\in N+$$

Where $h_s$ and $h_e$ are the required layer heights at the start and end of a linear increasing cross section. The parameter $E_S$ is user defined and describes the mean extrusion error per created sub-segment over the entire segment. The formular [7-5] is adapted from the error function of the Riemann Integral for monotonically increasing/decreasing functions. The segment of the previous layer directly below the currently considered segment $f(x)$ is described as a straight line $g(x)$.

#### Support Structures

Depending on the geometry to be printed, support structures are necessary to support overhangs (~>45°). Since components with undercuts have been excluded for this process, only support structures that lie directly on the build platform need to be considered. To generate the support structures, the underside of the component geometry (surfaces with angles to the Z-axis <90°) is identified. These are extracted and passed to a conventional slicer. The slicer generates the G-code for the support structures below the extracted surface. No G-code is generated for the surface itself when using supported slicers, as the surface does not represent a closed volume. Depending on the slicer, additional settings may have to be made so that such non-manifold geometries (extracted surfaces of the component underside) are not automatically repaired or ignored. Also, in this step it is important that the slicer does not reposition the part on the build plate, so that later the support structure is directly under the part to be printed. The G-code for the support structure printed first. Then the CLFFF G-code can be executed.

## Implmentation

![grafik](https://user-images.githubusercontent.com/24637325/231286004-9a9de0e7-a719-4a57-9db6-de3135acd2e9.png)


# Install
Real3DFFF requires a big set of dependencies, some of them being a bit dated.
To make things easier, a complete python environment is provided in addition to the code to get you started in no time.

1. Download the package from the GitHub release section.
2. Extract the files. 
3. Run `main.py` using the provided python interpreter in the root folder of the extracted files.

# Usage

1. Create a Preform for the to-be-printed geometry (an extruded shadow of the part)
2. Slice the Preform geometry with a supported slicer (Slic3r or variants, IdeaMaker) adding the necessary annootations (see below)
3. Run the below script within the provided Python environment.

Annotations necessary for Slic3r based slicers like PrusaSlicer, SuperSlicer etc.:  
Add a custom gcode at layer change: 
```
; layer_num=[layer_num]
; layer_z=[layer_z]
```

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
    # Path to file representing the final shape
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
