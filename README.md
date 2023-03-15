# <img src="real3dfff_logo.png" width="100" height="100" /> Real3DFFF
![Real3DFFF UI](wiki_content/real3dfff_ui.JPG)

# Features

*  Generate curved tool paths for FFF-Printers
*  Generate Preform geometries
*  Extract geometries for support only generation (not supported by Slic3r anymore)
*  Import/Export of G-Code, STL, STEP, IGES and ATOS point clouds
*  Generation of "print onto" geometries
*  Alignment of 3D scanned parts to their physical counterpart inside the print volume
*  Stereo camera support to measure exact location of ATOS tracking points in 3D space
*  Collision detection of nozzle and printed part
*  Translation/Rotation of geometry inside virtual print bed


**This applications supports Slic3r 1.3 - other slicing software is not supported!**

# Install
**[Instructions can be found inside the Wiki ](https://gitlab.lrz.de/wolfgangchristl/Real3DFDM/wikis/home)**

## Notes to some parts/classes of the software
### Getting the Z-Coordinates
 ```python
 zgetter = ZGetter('test_geometry/sem_test_6_flat_round.stp')
 while(True){
 	  z_upper, z_lower = zgetter.get_z(x, y)
 }
 ```
 
### Notes to the preform algorithm v2 (recommended)
1. Load → load file (STEP files are very much recommended
2. Right click on the imported element inside the tree → **Real3DFFF** -> **create preform v2**
3. Enter an angle so that only the **bottom** of the geometry is selected (check by clicking preview). Angles between 88°-92° should work well
    
    *Incorrect angle value*
    
    ![incorrect angle](wiki_content/geo_bottom_incorrect.png) 
    
    *Correct angle value*

    ![Correct angle](wiki_content/geo_bottom_correct.png)
4. Enter an angle so that only the **top** of the geometry is selected (check by clicking preview). Angles between 88°-92° should work well

    *Correct angle value*

    ![Correct angle](wiki_content/geo_top_correct.png)

5. wait!
6. Export → export preform to stl

#### Using first preform algorithm (deprecated, do not use!)
1. Import → import file
2. Tools → create preform
3. wait... (Bugs can happen with rounded edges on geometry)
4. Export → export preform to stl
