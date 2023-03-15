from OCC.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
from OCC.IGESControl import IGESControl_Reader
from OCC.SMESH import SMESH_Gen, SMESH_Mesh
from OCC.STEPControl import STEPControl_Reader
from OCC.StlAPI import StlAPI_Reader
from OCC.TopoDS import TopoDS_Shape
from CColors import CColors


def load_step(file_path) -> TopoDS_Shape or None:
    """
    Loads a STP file using OCCT

    :return: TopoDS_Shape if loading successfully or None if error during loading
    """
    print("Loading *.stp file...", end="")
    if file_path is None:
        return None
    _shape_loaded = None
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(file_path)
    if status == IFSelect_RetDone:  # check status
        failsonly = False
        step_reader.PrintCheckLoad(failsonly, IFSelect_ItemsByEntity)
        step_reader.PrintCheckTransfer(failsonly, IFSelect_ItemsByEntity)
        ok = step_reader.TransferRoot(1)
        _nbs = step_reader.NbShapes()
        _shape_loaded = step_reader.Shape(1)
        print("Done!")
        return _shape_loaded
    else:
        print(f"{CColors.FAIL}Error: Can't read *.step file.{CColors.ENDC}")
        return TopoDS_Shape()


def load_stl(file_path) -> TopoDS_Shape:
    """
    Converts every triangle into a TopoDS_Face. This takes very long. Do not load files >5MB. You will get old.

    :return: TopoDS_Shape of loaded STL file
    """
    stl_reader = StlAPI_Reader()
    my_shape = TopoDS_Shape()
    print("Starting to load *.stl file. If file is too large this might take forever :( ...", end="")
    stl_reader.Read(my_shape, file_path)
    print("Done!")
    return my_shape


def load_mesh(file_path) -> SMESH_Mesh:
    """
    Load an STL file as mesh using SMESH. Way faster than with OCE (and its conversion to TopoDS_Shape).
    You can not use OCE API methods on SMESH objects

    :return: SMESH_Gen of loaded STL file
    """
    aMeshGen = SMESH_Gen()
    a_mesh = aMeshGen.CreateMesh(0, True)
    print("Starting to load *.stl file. If file is too large this might take forever :( ...", end="")
    a_mesh.STLToMesh(file_path)
    print("Done!")
    return a_mesh


def load_igs(file_path) -> TopoDS_Shape:
    """
    Loads a IGES file using OCCT

    :param file_path: Path to the *.IGES
    :return: TopoDS_Shape representing the loaded data
    """
    iges_reader = IGESControl_Reader()
    status = iges_reader.ReadFile(file_path)
    if status == IFSelect_RetDone:  # check status
        failsonly = False
        iges_reader.PrintCheckLoad(failsonly, IFSelect_ItemsByEntity)
        iges_reader.PrintCheckTransfer(failsonly, IFSelect_ItemsByEntity)
        ok = iges_reader.TransferRoots()
        print("Finished loading *.igs file!")
        return iges_reader.Shape(1)
    else:
        print(f"{CColors.FAIL}Error: can't read *.igs file.{CColors.ENDC}")
        return TopoDS_Shape()
