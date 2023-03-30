from data_io.ImportType import ImportType
from data_io.LoadedData import LoadedData


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
