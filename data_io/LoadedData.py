from abc import ABC, abstractmethod

from OCC.TopoDS import TopoDS_Shape

from data_io.ImportType import ImportType


class LoadedData(ABC):
    """
    Abstract container class that defines important methods to display and transform loaded data inside the viewport.
    All sub-classes contain the data that was loaded and handle transformations etc.
    Subclass names start with LD* (LDShape, LDMesh etc.)
    """

    def __init__(self, new_data, name: str, import_type: ImportType, filepath=None):
        self.data = new_data
        self.name = name
        self.import_type = import_type
        self.ais_object = None  # of type AIS_InteractiveObject
        self.visible = True
        self.filepath = filepath
        super(LoadedData, self).__init__()

    @abstractmethod
    def get_shape(self) -> TopoDS_Shape:
        """
        Returns a valid TopoDS_Shape from LoadedGeometry. If geometry was loaded as mesh it is converted to TopoDS_Shape
        first. Does NOT get the AIS_Shape (The geometry instance that is actually displayed!)

        :return: A TopoDS_Shape representation of the geometry
        """
        raise NotImplementedError

    def __repr__(self):
        return "LoadedData(" + self.name + ")"

    def __str__(self):
        return "LoadedData:" + self.name