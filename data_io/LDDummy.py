from OCC.TopoDS import TopoDS_Shape

from data_io.LoadedData import LoadedData


class LDDummy(LoadedData):

    def get_shape(self) -> TopoDS_Shape:
        pass