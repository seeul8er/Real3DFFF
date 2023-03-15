from OCC.GeomLProp import GeomLProp_SLProps
from OCC.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCC.TopAbs import TopAbs_IN, TopAbs_FORWARD, TopAbs_REVERSED
from OCC.TopoDS import TopoDS_Shape, TopoDS_Face
from OCC.gp import gp_Dir, gp_Pnt, gp_Lin

from CColors import CColors
from OCCUtils import get_boundingbox
from OCCUtils.face import Face
from data_io.loaders import load_stl, load_step


class ZGetter:
    """
    Up to 90% faster than original ZGetter implementation using BRepIntCurveSurface_Inter()!
    Designed as a module that can exist outside of Real3DFFF application

    Alternative intersection algorithms that might give more exact results:

     - GeomAPI_IntCS (parametric geometry level)
     - BRepIntCurveSurface_Inter (topology level)
    """

    def __init__(self, _file_path: str or None, a_loaded_shape: None or TopoDS_Shape, _tol=1e-5):
        """
        A class to get the z-coordinates on the surface of a part. Supported file formats are *.stp, *.step and *.stl

        :param _file_path: The absolute or relative path to the file containing the geometry. Can be None if
            a_loaded_shape is provided
        :param a_loaded_shape: Use this TopoDS_Shape instead of loading a new one. If not set a file_path is required
        """
        self._file_path = _file_path
        if a_loaded_shape is None:
            _new_shape = self._load_data(self._determine_filetype(_file_path))

        _, _, self.zmin, _, _, self.zmax = get_boundingbox(a_loaded_shape)
        self.shp_inter = IntCurvesFace_ShapeIntersector()
        self.shp_inter.Load(a_loaded_shape, _tol)

        self._gp_Pnt_loc = gp_Pnt(0, 0, 0)
        self._gp_Lin_intersection = gp_Lin(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))

    @staticmethod
    def _determine_filetype(file_path):
        if file_path.endswith("stl"):
            return 1
        elif file_path.endswith("stp") or file_path.endswith("step"):
            return 2
        else:
            print("ERROR - Unknown file type. Supported types are *.stp, *.step and *.stl")
            return 0  # unknown file type

    def _load_data(self, _filetype):
        if _filetype == 1:
            return load_stl(self._file_path)
        elif _filetype == 2:
            return load_step(self._file_path)

    def get_z(self, x, y) -> (float, float):
        """
        Returns the lower and upper z-coordinate on the surface of the geometry at the given x and y coordinates.
        The coordinate system of the specified STEP or STL file is used.
        If only one z-coordinate is determined it is returned as highest and lowest.
        If more than two z-coordinates are determined, the highest and lowest coordinate is returned.

        :param x: X-position
        :param y: Y-position
        :return: (z-upper, z-lower) or (None, None) if no z-coordinates were found
        """
        self._gp_Pnt_loc.SetCoord(x, y, 0)  # update location
        self._gp_Lin_intersection.SetLocation(self._gp_Pnt_loc)
        self.shp_inter.Perform(self._gp_Lin_intersection, self.zmin - 10, self.zmax + 10)
        self.shp_inter.SortResult()
        num_points = self.shp_inter.NbPnt()
        if num_points == 2:
            return self.shp_inter.WParameter(2), self.shp_inter.WParameter(1)
        elif num_points == 3:  # some hard coded special cases. Not optimal. Requires no overhangs
            _in_pnt = None
            for i in range(1, num_points + 1):
                if self.shp_inter.State(i) is TopAbs_IN:
                    _in_pnt = self.shp_inter.WParameter(i)
            if _in_pnt is None:
                return self.shp_inter.WParameter(num_points), self.shp_inter.WParameter(1)
            if _in_pnt > self.shp_inter.WParameter(2):
                return _in_pnt, self.shp_inter.WParameter(2)
            else:
                return self.shp_inter.WParameter(2), _in_pnt
        elif num_points == 4:
            return self.shp_inter.WParameter(3), self.shp_inter.WParameter(2)
        elif num_points > 4:
            # for i in range(1, num_points+1):
            #     print(f"Z: {self.shp_inter.WParameter(i)} {self.shp_inter.State(i)}")
            # print("")
            print(f"{CColors.WARNING}WARNING - More than four z-coordinates found! [{x}, {y}]{CColors.ENDC}")
            return self.shp_inter.WParameter(num_points), self.shp_inter.WParameter(1)
        elif num_points == 1:
            print(f"{CColors.WARNING}WARNING - Only one z-coordinate found! [{x}, {y}]{CColors.ENDC}")
            return self.shp_inter.WParameter(1), self.shp_inter.WParameter(1)
        else:
            return None, None

    def get_z_normals(self, x, y) -> (float, float, gp_Dir, gp_Dir):
        """
        Returns the lower and upper z-coordinate on the surface of the geometry at the given x and y coordinates.
        Returns the surface normal at upper and lower z-coordinate. Normal is always pointing up in Z-direction.

        The coordinate system of the specified STEP or STL file is used.
        If only one z-coordinate is determined it is returned as highest and lowest.
        If more than two z-coordinates are determined, the highest and lowest coordinate is returned.

        :param x: X-position
        :param y: Y-position
        :return: (z-upper, z-lower, N-upper, N-lower) or (None, None, None, None) if no z-coordinates were found
        """
        self._gp_Pnt_loc.SetCoord(x, y, 0)  # update location
        self._gp_Lin_intersection.SetLocation(self._gp_Pnt_loc)
        self.shp_inter.Perform(self._gp_Lin_intersection, self.zmin - 10, self.zmax + 10)
        self.shp_inter.SortResult()
        num_points = self.shp_inter.NbPnt()
        if num_points == 2:
            return self.shp_inter.WParameter(2), self.shp_inter.WParameter(1), \
                   self._compute_normal(2), self._compute_normal(1)
        elif num_points == 3:  # some hard coded special cases. Not optimal. Requires no overhangs
            _in_pnt = None
            _index = None
            for i in range(1, num_points + 1):
                if self.shp_inter.State(i) is TopAbs_IN:
                    _in_pnt = self.shp_inter.WParameter(i)
                    _index = i
            if _in_pnt is None:
                return self.shp_inter.WParameter(num_points), self.shp_inter.WParameter(1),\
                       self._compute_normal(num_points), self._compute_normal(1)
            if _in_pnt > self.shp_inter.WParameter(2):
                return _in_pnt, self.shp_inter.WParameter(2), self._compute_normal(_index), self._compute_normal(2)
            else:
                return self.shp_inter.WParameter(2), _in_pnt, self._compute_normal(2), self._compute_normal(_index)
        elif num_points == 4:
            return self.shp_inter.WParameter(3), self.shp_inter.WParameter(2), \
                   self._compute_normal(3), self._compute_normal(2)
        elif num_points > 4:
            # for i in range(1, num_points+1):
            #     print(f"Z: {self.shp_inter.WParameter(i)} {self.shp_inter.State(i)}")
            # print("")
            print(f"{CColors.WARNING}WARNING - More than four z-coordinates found! [{x}, {y}]{CColors.ENDC}")
            return self.shp_inter.WParameter(num_points), self.shp_inter.WParameter(1), \
                   self._compute_normal(num_points), self._compute_normal(1)
        elif num_points == 1:
            print(f"{CColors.WARNING}WARNING - Only one z-coordinate found! [{x}, {y}]{CColors.ENDC}")
            return self.shp_inter.WParameter(1), self.shp_inter.WParameter(1), \
                   self._compute_normal(1), self._compute_normal(1)
        else:
            return None, None, None, None

    def count_layers(self, x, y) -> (int, float or None):
        """
        Count the layers inside the loaded geometry and get z-value of the lowest layer

        :param x:
        :param y:
        :return: number of layers, z-coordinate of the lowest layer
        """
        self._gp_Pnt_loc.SetCoord(x, y, 0)  # update location
        self._gp_Lin_intersection.SetLocation(self._gp_Pnt_loc)
        self.shp_inter.Perform(self._gp_Lin_intersection, self.zmin - 10, self.zmax + 10)
        self.shp_inter.SortResult()
        num_layers = self.shp_inter.NbPnt()
        count = 0
        first_z = None
        for i in range(1, num_layers + 1):
            if self.shp_inter.State(i) is TopAbs_IN:  # Only count/consider points inside the face, not on its edge
                count += 1
                if first_z is None:
                    first_z = self.shp_inter.WParameter(i)
        if num_layers > 0:
            return count, first_z
        else:
            return count, None

    def _compute_normal(self, _index: int) -> gp_Dir:
        """
        Computes the downwards surface normal of the face.
        For internal use in this class. No parameters. Use of class variables

        :type _index: Index of the Face inside self.shp_inter intersection list
        :return: Surface normal facing into the face
        """
        _f = Face(self.shp_inter.Face(_index))
        # uv_pnt = _f.point_to_parameter(self.shp_inter.Pnt(_index))
        # gp_dir_normal2 = _f.DiffGeom.normal(uv_pnt[0], uv_pnt[1])
        gp_dir_normal = _f.DiffGeom.normal(self.shp_inter.UParameter(_index), self.shp_inter.VParameter(_index))
        if gp_dir_normal.Z() > 0:  # Always point downwards
            return gp_dir_normal.Reversed()
        else:
            return gp_dir_normal

    @staticmethod
    def compute_normal(_the_face: TopoDS_Face, _u: float, _v: float) -> gp_Dir:
        """
        Can be called from anywhere providing parameters. Computes the downwards surface normal of the face

        :param _the_face: The face
        :param _u: U-Parameter on the face
        :param _v: V-Parameter on the face
        :return: Surface normal facing outside
        """
        _f = Face(_the_face)
        gp_dir_normal = _f.DiffGeom.normal(_u, _v)
        if gp_dir_normal.Z() > 0:  # Always point downwards
            return gp_dir_normal.Reversed()
        else:
            return gp_dir_normal
