"""
KSFD.Grid implements a distributed uniform Cartesian grid of points in
one, two, or three dimensions. It is built on PETSc DMDA elements.
"""
import numpy as np
import petsc4py
from mpi4py import MPI

#
# PETSc Vecs and arrays:
#
# Every PETSc Vec is physically one-dimensional, a long list of
# PETScScalars. For instance, when solving a PDE in 2 dimensions for a
# vector field with 4 dofs on a 64x128 spatial gird, the vector is
# logically three-dimensional with shape (4, 64, 128). vec[i, ix, iy]
# gives the value of field component i at the (x, y) location
# corresponding to (ix, iy). These dimensions are arranged in the
# linear vector in fortran order (order='F') in numpy. This is most
# clearly shown in the documentation of DMGetCoordinates [1].
# DMGetCoordinates returns a Vec with dim components, with the first
# component corresponding to x, the second to y (if dim >= 2), and the
# third to z (if dim >= 3). The index corresponding to the component
# varies most quickly in memory, then the index correspondign to x,
# and that corresponding to y (in two dimensions) most slowly. If, in
# the example above, the coordinate array is reshaped with
# array = coords.reshape(4, 64, 128, order='F') one gets a numpy array
# such that array[0] is a 64x128 array giving the x coordinates and
# array[1] a 64x128 array giving the y coordinates. 
#
# This indexing interacts with the member attributes of the MatStencil
# type, in that one needs to choose i, j, and k correctly. i
# corresponds to the x dimension, as explained most clearly in the man
# page for MatSetValuesStencil [2]. For instance, it is explained:
# "For example, if you create a DMDA with an overlap of one grid level
# and on a particular process its first local nonghost x logical
# coordinate is 6 (so its first ghost x logical coordinate is 5) the
# first i index you can use in your column and row indices in
# MatSetStencil() is 5."
#
# The Grid class defines many members that are shapes, intended for
# use in reshaping Vecs so that they can be nicely subscripted as
# defined above. They should typically be used like this:
#
# array = vec.array.reshape(grid.Vlshape, order='F')
# 
# to get an array that can be accessed by subscripting with dof,
# followed by x, y, z coordinate indexes.
#
# Unfortunately, the petsc4py function getVecArray confuses all this
# by producing an array that, for dof>1, is in neither F nor C
# order. The indexes for coordinates are in 'F' order, i.e. sorted in
# order of increasing stride, but then the dof index (with the
# smallest stride of all) is tacked on as the last index. Because of
# this confusion I eschew getVecArray altogether.
#
# Refs:
# [1] https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/DMGetCoordinates.html
# [2] https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatSetValuesStencil.html
#
class Grid:
    def __init__(
            self,
            dim=1,
            width=1.0,
            height=1.0,
            depth=1.0,
            nx=8,
            ny=8,
            nz=8,
            dof=2,
            order=3,
            stencil_width=None,
            stencil_type=None,
            boundary_type=None,
            comm=MPI.COMM_WORLD,
    ):
        """
        Create a grid. Optional keyword arguments:
        
        dim=1: spatial dimensions. Must be 1, 2, or 3.
        width=1.0: Extent of space in x direction.
        height=1.0: Extent of space in y direction,
        depth=1.0: Extend of space in z direction.
        nx=8: Number of grid points in x direction.
        ny=8: Number of grid points in y direction.
        nz=8: Number of grid points in z direction.
        dof=3: Number of degrees of freedom per grid point.
        order=3: Order of the approximating polynomials (used only to set
            default stencil_width).
        stencil_width=None: Number of ghost points to be held on
            each edge of the grid. Default will be based on
            order.
        stencil_type=DMDA.StencilType.STAR: (See PETSc DMDA
            documentation.)
        boundary_type=DMDA.BoundaryType.PERIODIC: (See PETSc
            documentation.)
        comm=MPI.COMM_WORLD: The MPI communicator.

        Most of these arguments end up as readonly properties of
        the Grid object. Additional properties (readonly unless
        otherwise specified):

        bounds: a float np.array of shape (dim,) listing the width
            in each dimension.
        nps: an int np.array of shape (dim,) listing the global number
            of gridpoints in each dimension.
        spacing: a float np.array of shape (dim,) giving the point
            spacing in each dimension.
        Sdmda: a PETSc.DMDA object (with 1 dof) for scalar fields
            on the Grid.
        Vdmda: a PETSc.DMDA object (with dof dofs) for vector
            fields on the Grid.
        globalSshape: The shape of the global scalar vector. (THis
            is just nps as a tuple).
        globalVshape: The shape of the global Vector field
            vector. (This is (dof,) + globalSshape.)
        globalCshape: The shape of the global coordinate vector. (This
            is (dim,) + globalSshape.)
        Slshape: the shape of the local scalar vector
        Vlshape: the shape of the local vector vector
        Sashape: the shape of the local scalar array (including
            ghost points)
        Vashape: the shape of the local vector array (including
            ghost points)
        Cashape: the shape of the local coordinate array (including
            ghost points)
        coordsNoGhosts: An ndarray of shape globalCshape containing the
            coordinates of the points in a global vector
        coordsWithGhosts: An ndarray of shape Cashape containing the
            coordinates of the points in a local array.
        """
        #
        # To avoid importing PETSc before petsc4py is initialized,
        # don't define these as ordinary default argument values.
        #
        if stencil_type is None:
            stencil_type = petsc4py.PETSc.DMDA.StencilType.STAR
        if boundary_type is None:
            boundary_type = petsc4py.PETSc.DMDA.BoundaryType.PERIODIC
        self._dim = dim
        self._width = width
        self._height = height
        self._depth = depth
        self._bounds = np.array([width, height, depth][:dim], dtype=float)
        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._nps = np.array([nx, ny, nz][:dim], dtype=int)
        self._spacing = self.bounds/self.nps
        self._dof = dof
        self._order = order
        if (stencil_width):
            self._stencil_width = stencil_width
        else:
            self._stencil_width = 1 + order//2
        self._stencil_type = stencil_type
        self._boundary_type = boundary_type
        self._comm = comm
        if dim not in {1, 2, 3}:
            raise ValueError(
                'KSFD.Grid dimension must be 1, 2, or 3'
            )
        self._globalSshape = tuple([self.nx,self.ny,self.nz][:self.dim])
        self._globalVshape = (dof,) + self.globalSshape
        self._globalCshape = (dim,) + self.globalSshape
        self._Sdmda = self.make_dmda(dof=1)   # Scalar space
        self._Vdmda = self.make_dmda(dof=dof) # Vector space
        self._Cdmda = self.make_dmda(dof=dim) # Coordinates
        self._ranges = self.Sdmda.getRanges()
        self._Slshape = tuple([r[1] - r[0] for r in self.ranges])
        self._Vlshape = (dof,) + self.Slshape 
        self._Clshape = (dim,) + self.Slshape
        self._Sashape = tuple(
            np.array(self.Slshape) + 2*self.stencil_width
        )
        self._Cashape = (dim,) + self.Sashape
        self._Vashape = (dof,) + self.Sashape


    @property
    def dim(self):
        return self._dim

    @property
    def width(self):
        return self._width
    
    @property
    def height(self):
        return self._height

    @property
    def depth(self):
        return self._depth
    
    @property
    def bounds(self):
        return self._bounds

    @property
    def nx(self):
        return self._nx

    @property
    def ny(self):
        return self._ny

    @property
    def nz(self):
        return self._nz

    @property
    def nps(self):
        return self._nps

    @property
    def spacing(self):
        return self._spacing

    @property
    def dof(self):
        return self._dof

    @property
    def order(self):
        return self._order

    @property
    def stencil_width(self):
        return self._stencil_width

    @property
    def stencil_type(self):
        return self._stencil_type

    @property
    def boundary_type(self):
        return self._boundary_type

    @property
    def comm(self):
        return self._comm

    @property
    def Sdmda(self):
        return self._Sdmda

    @property
    def globalSshape(self):
        return self._globalSshape

    @property
    def Vdmda(self):
        return self._Vdmda

    @property
    def globalVshape(self):
        return self._globalVshape

    @property
    def Cdmda(self):
        return self._Cdmda

    @property
    def globalCshape(self):
        return self._globalCshape

    @property
    def ranges(self):
        """
        Coordinate ranges owned by this process

        ranges is the result returned by self.Sdmda.getRanges() or
        self.Vdmda.getRanges(). It is a nested tuple of length
        dim. Each element is a two-tuple of form (start, end) giving
        the range of point values owned by this process. For instance,
        ranges = ((0, 9), (5, 9)) would mean that this process owns
        the points numbered from 0 to 8 (i.e., range(0, 9) in the
        x-direction and 5 to 8 in the y direction. The logical shape
        of the local scalar vector would therefore be (9, 4), and the
        local vector vector would be (dof, 9, 4).
        """
        return self._ranges

    @property
    def Slshape(self):
        """
        shape of the local scalar vector
        """
        return self._Slshape
                             

    @property
    def Vlshape(self):
        """
        shape of the local scalar vector
        """
        return self._Vlshape
                             

    @property
    def Clshape(self):
        """
        shape of the local scalar vector
        """
        return self._Clshape

    @property
    def Sashape(self):
        """
        shape of the local scalar vector
        """
        return self._Sashape

    @property
    def Vashape(self):
        """
        shape of the local scalar vector
        """
        return self._Vashape

    @property
    def Cashape(self):
        """
        shape of the local scalar vector
        """
        return self._Cashape

    @property
    def cvec(self):
        """
        The global Vec of coordinates
        """
        if not hasattr(self, '_cvec'):
            coords = self.Sdmda.getCoordinates()
            coords.assemble()
            self._cvec = coords
        return self._cvec

    @property
    def clocal(self):
        """
        Local coordinate vec (with ghosts)
        """
        if not hasattr(self, '_clocal'):
            self._clocal = self.Cdmda.createLocalVec()
            self.Cdmda.globalToLocal(self.cvec, self.clocal)
        return self._clocal

    @property
    def coordsWithGhosts(self):
        """
        Coordinates of local points, including ghost points
        """
        if not hasattr(self, '_coordsWithGhosts'):
            cview = np.copy(
                self.clocal.array.reshape(self.Cashape, order='F'),
                order='F'
            )
            assert cview.flags['F_CONTIGUOUS']
            cview.flags['WRITEABLE'] = False
            self._coordsWithGhosts = cview
        return(self._coordsWithGhosts)

    @property
    def coordsNoGhosts(self):
        """
        Get the coordinates of the points owned by and local to this
        process. Returns coords, an np.ndarry of shape

        1D: (nx, 1)
        2D: (nx, ny, 2)
        3D: (nx, ny, nz, 3)

        such that coords[i], coords[i, j], or coords[i, j, k] is the
        coordinates of the point. This is a readonly view.
        """
        if not hasattr(self, '_coordsNoGhosts'):
            cview = np.copy(
                self.cvec.array.reshape(self.Clshape, order='F'),
                order='F'
            )
            assert cview.flags['F_CONTIGUOUS']
            cview.flags['WRITEABLE'] = False
            self._coordsNoGhosts = cview
        return self._coordsNoGhosts

    def make_dmda(self, dof=1):
        """
        Make a new DMDA, using self attributes for inputs
        """
        dmda = petsc4py.PETSc.DMDA().create(
            dim=self.dim,
            sizes=self.globalSshape,
            boundary_type=self.boundary_type,
            dof=dof,
            stencil_type=self.stencil_type,
            stencil_width=self.stencil_width,
            comm=self.comm
        )
        dmda.setUniformCoordinates(
            xmin=0.0,
            xmax=self.width,
            ymin=0.0,
            ymax=self.height,
            zmin=0.0,
            zmax=self.depth
        )
        dmda.setFromOptions()
        dmda.setUp()
        return dmda

    def stencil_slice(self, stencil, array, G=None, requireF=True):
        if not isinstance(array, np.ndarray):
            array = array.array
        assert (not requireF) or array.flags['F_CONTIGUOUS']
        assert isinstance(array, np.ndarray) and (
            array.shape == self.Sashape or
            array.shape == self.Vashape
        )
        assert G is None or (
            isinstance(G, np.ndarray) and G.shape == self.Sashape
        )
        sw = self.stencil_width
        slices = [slice(0, None)] * self.dim
        for i in range(self.dim):
            slices[i] = slice(
                stencil[i] + sw,
                stencil[i] + sw + self.Slshape[i]
            )
        outarray = array if stencil[-1] != -1 else G
        if outarray.ndim > self.dim:
            slices[:0] = [stencil[-1]]
        return outarray[tuple(slices)]

    def cleanup(self):
        """Destroy any PETSc objects we created"""
        for a in ['_clocal', '_Sdmda', '_Vdmda', '_Cdmda']:
            obj = getattr(self, a, None)
            if obj:
                obj.destroy()

    def __del__(self):
        self.cleanup()

    #
    # Pickling: The assumption behind these pickling functions is that
    # the grid is a static structure: it is not changed after
    # creation. The comm in not pickable, so a kludge is used that
    # will work for MPI.COMM_SELF and MPI.COMM_WORLD. Otherwise the
    # default is used. If you really must set the comm to something
    # else after unpickling, assign to grid._comm.
    #
    def __getstate__(self):
        if self.comm is MPI.COMM_SELF:
            commstr = 'COMM_SELF'
        else:
            commstr = 'COMM_WORLD'
        state = dict(
            dim=self.dim,
            width=self.width,
            height=self.height,
            depth=self.depth,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            dof=self.dof,
            order=self.order,
            stencil_width=self.stencil_width,
            stencil_type=self.stencil_type,
            boundary_type=self.boundary_type,
            commstr=commstr
        )
        return(state)

    def __setstate__(self, state):
        try:
            state['comm'] = getattr(MPI, state['commstr'])
            del state['commstr']
        except (KeyError, AttributeError):
            state['comm'] = None
        self.__init__(**state)
        
