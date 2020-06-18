from mpi4py import MPI
import numpy as np
from numpy.random import SeedSequence, default_rng
from scipy.spatial import KDTree
try:
    from .ksfddebug import log
except ImportError:
    from ksfddebug import log

def logRANDOM(*args, **kwargs):
    log(*args, system='RANDOM', **kwargs)

class Generator:
    """KSFDGenerator -- MPI-aware random number generator"""
    _rng = None
    _seeds = None

    def __init__(self, seed=None, comm=MPI.COMM_WORLD):
        """Create independent random number generators in parallel

        Optional keyword arguments:
        seed=None: seed the Gnerator to get a reproducible stream.
        comm=MPI.COMM_WORLD: The MPI communicator

        Creates an independent np.random.Generator in each MPI process. This
        generator can be retrived with the __call__ method, e.g.

        from KSFD import Generator
        ...
        kgen = Generator(seed)
        rng = kgen()

        Also, the class method get_rng() will retrieve the process-wide
        npp.random.Generator, so that you don't need to carry the Generator
        instance around with you:

        rng = Generator.get_rng()
        """
        if seed is None and self._rng is not None:
            #
            # already set -- nothing to do
            #
            return
        size = comm.size
        rank = comm.rank
        ss = SeedSequence(seed)
        seeds = ss.spawn(size)
        type(self)._seeds = seeds
        type(self)._rng = default_rng(seeds[rank])
        return

    def __call__(self):
        """Return the np.random.Generator"""
        return self.get_rng()

    @classmethod
    def get_rng(cls):
        if cls._rng is None:
            gen = cls()
        return cls._rng

def extended_coords(grid):
    """
    Get coordinates of grid points, including ghost points

    This function returns an array of coorindates. The central region
    of the array (i.e., all the points owned by this process) is
    identical to the central region of grid.coordsWithGhosts. However,
    the coordinates of the fringe of grid points are constructed by
    extending the grid past its normal boundaries by adding or
    substracting the grid spacing from the edge points. The result
    will be coordinates not within the space represented by the grid:
    negative at the left and bottom edges and >= the width
    or height at the right or top edges.
    """
    sw = grid.stencil_width
    dim = grid.dim
    ecoords = grid.coordsWithGhosts.copy()
    ecoords = ecoords.reshape(grid.Cashape, order='F')
    logRANDOM('dim', dim)
    logRANDOM('grid.Slshape', grid.Slshape)
    for d in range(dim):
        np = grid.Slshape[d]
        logRANDOM('d, np', d, np)
        space = grid.spacing[d]
        indexesin = (d,) + tuple(
            [slice(0, None)]*d + [sw] + [slice(0, None)]*(dim-d-1)
        )
        logRANDOM('indexesin', indexesin)
        for i, delta in enumerate(range(-sw, 0)):
            indexesout = (d,) + tuple(
                [slice(0, None)]*d + [i] + [slice(0,None)]*(grid.dim-d-1)
            )
            logRANDOM('indexesout', indexesout)
            ecoords[indexesout] = ecoords[indexesin] + delta * space
        indexesin = (d,) + tuple(
            [slice(0, None)]*d + [np+sw-1] + [slice(0, None)]*(dim-d-1)
        )
        logRANDOM('indexesin', indexesin)
        for i, delta in enumerate(range(1, sw+1)):
            indexesout = (d,) + tuple(
                [slice(0, None)]*d + [np+sw+i] + [slice(0,None)]*(grid.dim-d-1)
            )
            logRANDOM('indexesout', indexesout)
            ecoords[indexesout] = ecoords[indexesin] + delta * space
    return ecoords

def random_function(
    grid,
    randgrid=None,
    vals=None,
    mu=0.0,
    sigma=0.01,
    tol=1e-10,
    periodic=True,
    f=(lambda x: 2*x**3 - 3*x**2 + 1),
    seed=None
):
    """define a pseudorandom Function

    random_function defines a scalar-valued random function on a KSFD
    Grid.

    Required positional argument:
    grid: the grid on which the function is to be defined

    Keyword arguments:

    randgrid=None: The grid on which random values are to be
        drawn. Interpolation (using argument f is used to fill in
        other points of grid. grid is used if not randgrid is
        passed. Ideally, the vertexes of randgrid should be a subset of
        the points of grid. random_function may work if not, but this
        is not guaranteed. randgrid should be created with
        stencil_type = PETSc.DMDA.StencilType.BOX.  (This is not the
        default.)
    vals=None: A PETSc Vec of random values for the vertexes of
        randgrid. This Vec will typically have the structure of a Vec
        returned by randgrid.Sdmda.createGlobalVec(). If not provided,
        random values are drawn from a normal distribution with mean
        mu and standard deviation sigma.
    mu=0.0: See vals.
    sigma=0.01: See vals.
    tol=1e-10: A relative tolerance used in determining if two points
        specified by coordinates are the same. 
    f=(lambda x: 2*x**3 - 3*x**2 + 1): the function used to fill in the
        space between the vertexes.
    seed=None: This object is passed to numpy.random.seed to seed the
        random number generator.

    Return value:
    
    vec: a global PETSc.Vec containing the values of the random
        function. This vector is obtained with
        grid.Sdmda.createGlobalVec().
    """
    comm = grid.comm
    if not randgrid:
        randgrid = grid
    gdim = grid.dim
    if gdim != randgrid.dim:
        raise ValueError(
            'randgrid and grid must have the same dimension'
        )
    valsSupplied = bool(vals)
    if not valsSupplied:
        vals = randgrid.Sdmda.createGlobalVec()
        kgen = Generator(seed=seed, comm=comm)
        vals.array = kgen().normal(loc=mu, scale=sigma, size=vals.array.shape)
        vals.assemble()
    if (                        # shortcut for when grids match
            np.all(randgrid.nps == grid.nps) and
            np.all(randgrid.spacing == grid.spacing)
    ):
        vec = grid.Sdmda.createGlobalVec()
        vec.array = vals.array
        vec.assemble()
        return vec
    lvals = randgrid.Sdmda.createLocalVec()
    randgrid.Sdmda.globalToLocal(vals, lvals)
    larr = lvals.array
    dcoords = grid.coordsNoGhosts.reshape(gdim, -1).transpose()
    logRANDOM('dcoords', dcoords)
    logRANDOM('dcoords.shape', dcoords.shape)
    tree = KDTree(dcoords)
    dvec = np.zeros(len(dcoords), dtype=float)
    if not valsSupplied and sigma == 0.0: # We don't need the computation
        dvec[:] = mu
    else:
    # coordinates of randgrid vertexes (including ghost points)
        vcoords = extended_coords(randgrid).reshape(gdim, -1).transpose()
        logRANDOM('vcoords', vcoords)
        logRANDOM('vcoords.shape', vcoords.shape)
        hs = randgrid.spacing
        hmax = np.max(hs)
        vf = lambda x: np.product(f(x))
        # for every point in randgrid
        for v, vc in enumerate(vcoords):
            # numbers of points of grid w/in hmax of point v in randgrid
            touched = np.array(tree.query_ball_point(vc, hmax, float('inf')))
            if len(touched) > 0:                    # there may be none
                # difference between touched points and vc in randgrid units
                x = np.abs(dcoords[touched] - vc) / hs
                # touched2 flags those w/i 1 randgrid unit of vc in
                # each dimension
                touched2 = np.where(np.amax(x, 1) < 1 - tol)[0]
                # Now select those from touched
                touched = touched[touched2]
                # if (127 in touched):
                #     logRANDOM('v, vc touch 127', v, vc, dcoords[127])
                # apply vf to x and add that to the touched points
                dvec[touched] += (
                    larr[v] * np.fromiter(map(vf, x[touched2]), float)
                )
    logRANDOM('dvec', dvec)
    vec = grid.Sdmda.createGlobalVec()
    assert vec.array.shape == dvec.shape
    vec.array = dvec
    vec.assemble()
    return vec

#
# Old version: no longer in use, btu kept because the code took some effort.
#

_stored_state = None

def mpi_sample(
    call=(np.random.randn, [], {}),
    seed=None,
    comm=MPI.COMM_WORLD
):
    """Consistent random samples across MPI processes

    Optional keyword parameters:
    call=(np.random.randn, [], {}): call is a tuple of the form
        (callable, args, kwargs). After setting up the numpy random
        state, mpi_sample will return the values returned by
        callable(*args, **kwargs).
    seed=None: This is either an object acceptable to
        numpy.random.set_state or an object acceptable to
        numpy.random.seed.
    comm=MPI.COMM_WORLD: The MPI communicator.

    On the first call, seed is used to seed the numpy.random system in
    process 0. callable(*args, **kwargs) is then called to get the
    results to return on process 0, and the state of the numpy.random
    system is sent to process 1. Process 1 initializes the state from
    the received state, calls callable(*args, **kwargs), and send the
    state on to process 2, etc. The last process sends the state back
    to process 0, where it is stored for use in subsequent call to
    mpi_sample.

    Of course, this forces your MPI program to execute
    sequentially. This is acceptable if you don't expect random number
    generation to be a big part computationally of what your MPi
    program does.
    """
    global _stored_state
    rank = comm.rank
    size = comm.size
    src = rank - 1 if rank > 0 else size - 1
    dst = rank + 1 if rank < size - 1 else 0
    if rank == 0:
        if seed is not None:
            # logRANDOM('trying to seed np.random')
            try:
                np.random.set_state(seed)
            except (ValueError, TypeError):
                np.random.seed(seed)
        elif _stored_state is not None:
            # logRANDOM('restoring stored state')
            np.random.set_state(_stored_state)
        else:
            pass                # just leave the system alone
    else:
        instate = comm.recv(source=src)
        # logRANDOM('received state from', src)
        if isinstance(instate, Exception):
            # logRANDOM('received Exception from src', src)
            comm.send(instate, dest=dst)
            raise instate
        np.random.set_state(instate)
    if 1 == len(call):
        callable = call[0]
        args = []
        kwargs = {}
    elif 2 == len(call):
        callable, args = call
        kwargs = {}
    elif 3 == len(call):
        callable, args, kwargs = call
    else:
        e = ValueError(
            'call must be a tuple of length 1-3'
        )
        comm.send(e, dest=dst)
        raise e
    if not args: args = []
    if not kwargs: kwargs = {}
    try:
        ret = callable(*args, **kwargs)
    except Exception as e:
        if (dst != rank):
            # logRANDOM('sending exception to', dst)
            comm.send(e, dest=dst)
        raise e
    outstate = np.random.get_state()
    if size == 1:               # finished
        # logRANDOM('mpi_sample returns', ret)
        _stored_state = outstate
        return ret
    comm.send(outstate, dest=dst)
    # logRANDOM('sent state to', dst)
    if rank == 0:
        instate = comm.recv(source=src)
        _stored_state = instate
        if isinstance(instate, Exception):
            _stored_state = outstate
            # logRANDOM('received Exception from src', src)
            raise instate
        # logRANDOM('received state from', src)
    # logRANDOM('mpi_sample returns', ret)
    return ret
