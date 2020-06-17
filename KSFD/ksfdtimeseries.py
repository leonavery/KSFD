"""
MPI-aware read and write PETSc Vec to HDF5

The goal of this module is to save snapshots of a PETSc Vec to HDF5
files, and obviously to read them again later. The obvious way to do
this is parallel HDF5. Unfortunately, distributions of HDF5 and h5py
may be built without support for parallel operation. (In particular,
the conda-forge version doesn't have it.) This is accomplished through
the following kludge:

When a KSFD.TimeSeries is created with name tsname, the runtime
envirnoment is checked to find out if parallel HDF5 is enabled (using
h5py.getconfig().mpi). If so, the data are stored in an HDF5 file
named 

'{name}MPI.h5'.format(name=tsname). 

If not, each process stores the data it owns in a file named

'{name}s{size}r{rank}.h5'.format(name=tsname, size=comm.size, rank=comm.rank)

where comm is the MPI communicator. If run sequentially the data will
all be stored in a file called '{name}s1r0.h5'. It is intended that
the *MPI.h5 file created using parallele HDF5 and the *s1r0.h5 file
created when running sequentially and parallel HDF5 is not available
will be the same. 

The same procedure is used for finding the filename when opening in
read/write mode ('r+' or 'a'). 

When opening a TimeSeries for read (mode 'r') TimeSeries checks (in
order) for the *s<size>r<rank>.h5 file, then the *MPI.h5 file ,and
finally a *s1r0.h5 file, and opens the first it finds. In this case
the retrieve methods will only return the components of the vector
owned by the local process. 

Finally, I will write a simple script to merge all the files of
*s<size>r<rank>.h5 series into a single *MPI.h5 file. In this way an
MPi process group of any size will be able to retrieve data written by
a process group of any size. 
"""
import h5py, os, re, gc
import numpy as np
import petsc4py
from mpi4py import MPI
#
# These imports are placed inside a try/except so that this script can
# be executed standalone to check for syntax errors.
#
try:
    from .ksfddebug import log
    from .ksfdgrid import Grid
except ImportError:
    from ksfddebug import log
    from ksfdgrid import Grid

def logSERIES(*args, **kwargs):
    log(*args, system='SERIES', **kwargs)


class KSFDTimeSeries:
    """
    Base class for TimeSeries

    KSFDTimeSeries is intended as an abstract base class for reading and
    writing time series from KSFD solutions to HDF5 files. It is not
    formally defined as an ABC: you can instantiate it if you really
    wish, but it is not designed to make that a useful thing to do.
    """
    def __init__(
            self,
            basename,
            size=1,
            rank=0,
            mpiok=True,
            mode='r+'
    ):
        """
        Required parameter:

        basename: the prefix of the filename.

        Optional keyword parameters:
        size=1: Number of MPI processes. This typically corresponds to
            comm.size for an MPI communicator comm.
        rank=0: Number of the MPI process that created this
            file. Typically comm.rank.
        mpiOK=True: Whether parallel HDF5 should be used to store to
            store all the data from all MPI processes in a single
            file.
        mode='r+': The file mode for opening the h5py.File.

        size, rank, and mpiok are used mostly to figure out what
        filename to use. They need not correspond to the actual
        current MPU configuration. For instance, they may correspond
        to the config when the time series was created.
        """
        self.get_filename(basename, size, rank, mpiok, mode)
        self._tsf = h5py.File(self.filename, mode=mode,
                              driver=self.driver)
        self._size = size
        self._rank = rank
        self._mode = mode
        _ = self.info           # make sure '/info' exists
        self.try_to_set('size', self.size)
        self.try_to_set('rank', self.rank)
        if 'times' in self.tsf:
            self.ts = np.array(self.tsf['times'][()])
            try:
                self.ks = np.array(self.tsf['ks'][()])
            except KeyError:
                self.ks = np.arange(len(self.ts))
            self.order = np.array(self.tsf['order'][()])
        else:
            self.ts = np.array([], dtype=float)
            self.ks = np.array([], dtype=int)
            self.order = np.array([], dtype=int)
        self.lastk = self.ks.size - 1
        self.sorted = False
        self.tsf.flush()

    def parse_filename(filename):
        """
        filename is a name like 'bases2r1.h5'. parse_filename returns
        (basename, size, rank, mpi) (('base', 2, 1, False) for the
        example). For a filename like 'tests/test1mpi.h5', returns
        ('base', 1, 0, True). 
        """
        mpipat = '(.*)MPI\.h5'
        nompi_pat = '(.*)s(\d+)r(\d+)\.h5'
        res = re.fullmatch(mpipat, filename)
        if res:
            return (res[1], 1, 0, True)
        res = re.fullmatch(nompi_pat, filename)
        if res:
            return (res[1], res[2], res[3], False)
        raise ValueError(
            "Couldn't parse filename {fname}".format(fname=filename)
        )

    def set_grid(self, grid):
        self._grid = grid
        self._dim = grid.dim
        self._dof = grid.dof
        if self.rank_owns_file:
            self._ranges =  grid.ranges
            # if (
            #     'ranges' in self.tsf and
            #     not np.all(self.tsf['ranges'][()] == self.ranges)
            # ):
            #     raise ValueError(
            #         "data ranges {filerange} in {file} doesn't " +
            #         "match grid range {gridrange}".format(
            #             filerange=str(self.tsf['ranges'][()]),
            #             file=self.filename,
            #             gridrange=str(grid.ranges)
            #         )
            #     )
            self.myslice = (slice(0, None),)*(self.dim + 1)
        else:
            self._ranges = ((0, np) for np in grid.nps)
            #
            # Slice of the global array belonging to this process:
            self.myslice = (slice(0, None),) + tuple(
                slice(*r) for r in grid.ranges
            )
        self.try_to_set('ranges', self.ranges)
        
    def get_filename(self, basename, size=1, rank=0, mpiok=True,
                     mode='r+'):
        """
        Get name of file to be opened by this process

        self.filename is set to the name of the HDF5 file to be
        opened. This is also returned as the function value. In
        addition, the following flags are set:
        self.creating: True if creating a new file.
        self.rank_owns_file: True if the file will be exclusively
            owned by this process.
        """
        self.usempi = mpiok and h5py.get_config().mpi
        name_nompi = '{name}s{size}r{rank}.h5'.format(
            name=basename,
            size=size,
            rank=rank
        )
        name_mpi = '{name}MPI.h5'.format(name=basename)
        name_seq = '{name}s1r0.h5'.format(name=basename)
        self.driver = None
        if os.path.isfile(name_nompi):
            self.creating = mode[0] == 'w' or mode[0] == 'x'
            self.rank_owns_file = True
            self.filename = name_nompi
        elif self.usempi and os.path.isfile(name_mpi):
            self.creating = mode[0] == 'w' or mode[0] == 'x'
            self.rank_owns_file = False
            self.filename = name_mpi
        elif (mode == 'r' or mode == 'a') and os.path.isfile(name_seq):
            self.creating = False
            self.rank_owns_file = size == 1
            self.filename = name_seq
            # Allow reading from MPi file even if we're not using MPI:
        elif (mode == 'r' or mode == 'a') and os.path.isfile(name_mpi):
            self.creating = False
            self.rank_owns_file = size == 1
            self.filename = name_mpi
        else:
            self.creating = mode != 'r'
            self.rank_owns_file = not self.usempi
            self.filename = name_mpi if self.usempi else name_nompi
        if self.creating and not self.rank_owns_file and usempi:
            self.driver = 'mpi'
        return self.filename

    def open(self, filename, usempi, mode):
        if mode in ['w', 'w-', 'x', 'a']:
            dirname = os.path.dirname(os.path.abspath(filename))
            try:
                os.makedirs(dirname, exist_ok=True)
            except FileExistsError:
                pass

    def grid_save(self):
        grid = self.grid
        attrs = ['dim', 'dof', 'nps', 'bounds', 'spacing', 'order',
                 'stencil_width', 'stencil_type', 'boundary_type',
                 'globalSshape', 'globalVshape', 'globalCshape', 'Slshape',
                 'Vlshape', 'ranges', 'Clshape', 'Cashape',
                 'coordsNoGhosts', 'coordsWithGhosts',
        ]
        for a in attrs:
            self.try_to_set('/grid/' + a, getattr(grid, a))

    def grid_read(self):
        """Reads grid params from open file, returns dict"""
        ggroup = self.tsf['grid']
        gd = {}
        attrs = ['dim', 'dof', 'nps', 'bounds', 'spacing', 'order',
                 'stencil_width', 'stencil_type', 'boundary_type',
                 'globalSshape', 'globalVshape', 'globalCshape', 'Slshape',
                 'Vlshape', 'ranges', 'Clshape', 'Cashape',
                 'coordsNoGhosts', 'coordsWithGhosts',
        ]
        for a in attrs:
            try:
                val = ggroup[a][()]
                if a.endswith('shape'):
                    gd[a] = tuple(val)
                elif np.isscalar(val):
                    gd[a] = val.item()
                else:
                    gd[a] = val
            except KeyError:
                gd[a] = None
        gd['width'] = gd['bounds'][0]
        gd['height'] = gd['bounds'][1] if gd['dim'] > 1 else 1.0
        gd['depth'] = gd['bounds'][2] if gd['dim'] > 2 else 1.0
        gd['nx'] = gd['nps'][0]
        gd['ny'] = gd['nps'][1] if gd['dim'] > 1 else 8
        gd['nz'] = gd['nps'][2] if gd['dim'] > 2 else 8
        return gd

    def grid_load(self, gd=None):
        """Reads grid params from open file and creates new Grid."""
        if gd is None:
            gd = self.grid_read()
        grid = Grid(
            dim=gd['dim'],
            width=gd['width'],
            height=gd['height'],
            depth=gd['depth'],
            nx=gd['nx'],
            ny=gd['ny'],
            nz=gd['nz'],
            dof=gd['dof'],
            order=gd['order'],
            stencil_width=gd['stencil_width'],
            stencil_type=gd['stencil_type'],
            boundary_type=gd['boundary_type']
        )
        self.set_grid(grid)

    #
    # info is a place for caller to store stuff
    @property
    def info(self):
        """Place for caller to store extra stuff"""
        if not hasattr(self, '_info') or not self._info:
            self._info = self.tsf.require_group('/info')
        return self._info

    @property
    def tsFile(self):
        """The open h5File object"""
        return self._tsf

    @property
    def tsf(self):
        return self._tsf

    @property
    def size(self):
        return self._size

    @property
    def rank(self):
        return self._rank

    @property
    def mode(self):
        return self._mode

    @property
    def ranges(self):
        return self._ranges

    @property
    def comm(self):
        return self._comm

    @property
    def grid(self):
        return self._grid

    @property
    def dim(self):
        return self._dim

    @property
    def dof(self):
        return self._dof

    def try_to_set(self, key, val):
        """Try to set self.tsf[key] to val, but ignore exceptions"""
        if (self.mode == 'r'): return
        try:
            del self.tsf[key]
        except KeyError:
            pass
        try:
            self.tsf[key] = val
        except ValueError:
            pass
        
    def _sort(self):
        if getattr(self, 'sorted', False): return
        ts = getattr(self, 'ts', np.array([]))
        self.try_to_set('times', ts)
        self.order = ts.argsort()
        self.try_to_set('order', self.order)
        self.sts = ts
        self.sts.sort()
        ks = getattr(self, 'ks', [])
        lastk = getattr(self, 'lastk', -1)
        self.try_to_set('ks', ks)
        self.try_to_set('lastk', lastk)
        self.sorted = True

    def flush(self):
        self._sort()
        self.tsf.flush()

    def temp_close(self):
        """
        temp_close closes the HDF5 file in which the TimeSeries is
        stored without destroying associated information. The  file
        can be reopened with little loss of time. temp_close and
        reopen are intended for use during long solutions. If there is
        a crash during solution, a temp-closed TimeSeries will be left
        in a valid state for later use.
        """
        self._sort()
        self.tsf.close()

    def reopen(self):
        """
        Reopen a temp_closed TimeSeries
        """
        mode = self.mode if self.mode == 'r' else 'r+'
        self._tsf = h5py.File(self.filename, mode=mode,
                              driver=self.driver)

    def close(self):
        if not hasattr(self, '_tsf'):
            self.reopen()
        self._sort()
        self.tsf.close()
        del self._tsf
        gc.collect()
        
    def __del__(self):
        self.close()

    def store(self, data, t, k=None):
        if isinstance(data, petsc4py.PETSc.Vec):
            vals = data.array.reshape(self.grid.Vlshape, order='F')
        else:
            vals = data.reshape(self.grid.Vlshape, order='F')
        logSERIES('k, t', k, t)
        if k is None:
            k = self.lastk + 1
        self.lastk = k
        self.ks = np.append(self.ks, k)
        self.ts = np.append(self.ts, t)
        key = 'data' + str(k)
        try:
            dset = self.tsf.create_dataset(key, self.grid.Vlshape,
                                           dtype=vals.dtype)
        except OSError:
            dset = self.tsf[key]     # dset already exists
        Cvals = vals.copy(order='C') # h5py requires C order
        if self.rank_owns_file:
            dset.write_direct(Cvals)
        else:
            dset[self.myslice] = Cvals 
        dset.attrs['k'] = k
        dset.attrs['t'] = t
        self.sorted = False
        self.tsf.flush()

    def store_slice(self, ranges, data, t, tol=1e-7):
        shape = (self.grid.dof,) + tuple(
            r[1] - r[0] for r in ranges
        )
        slc =  (slice(0, None),) + tuple(
            slice(*r) for r in ranges
        )
        vals = data.reshape(shape, order='F')
        na, nb, ta, tb = self.find_time(t)
        logSERIES('na, nb, ta, tb', na, nb, ta, tb)
        if abs(t-ta) <= abs(tb-t):
            n, tn = na, ta
        else:
            n, tn = nb, tb
        if (
                (not (t == 0.0 and tn == 0.0)) and
                ((self.sts.size <= n) or
                 (abs(t-tn)/max(abs(t), abs(tn)) > tol))
        ):
            #
            # New time point: append it to the lists
            #
            k = self.lastk + 1
            self.lastk = k
            self.ks = np.append(self.ks, k)
            self.ts = np.append(self.ts, t)
            key = 'data' + str(k)
            dset = self.tsf.create_dataset(key, self.grid.Vlshape,
                                       dtype=vals.dtype)
            logSERIES('k, t', k, t)
            dset.attrs['k'] = k
            dset.attrs['t'] = t
            self.sorted = False
        else:
            k = n
            key = 'data' + str(k)
            dset = self.tsf[key]
        dset[slc] = vals 
        self.tsf.flush()

    def times(self):
        self._sort()
        return self.ts

    def steps(self):
        self._sort()
        return self.ks

    def sorted_times(self):
        self._sort()
        return self.sts

    def sorted_steps(self):
        self._sort()
        return self.order

    def retrieve_by_number(self, k):
        key = 'data' + str(k)
        dset = self.tsf[key]
        if self.rank_owns_file:
            return np.array(dset)
        else:
            return np.array(dset)[self.myslice]

    def find_time(self, t):
        """
        Find the time points closest to t

        Returns tuple (a, b, ta, tb)
        a and b are the numbers (ints) of the points flanking t. ta
        and tb (floats) are the corresponding times. If there is a
        time point exactly matchig nt, than a == b, ta == tb == t.
        """
        self._sort()
        if self.sts.size == 0:
            return (0, 0, t - 1.0, t - 1.0)
        if (t <= self.sts[0]):
            a = 0
            return (self.ks[a], self.ks[a], self.sts[a], self.sts[a])
        elif (t >= self.sts[-1]):
            a = len(self.sts) - 1
            return (self.ks[a], self.ks[a], self.sts[a], self.sts[a])
        else:
            b = self.sts.searchsorted(t)
        nb = self.order[b]
        tb = self.sts[b]
        if (b >= len(self.order) - 1):
            return(b, b, self.sts[b], self.sts[b])
        elif tb == t:
            return(b, b, tb, tb)
        a = b - 1
        na = self.order[a]
        ta = self.sts[a]
        return (a, b, ta, tb)

    def retrieve_by_time(self, t):
        """
        Retrieve a time point.
        
        Arguments:
        t: the time to be retrieved.
        """
        na, nb, ta, tb = self.find_time(t)
        adata = self.retrieve_by_number(na)
        if na == nb:
            return adata
        bdata = self.retrieve_by_number(nb)
        data = ((t-ta)*bdata + (tb-t)*adata)/(tb-ta)
        return(data)

class TimeSeries(KSFDTimeSeries):

    def __init__(self, basename, grid=None, comm=None, mode='r+'):
        """
        Open a KSFD.TimeSeries

        Required parameters:
        basename: the name of the TimeSeries. (This is a prefix of the
            names of the HDF5 files in which data are stored.)

        Optional parameters:
        grid: The KSFD.Grid on which the PETSc Vecs to be saved are
            defined. This must be supplied when creating a new
            TimeSeries. When opening an existig nseries, it will be
            read from the file if not supplied.
        comm: the MPI communicator. (If not supplied, grid.comm is
            used.)
        mode: the file mode (See h5py.h5File.)
        """
        if comm:
            self._comm = comm
        elif grid:
            self._comm = grid.comm
        else:
            self._comm = MPI.COMM_SELF
        self._mode = mode
        self._size = self.comm.size
        self._rank = self.comm.rank
        super().__init__(basename, size=self.size, rank=self.rank,
                         mode=mode)
        if (grid):
            self.set_grid(grid)
            self.grid_save()
        else:
            self.grid_load()


class Gatherer(KSFDTimeSeries):
    """
    Gatherer is a special-purpose iterator to allow a single
    sequential process to read the separate files written by a
    TimeSeries run under MPI. For instance, to reconstruct the global
    vector at the last time (assuming it fits in memory in a single
    process):

    gather = Gatherer(basename='base', size=4)
    grid = gather.grid
    lastk = gather.sorted_steps()[-1]
    vec = grid.Vdmda.createGlobalVec()
    vecarray = vec.array.reshape(grid.globalVshape, order='F')
    for series in gather:
        vec = grid.Vdmda.createGlobalVec()
        rank = series.rank
        vecarray[series.slice] = series.retrieve_by_number(lastk)
        
    <do something with vec...>

    This gatherer would iterate through files bases4r0.h5,
    bases4r1.h5, bases4r2.h5, and bases4r3.h5. Note that with every
    iteration it closes the last file and opens the next. Thus, if you
    want to iterate over all times, it is more efficient to nest the
    loops like this:

    for series in gather:
        for t in series.times():
            <do something for this file at this time)

    than the other way. (The other way would be more intuitive, but my
    expectation is that this class will be used mostly to gather all
    TimeSeries files into a single file, which then can be processed
    efficiently as a TimeSeries.)
    """
    
    def __init__(
            self,
            basename,
            size=None
    ):
        """
        Required positional parameter
        
        basename: the prefix of the filenames for the TimeSeries being
            read. As a convenience, this can be a special filename
            that matches the regular expression '(.+)s(\d+)@.*' (That
            is a literal '@'. Then the basename is the (.+) and the
            size is the (\d+) following the 's' and preceding
            '@'. For example, "bases4@' or 'bases4@.h5' would both
            serve for a series with basename 'base' and size 4.

        Optional keyword parameter:
        size=None: This argument can be omitted only if the basename
            has the special @ filename format. Otherwise, it must be
            supplied.

        Gatherer is read-only (mode 'r'). 
        """
        gatherre = '(.+)s(\d+)@.*'
        fname_match = re.fullmatch(gatherre, basename)
        if fname_match:
            base = fname_match[1]
            size = int(fname_match[2])
        else:
            base = basename
            size = size
        self.basename = base
        if not isinstance(size, int) or size <= 0:
            raise ValueError(
                'size {size} is not a positive int'
            )
        #
        # This opens the first file. We have to do that so as to read
        # and initialize things like grid, times, etc.
        #
        super().__init__(
            basename=base,
            size=size,
            rank=0,
            mpiok=False,
            mode='r'
        )
        self.set_ranges()
        #
        # Since we have to open the rank 0 file before startig
        # iteration, the following flag is used to determine whether
        # to open a new file when __iter__ is called
        #
        self.iter_started = False
        self.iter_stopped = False

    def set_ranges(self):
        self.rank_owns_file = True
        gd = self.grid_read()
        self.grid_load(gd)
        self._ranges = gd['ranges']
        self._shape = (self.dof,) + tuple(
            r[1] - r[0] for r in self.ranges
        )
        self._slice = (slice(0, None),) + tuple(
            slice(*r) for r in self.ranges
        )
        
    @property
    def slice(self):
        return self._slice

    @property
    def shape(self):
        return self._shape

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.iter_stopped:
            #
            # We previously exhausted the iteration. Restart it
            #
            self.tsf.close()
            self.__init__(self.basename, self.size)
        elif self.iter_started:
            #
            # We're not just starting: move on to next file
            #
            self.tsf.close()
            self._rank = self.rank + 1
            if self.rank >= self.size:
                self.iter_stopped = True
                raise StopIteration
            super().__init__(
                basename=self.basename,
                size=self.size,
                rank=self.rank,
                mpiok=False,
                mode='r'
            )
            self.set_ranges()
        self.iter_started = True
        self.iter_stopped = False
        return self

