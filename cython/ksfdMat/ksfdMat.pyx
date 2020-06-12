import numpy as pnp             # python version
cimport numpy as cnp            # cython version
cimport cython
cnp.import_array()
import warnings
from petsc4py.PETSc cimport Mat, PetscMat
# Unsigned int type big enough to hold a void*
from libc.stdint cimport uintptr_t
from cython.view cimport array as cvarray
petscIntDtype = pnp.dtype('int32')
petscRealDtype = pnp.dtype('double')

# From PETSc.pyx
#
# --------------------------------------------------------------------

cdef inline int CHKERR(int ierr) with gil:
    if ierr == 0:
        return 0 # no error
    raise RuntimeError('PETSc Exception %d'%ierr)

# --------------------------------------------------------------------

cdef class ksfdMat(Mat):
    cdef object PyMat
    stencilDtype = pnp.dtype([
        ('k', petscIntDtype),
        ('j', petscIntDtype),
        ('i', petscIntDtype),
        ('c', petscIntDtype),
    ])

    def __cinit__(self, Mat mat, *args, **kwargs):
        self.mat = <PetscMat>mat.mat # keep a copy
        self.obj = <PetscObject*> &mat.mat

    def __init__(self, Mat mat, *args, **kwargs):
        Mat.__init__(self)
        self.PyMat = mat        # ensure that mat isn't GC'd
        self.mat = <PetscMat> mat.mat # the PETSc object
        assert self.stencilDtype.itemsize == sizeof(PetscMatStencil)
        
    def __dealloc__(self):
        self.PyMat = None       # DECREF mat so it can be GC'd
        self.mat = NULL

    def objaddr(self):
        cdef PetscMat retv = NULL
        retv = self.mat
        return <uintptr_t>retv

    cpdef pyMat(self):
        return self.PyMat

    cpdef setValuesJacobian(
        self,
        cnp.ndarray rows,
        cnp.ndarray col_offsets,
        cnp.ndarray values,
        PetscInsertMode insert_mode = PETSC_INSERT_VALUES,
    ):
        """Set rows of a Jacobian

        This function is the raison d'etre of the KSFD.Mat class. It
        is intended as an efficient way to set up a Jacobian
        matrix.

        Required arguments:

        rows is an nrows x 4 numpy ndarray of ints specifying the
        rows of the matrix to be modified. The four values in rows[r]
        are i, j, k, and c -- they are copied to a PETSc MatStencil
        structure that specifies a row.

        col_offsets may either be an ncols x 4 numpy ndarray of ints
        or an nrows x ncols x 4 ndarray. If the dimension of
        col_offsets is 3, the first dimension must match that of the
        first dimension of rows. cols_offsets[col] (in the first case)
        of col_offsets[col, row] (in the second case is a list of
        arrays of form [di, dj, dk, c]. The column to be modified is
        that referenced by the MatStencil [i+di, j+dj, k+dk, c], where
        i, j, and k are taken from the corresponding elements of
        rows[row]. c normally specified the degree of freedom, but it
        may also be used to censor rows or col_offsets. Any row such
        that rows[row, 4] < 0 will be skipped, and likewise any column
        such that col_offsets[col, 4] or col_offsets[row, col, 4] < 0
        will be skipped. (This is mainly intended to make it possible
        to use the function when the number of columns to be modified
        is not the same for every row.

        double vals[nows, ncols] contains the vaues to be inserted or
        added (depending on the insert mode) to matyrix A at the
        indicated locations.

        Optional keyword argument:

        insert_mode = PETSc.InsertMode.INSERT_VALUES

        This is a wrapper function that checks types and sizes of the
        arguments, then dispatches to implementation functions.
        """
        cdef PetscMat A = self.mat
        cdef PetscInt testint
        cdef PetscReal testreal
        #
        # Validity checks.
        #
        if (
            values.ndim != 2
            or not pnp.issubdtype(values.dtype, pnp.number)
        ):
            raise ValueError('values must be an nrows x ncols numeric array')
        cdef int nrows, ncols
        nrows = values.shape[0]
        ncols = values.shape[1]
        #
        # Make sure we can cast to a PetscReal (if this raises an
        # exception, just let it propagate)
        #
        testreal = <PetscReal>values[0, 0]
        if (
            rows.ndim != 2
            or rows.shape[0] != nrows
            or rows.shape[1] != 4
            or not pnp.issubdtype(rows.dtype, pnp.integer)
        ):
            raise ValueError('rows must be an nrows x 4 integer array')
        #
        # Make sure we can cast to a PetscInt (if this raises an
        # exception, just let it propagate)
        #
        testint = <PetscInt>rows[0, 0]
        if (
            (col_offsets.ndim != 2 and col_offsets.ndim != 3)
            or col_offsets.shape[col_offsets.ndim-2] != ncols
            or col_offsets.shape[col_offsets.ndim-1] != 4
            or not pnp.issubdtype(col_offsets.dtype, pnp.integer)
        ):
            raise ValueError(
                'col_offsets must be an nrows x ncols x 4 integer array'
                + 'or an ncols x 4 integer array'
            )
        #
        # Now hand off to the appropriate implementation
        #
        if col_offsets.ndim == 2:
            if (
                pnp.issubdtype(values.dtype, pnp.dtype('float64')) and
                pnp.issubdtype(rows.dtype, pnp.dtype('int32')) and
                pnp.issubdtype(col_offsets.dtype, pnp.dtype('int32'))
            ):
                self.sVJci32(rows, col_offsets, values, insert_mode)
            elif (
                pnp.issubdtype(values.dtype, pnp.dtype('float64')) and
                pnp.issubdtype(rows.dtype, pnp.dtype('int64')) and
                pnp.issubdtype(col_offsets.dtype, pnp.dtype('int64'))
            ):
                self.sVJci64(rows, col_offsets, values, insert_mode)
            else:
                self.sVJcGeneric(rows, col_offsets, values, insert_mode)
        else:
            if (
                pnp.issubdtype(values.dtype, pnp.dtype('float64')) and
                pnp.issubdtype(rows.dtype, pnp.dtype('int32')) and
                pnp.issubdtype(col_offsets.dtype, pnp.dtype('int32'))
            ):
                self.sVJrci32(rows, col_offsets, values, insert_mode)
            elif (
                pnp.issubdtype(values.dtype, pnp.dtype('float64')) and
                pnp.issubdtype(rows.dtype, pnp.dtype('int64')) and
                pnp.issubdtype(col_offsets.dtype, pnp.dtype('int64'))
            ):
                self.sVJrci64(rows, col_offsets, values, insert_mode)
            else:
                self.sVJrcGeneric(rows, col_offsets, values, insert_mode)
        return dict(
            nrows = values.shape[0],
            ncols = values.shape[1],
            row_type = rows.dtype
        )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef sVJcGeneric(
        self,
        rows,
        cols,
        vals,
        im
    ):
        warnings.warn(
            'unrecognized data types: using (slow) generic implementation'
        )
        cdef PetscMat A = self.mat
        cdef PetscInsertMode mode = im
        cdef int nc, nr, ncc
        nr = vals.shape[0]
        nc = vals.shape[1]
        #
        # Copy cols and values into guaranteed contiguous array
        #
        pscr = pnp.zeros((nc,), dtype=self.stencilDtype)
        cdef PetscMatStencil[::1] scr = pscr
        pvscr = pnp.zeros((nc,), dtype = petscRealDtype)
        cdef PetscReal[::1] vscr = pvscr
        # Just one row stencil
        prst = pnp.zeros((1,), dtype=self.stencilDtype)
        cdef PetscMatStencil[::1] rst = prst
        for r in range(nr):
            if rows[r, 3] < 0: continue
            rst[0].i = rows[r, 0]
            rst[0].j = rows[r, 1]
            rst[0].k = rows[r, 2]
            rst[0].c = rows[r, 3]
            ncc = 0
            #
            # Even though cols doesn't change, we do the column copy
            # for each iteration in order to correctly censor values
            #
            for c in range(nc):
                if cols[c, 3] < 0: continue
                scr[ncc].i = rows[r, 0] + cols[c, 0]
                scr[ncc].j = rows[r, 1] + cols[c, 1]
                scr[ncc].k = rows[r, 2] + cols[c, 2]
                scr[ncc].c = cols[c, 3]
                vscr[ncc] = vals[r, c]
                ncc += 1
            CHKERR(MatSetValuesStencil(
                A,
                1, &rst[0],
                ncc, &scr[0],
                &vscr[0], mode
            ))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef sVJrcGeneric(self, rows, cols, vals, im):
        warnings.warn(
            'unrecognized data types: using (slow) generic implementation'
        )
        cdef PetscMat A = self.mat
        cdef PetscInsertMode mode = im
        cdef int nc, nr, ncc
        nr = vals.shape[0]
        nc = vals.shape[1]
        #
        # Copy cols and values into guaranteed contiguous array
        #
        pscr = pnp.zeros((nc,), dtype=self.stencilDtype)
        cdef PetscMatStencil[::1] scr = pscr
        pvscr = pnp.zeros((nc,), dtype = petscRealDtype)
        cdef PetscReal[::1] vscr = pvscr
        # Just one row stencil
        prst = pnp.zeros((1,), dtype=self.stencilDtype)
        cdef PetscMatStencil[::1] rst = prst
        for r in range(nr):
            if rows[r, 3] < 0: continue
            rst[0].i = rows[r, 0]
            rst[0].j = rows[r, 1]
            rst[0].k = rows[r, 2]
            rst[0].c = rows[r, 3]
            ncc = 0
            for c in range(nc):
                if cols[r, c, 3] < 0: continue
                scr[ncc].i = rows[r, 0] + cols[r, c, 0]
                scr[ncc].j = rows[r, 1] + cols[r, c, 1]
                scr[ncc].k = rows[r, 2] + cols[r, c, 2]
                scr[ncc].c = cols[r, c, 3]
                vscr[ncc] = vals[r, c]
                ncc += 1
            CHKERR(MatSetValuesStencil(
                A,
                1, &rst[0],
                ncc, &scr[0],
                &vscr[0], mode
            ))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef sVJci32(
        self,
        const cnp.int32_t[:, :] rows,
        const cnp.int32_t[:, :] cols,
        const cnp.float64_t[:, :] vals,
        const PetscInsertMode im
    ):
        cdef PetscMat A = self.mat
        cdef int nc, nr, ncc
        nr = vals.shape[0]
        nc = vals.shape[1]
        #
        # Copy cols and values into guaranteed contiguous array
        #
        pscr = pnp.zeros((nc,), dtype=self.stencilDtype)
        cdef PetscMatStencil[::1] scr = pscr
        pvscr = pnp.zeros((nc,), dtype = petscRealDtype)
        cdef PetscReal[::1] vscr = pvscr
        # Just one row stencil
        prst = pnp.zeros((1,), dtype=self.stencilDtype)
        cdef PetscMatStencil[::1] rst = prst
        for r in range(nr):
            if rows[r, 3] < 0: continue
            rst[0].i = rows[r, 0]
            rst[0].j = rows[r, 1]
            rst[0].k = rows[r, 2]
            rst[0].c = rows[r, 3]
            ncc = 0
            #
            # Even though cols doesn't change, we do the column copy
            # for each iteration in order to correctly censor values
            #
            for c in range(nc):
                if cols[c, 3] < 0: continue
                scr[ncc].i = rows[r, 0] + cols[c, 0]
                scr[ncc].j = rows[r, 1] + cols[c, 1]
                scr[ncc].k = rows[r, 2] + cols[c, 2]
                scr[ncc].c = cols[c, 3]
                vscr[ncc] = vals[r, c]
                ncc += 1
            CHKERR(MatSetValuesStencil(
                A,
                1, &rst[0],
                ncc, &scr[0],
                &vscr[0], im
            ))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef sVJrci32(
        self,
        const cnp.int32_t[:, :] rows,
        const cnp.int32_t[:, :, :] cols,
        const cnp.float64_t[:, :] vals,
        const PetscInsertMode im
    ):
        cdef PetscMat A = self.mat
        cdef int nc, nr, ncc
        nr = vals.shape[0]
        nc = vals.shape[1]
        #
        # Copy cols and values into guaranteed contiguous array
        #
        pscr = pnp.zeros((nc,), dtype=self.stencilDtype)
        cdef PetscMatStencil[::1] scr = pscr
        pvscr = pnp.zeros((nc,), dtype = petscRealDtype)
        cdef PetscReal[::1] vscr = pvscr
        # Just one row stencil
        prst = pnp.zeros((1,), dtype=self.stencilDtype)
        cdef PetscMatStencil[::1] rst = prst
        for r in range(nr):
            if rows[r, 3] < 0: continue
            rst[0].i = rows[r, 0]
            rst[0].j = rows[r, 1]
            rst[0].k = rows[r, 2]
            rst[0].c = rows[r, 3]
            ncc = 0
            for c in range(nc):
                if cols[r, c, 3] < 0: continue
                scr[ncc].i = rows[r, 0] + cols[r, c, 0]
                scr[ncc].j = rows[r, 1] + cols[r, c, 1]
                scr[ncc].k = rows[r, 2] + cols[r, c, 2]
                scr[ncc].c = cols[r, c, 3]
                vscr[ncc] = vals[r, c]
                ncc += 1
            CHKERR(MatSetValuesStencil(
                A,
                1, &rst[0],
                ncc, &scr[0],
                &vscr[0], im
            ))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef sVJci64(
        self,
        const cnp.int64_t[:, :] rows,
        const cnp.int64_t[:, :] cols,
        const cnp.float64_t[:, :] vals,
        const PetscInsertMode im
    ):
        cdef PetscMat A = self.mat
        cdef int nc, nr, ncc
        nr = vals.shape[0]
        nc = vals.shape[1]
        #
        # Copy cols and values into guaranteed contiguous array
        #
        pscr = pnp.zeros((nc,), dtype=self.stencilDtype)
        cdef PetscMatStencil[::1] scr = pscr
        pvscr = pnp.zeros((nc,), dtype = petscRealDtype)
        cdef PetscReal[::1] vscr = pvscr
        # Just one row stencil
        prst = pnp.zeros((1,), dtype=self.stencilDtype)
        cdef PetscMatStencil[::1] rst = prst
        for r in range(nr):
            if rows[r, 3] < 0: continue
            rst[0].i = rows[r, 0]
            rst[0].j = rows[r, 1]
            rst[0].k = rows[r, 2]
            rst[0].c = rows[r, 3]
            ncc = 0
            #
            # Even though cols doesn't change, we do the column copy
            # for each iteration in order to correctly censor values
            #
            for c in range(nc):
                if cols[c, 3] < 0: continue
                scr[ncc].i = rows[r, 0] + cols[c, 0]
                scr[ncc].j = rows[r, 1] + cols[c, 1]
                scr[ncc].k = rows[r, 2] + cols[c, 2]
                scr[ncc].c = cols[c, 3]
                vscr[ncc] = vals[r, c]
                ncc += 1
            CHKERR(MatSetValuesStencil(
                A,
                1, &rst[0],
                ncc, &scr[0],
                &vscr[0], im
            ))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef sVJrci64(
        self,
        const cnp.int64_t[:, :] rows,
        const cnp.int64_t[:, :, :] cols,
        const cnp.float64_t[:, :] vals,
        const PetscInsertMode im
    ):
        cdef PetscMat A = self.mat
        cdef int nc, nr, ncc
        nr = vals.shape[0]
        nc = vals.shape[1]
        #
        # Copy cols and values into guaranteed contiguous array
        #
        pscr = pnp.zeros((nc,), dtype=self.stencilDtype)
        cdef PetscMatStencil[::1] scr = pscr
        pvscr = pnp.zeros((nc,), dtype = petscRealDtype)
        cdef PetscReal[::1] vscr = pvscr
        # Just one row stencil
        prst = pnp.zeros((1,), dtype=self.stencilDtype)
        cdef PetscMatStencil[::1] rst = prst
        for r in range(nr):
            if rows[r, 3] < 0: continue
            rst[0].i = rows[r, 0]
            rst[0].j = rows[r, 1]
            rst[0].k = rows[r, 2]
            rst[0].c = rows[r, 3]
            ncc = 0
            for c in range(nc):
                if cols[r, c, 3] < 0: continue
                scr[ncc].i = rows[r, 0] + cols[r, c, 0]
                scr[ncc].j = rows[r, 1] + cols[r, c, 1]
                scr[ncc].k = rows[r, 2] + cols[r, c, 2]
                scr[ncc].c = cols[r, c, 3]
                vscr[ncc] = vals[r, c]
                ncc += 1
            CHKERR(MatSetValuesStencil(
                A,
                1, &rst[0],
                ncc, &scr[0],
                &vscr[0], im
            ))
