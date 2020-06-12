cdef extern from "petsc.h":
    ctypedef double PetscReal
    ctypedef long PetscInt
    ctypedef struct _p_PetscObject
    ctypedef _p_PetscObject* PetscObject
#
# Copied from petscmat.pxi in the petsc4py source distribution
cdef extern from *:
    ctypedef struct PetscMatStencil "MatStencil":
        PetscInt k,j,i,c

#
# Copied from petscdef.pxi in the petsc4py source distribution
cdef extern from * nogil:
    ctypedef enum PetscInsertMode "InsertMode":
        PETSC_NOT_SET_VALUES    "NOT_SET_VALUES"
        PETSC_INSERT_VALUES     "INSERT_VALUES"
        PETSC_ADD_VALUES        "ADD_VALUES"
        PETSC_MAX_VALUES        "MAX_VALUES"
        PETSC_INSERT_ALL_VALUES "INSERT_ALL_VALUES"
        PETSC_ADD_ALL_VALUES    "ADD_ALL_VALUES"
        PETSC_INSERT_BC_VALUES  "INSERT_BC_VALUES"
        PETSC_ADD_BC_VALUES     "ADD_BC_VALUES"

    ctypedef enum PetscMatAssemblyType "MatAssemblyType":
        MAT_FLUSH_ASSEMBLY
        MAT_FINAL_ASSEMBLY

cdef extern from * nogil:
    int MatSetValuesStencil(void*,PetscInt,PetscMatStencil[],PetscInt,PetscMatStencil[],PetscReal[],PetscInsertMode)
    int MatAssemblyBegin(void*,PetscMatAssemblyType)
    int MatAssemblyEnd(void*,PetscMatAssemblyType)
