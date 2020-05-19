"""Factory functions to make KSFD Teim steppers."""
import petsc4py
from mpi4py import MPI
from importlib import import_module

def ksfdTS(
    derivs,
    t0 = 0.0,
    dt = 0.001,
    tmax = 20,
    maxsteps = 100,
    rtol = 1e-5,
    atol = 1e-5,
    restart=True,
    tstype=None,
    finaltime=None,
    rollback_factor = None,
    hmin = None,
    comm = MPI.COMM_WORLD
):
    """
    Create a timestepper

    Required positional argument:
    derivs: the KSFD.Derivatives object to be used to calculate
        derivatives

    Keyword arguments:
    t0=0.0: the initial time.
    dt=0.001: the initial time step.
    maxdt = 0.5: the maximum time step
    tmax=20: the final time.
    maxsteps=100: maximum number of steps to take.
    rtol=1e-5: relative error tolerance.
    atol=1e-5: absolute error tolerance.
    restart=True: whether to set the initial condition to rho0, U0
    tstype=PETSc.TS.Type.ROSW: implicit solver to use.
    finaltime=PETSc.TS.ExactFinalTime.STEPOVER: how to handle
        final time step.
    fpe_factor=0.1: scale time step by this factor on floating
        point error.
    hmin=1e-20: minimum step size
    comm = PETSc.COMM_WORLD

    You can set other options by calling TS member functions
    directly. If you want to pay attention to PETSc command-line
    options, you should call ts.setFromOptions(). Call ts.solve()
    to run the timestepper, call ts.cleanup() to destroy its
    data when you're finished. 

    """
    parent = '.'.join(__name__.split('.')[:-1])
    ksfdts = import_module('.ksfdts', package=parent)
    if tstype is None:
        tstype = petsc4py.PETSc.TS.Type.ROSW,
    if finaltime is None:
        finaltime = petsc4py.PETSc.TS.ExactFinalTime.STEPOVER,
    TS = ksfdts.KSFDTS(
        derivs,
        t0=t0,
        dt=dt,
        tmax=tmax,
        maxsteps=maxsteps,
        rtol=rtol,
        atol=atol,
        restart=restart,
        tstype=tstype,
        finaltime=finaltime,
        rollback_factor=rollback_factor,
        hmin=hmin,
        comm=comm
    )
    return TS


def implicitTS(
    derivs,
    t0 = 0.0,
    dt = 0.001,
    tmax = 20,
    maxsteps = 100,
    rtol = 1e-5,
    atol = 1e-5,
    restart=True,
    tstype=None,
    finaltime=None,
    rollback_factor = None,
    hmin = None,
    comm = MPI.COMM_WORLD
):
    """
    Create an implicit timestepper

    Required positional argument:
    derivs: the KSFD.Derivatives object to be used to calculate
        derivatives

    Keyword arguments:
    t0=0.0: the initial time.
    dt=0.001: the initial time step.
    maxdt = 0.5: the maximum time step
    tmax=20: the final time.
    maxsteps=100: maximum number of steps to take.
    rtol=1e-5: relative error tolerance.
    atol=1e-5: absolute error tolerance.
    restart=True: whether to set the initial condition to rho0, U0
    tstype=PETSc.TS.Type.ROSW: implicit solver to use.
    finaltime=PETSc.TS.ExactFinalTime.STEPOVER: how to handle
        final time step.
    fpe_factor=0.1: scale time step by this factor on floating
        point error.
    hmin=1e-20: minimum step size
    comm = PETSc.COMM_WORLD

    You can set other options by calling TS member functions
    directly. If you want to pay attention to PETSc command-line
    options, you should call ts.setFromOptions(). Call ts.solve()
    to run the timestepper, call ts.cleanup() to destroy its
    data when you're finished. 

    """
    parent = '.'.join(__name__.split('.')[:-1])
    ksfdts = import_module('.ksfdts', package=parent)
    if tstype is None:
        tstype = petsc4py.PETSc.TS.Type.ROSW
    if finaltime is None:
        finaltime = petsc4py.PETSc.TS.ExactFinalTime.STEPOVER
    TS = ksfdts.implicitTS(
        derivs,
        t0=t0,
        dt=dt,
        tmax=tmax,
        maxsteps=maxsteps,
        rtol=rtol,
        atol=atol,
        restart=restart,
        tstype=tstype,
        finaltime=finaltime,
        rollback_factor=rollback_factor,
        hmin=hmin,
        comm=comm
    )
    return TS

