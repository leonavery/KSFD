"""Time-steppers for solution of the Keller-Segel PDEs."""

import sys, os, traceback, gc
import numpy as np
from datetime import datetime
import h5py
import pickle
import dill
import petsc4py
from mpi4py import MPI
try:
    from .ksfddebug import log
    from .ksfdtimeseries import TimeSeries
    from .ksfdmat import getMat
    from .ksfdsym import MPIINT, PetscInt
except ImportError:
    from ksfddebug import log
    from ksfdtimeseries import TimeSeries
    from ksfdmat import getMat
    from ksfdsym import MPIINT, PetscInt

def logTS(*args, **kwargs):
    log(*args, system='TS', **kwargs)

def dumpTS(obj):
    for key in dir(obj):
        if key[0:2] != '__':
            try:
                logTS(key, getattr(obj, key))
            except:
                pass

# This module attempts to solve an annoying problem. I want KSFDTS
# to be a subclass of petsc4py.PETSc.TS. However, petsc4py.PETSc is
# not defined until petsc4py.init has been called. And I don't want to
# call petsc4py.init at import time, because it gobbles up argv, which
# is bad if I want to interpret the command line. 'from petsc4py
# import PETSc' automagically calls petsc4py.init, so I can't simply
# use an import statement to import PETSc.
#
# The solution is not to import this module when KSFD is
# imported. (That is, it is not imported by __init__.py.) Instead,
# __init__.py import a small module that defines factory functions
# that import these modules, instantiate the desired classes, and
# return them.
#
# This module will throw an AttributeError if imported before
# petsc4py/init has been called.
#
class KSFDTS(petsc4py.PETSc.TS):
    """Base class for KSFD timesteppers."""

    default_rollback_factor = 0.25
    default_hmin = 1e-20

    def __init__(
        self,
        derivs,
        t0 = 0.0,
        dt = 0.001,
        tmax = 20,
        maxsteps = 100,
        rtol = 1e-5,
        atol = 1e-5,
        restart=True,
        tstype = petsc4py.PETSc.TS.Type.ROSW,
        finaltime = petsc4py.PETSc.TS.ExactFinalTime.STEPOVER,
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
        super().__init__()
        self.create(comm=comm)
        self.mpi_comm = self.comm
        if (not isinstance(self.comm, type(MPI.COMM_SELF))):
            self.mpi_comm = self.comm.tompi4py()
        self.derivs = derivs
        self.t0 = t0
        self.tmax = tmax
        self.maxsteps = maxsteps
        self.rtol = rtol
        self.atol = atol
        self.restart = restart
        self.tstype = tstype
        self.finaltime = finaltime
        popts = petsc4py.PETSc.Options()
        if rollback_factor is None:
            try:
                self.rollback_factor = popts.getReal(
                    'ts_adapt_scale_solve_failed')
            except KeyError:
                self.rollback_factor = self.default_rollback_factor
        else:
            self.rollback_factor = rollback_factor
        if hmin:
            self.hmin = hmin
        else:
            self.hmin = self.default_hmin
        self.setProblemType(self.ProblemType.NONLINEAR)
        self.setMaxSNESFailures(1)
        self.setTolerances(atol=atol, rtol=rtol)
        self.setType(tstype)
        self.history = []
        self.kJ = self.derivs.Jacobian(derivs.u0)
        self.u = self.derivs.u0.duplicate()
        self.u.setUp()
        self.f = self.u.duplicate()
        self.f.setUp()
        self.setSolution(self.u)
        self.setTimeStep(dt)
        self.setMaxSteps(max_steps=maxsteps)
        self.setMaxTime(max_time=tmax)
        self.setExactFinalTime(finaltime)
        self.setMaxTime(tmax)
        self.setMaxSteps(maxsteps)
        self.derivs.u0.copy(self.u)
        self.setTime(self.t0)

    def solve(self, u=None):
        """Run the timestepper.

        Calls ts.setFromOptions before solving. Ideally this would be
        left to the user's control, but solve crashes without.
        """
        if u:
            u.copy(self.u)
        else:
            u = self.u
        t0 = self.t0
        self.setSolution(u)
        self.setTime(t0)
        self.setFromOptions()
        tmax = self.getMaxTime()
        kmax = self.getMaxSteps()
        k = self.getStepNumber()
        h = self.getTimeStep()
        t = self.getTime()
        lastu = u.duplicate()
        lastu.setUp()
        self.monitor(k, t, u)
        while (
                (not self.diverged) and
                k <= kmax and t <= tmax and
                h >= self.hmin
        ):
            lastk, lasth, lastt = k, h, t
            u.copy(lastu)
            lastu.assemble()
            super().step()
            gc.collect()
            k = self.getStepNumber()
            h = self.getTimeStep()
            t = self.getTime()
            u = self.getSolution()
            solvec = self.u.array
            logTS('solvec - lastu.array', solvec - lastu.array)
            self.monitor(k, t, u)

    def cleanup(self):
        """Should be called when finished with a TS

        Leaves history unchanged.
        """
        if hasattr(self, 'J'):
            del self.J
        del self.kJ
        del self.u
        del self.f

    def printMonitor(self, ts, k, t, u):
        """For use as TS monitor. Prints status of solution."""
        if self.comm.rank == 0:
            h = ts.getTimeStep()
            if hasattr(self, 'lastt'):
                dt = t - self.lastt
                out = "clock: %s, step %3d t=%8.3g dt=%8.3g h=%8.3g" % (
                          datetime.now().strftime('%H:%M:%S'), k, t, dt, h
                      )
            else:
                out = "clock: %s, step %3d t=%8.3g h=%8.3g" % (
                          datetime.now().strftime('%H:%M:%S'), k, t, h
                       )
            print(out, flush=True)
            self.lastt = t

    def historyMonitor(self, ts, k, t, u):
        """For use as TS monitor. Stores results in history"""
        h = ts.getTimeStep()
        if not hasattr(self, 'history'):
            self.history = []
        #
        # make a local copy of the dof vector
        #
        self.history.append(dict(
            step = k,
            h = h,
            t = t,
            u = u.array.copy()
        ))

    def checkpointMonitor(self, ts, k, t, u, prefix):
        """For use as TS monitor. Checkpoints results"""
        h = ts.getTimeStep()
        #
        # make a local copy of the dof vector
        #
        cpname = prefix + '_' + str(k) + '.h5'
        cpf = TimeSeries(cpname, grid=self.derivs.grid, mode='w')
        cpf.store(u, t, k=k)
        cpf.close()

    def makeSaveMonitor(self, timeseries):
        """Make a saveMonitor for use as a TS monitor

        Note that makesaveMonitor is not itself a monitor
        function. It returns a callable that can be used as a save
        monitor. Typical usage:

        timeseries = KSFD.TimeSeries(prefix='solution')
        (saveMonitor, closer) = ts.makeSaveMonitor(timeseries)
        ts.setMonitor(save_Monitor)
        ts.solve()
        ...
        closer()

        Required positional argument
        timeseries: The KSFD TimeSeries object into which to store
            time points. It is up to the caller to create and
            initialize this. 
        """
        self.timeseries = timeseries
        def closeSaveMonitor():
            """
            Currently does nothing, Relies on creator to close TimeSeries
            """
            pass
        

        def saveMonitor(ts, k, t, u):
            #
            # reopen and close every time, so file valid after abort
            #
            # self is defined in the closure.
            #
            if not self.timeseries.tsFile:
                self.timeseries.reopen()
            #
            # compute G so it can be saved along with other dofs
            #
            u.assemble()         # just to be safe
            fva = u.array.reshape(self.derivs.grid.Vlshape, order='F')
            Gsave = fva[-1]
            lfvec = self.derivs.grid.Vdmda.getLocalVec()
            self.derivs.grid.Vdmda.globalToLocal(u, lfvec)
            farr = lfvec.array.reshape(self.derivs.grid.Vashape, order='F')
            self.derivs.Guf(farr, t=t, out=(fva[-1],))
            u.assemble()
            self.derivs.grid.Vdmda.restoreLocalVec(lfvec)
            self.timeseries.store(u, t, k=k)
            np.copyto(fva[-1], Gsave)
            u.assemble()
            self.timeseries.temp_close()

        return (saveMonitor, closeSaveMonitor)


class implicitTS(KSFDTS):
    """Fully implicit timestepper."""
    
    def __init__(
        self,
        derivs,
        t0 = 0.0,
        dt = 0.001,
        tmax = 20,
        maxsteps = 100,
        rtol = 1e-5,
        atol = 1e-5,
        restart=True,
        tstype = petsc4py.PETSc.TS.Type.ROSW,
        finaltime = petsc4py.PETSc.TS.ExactFinalTime.STEPOVER,
        rollback_factor = None,
        hmin = None,
        comm = MPI.COMM_WORLD
    ):
        """
        Create an implicit timestepper
        
        Required positional argument:
        derivs: the KSFD.Derivatives object to compute derivatives

        Keyword arguments:
        t0=0.0: the initial time.
        dt=0.001: the initial time step.
        tmax=20: the final time.
        maxsteps=100: maximum number of steps to take.
        rtol=1e-5: relative error tolerance.
        atol=1e-5: absolute error tolerance.
        restart=True: whether to set the initial condition to rho0, U0
        tstype=petsc4py.PETSc.TS.Type.ROSW: implicit solver to use.
        finaltime=petsc4py. PETSc.TS.ExactFinalTime.STEPOVER: how to
            handle final time step. 
        comm = MPi.COMM_WORLD.

        Other options can be set by modifying the PETSc Options
        database.
        """
        logTS("implicitTS __init__ entered")
        super().__init__(
            derivs,
            t0 = t0,
            dt = dt,
            tmax = tmax,
            maxsteps = maxsteps,
            rtol = rtol,
            atol = atol,
            restart=restart,
            tstype = tstype,
            finaltime = finaltime,
            rollback_factor = rollback_factor,
            hmin = hmin,
            comm = comm
        )
        f = self.f
        kJ = self.kJ
        self.setEquationType(self.EquationType.IMPLICIT)
        self.setIFunction(self.implicitIF, f=f)
        self.setIJacobian(self.implicitIJ, J=kJ, P=kJ)

    def implicitIF(self, ts, t, u, udot, f):
        """Fully implicit IFunction for PETSc TS

        This is designed for use as the LHS in a fully implicit PETSc
        time-stepper. The DE to be solved is always A.u' = b(u). u
        corresponds to self.ksfd.sol. A is a constant (i.e., u-independent)
        matrix. In this solver, it is always the identity matrix. b is
        a the u-dependent time derivative of u.

            Fully implicit: A.u' - b(u) = 0
            implicit/explicit: A.u' = b(u)
            Fully explicit: u' = A^(-1).b(u)

        Corresponding to these are seven functions that can be
        provided to PETSc.TS: implicitIF and implicitIJ calculate the
        LHS A.u' - b and its Jacobian for the fully implicit
        form. Arguments: 

        t: the time
        u: a petsc4py.PETSc.Vec containing the state vector
        udot: a petsc4py.PETSc.Vec containing the time derivative u'
        f: a petsc4py.PETSc.Vec in which A.u' - b will be left.
        """
        logTS('implicitIF entered')
        f = self.derivs.dfdt(u, t=t, out=f)
        f.aypx(-1.0, udot)
        f.assemble()
        return

    def implicitIJ(self, ts, t, u, udot, shift, J, B):
        """Fully implicit IJacobian for PETSc TS

        This is designed for use as the LHS in a fully implicit PETSc
        time-stepper. The DE to be solved is always A.u' = b(u). u
        corresponds to self.ksfd.sol. A is a constant (i.e., u-independent)
        matrix. In this solver, it is always the identity matrix. b is
        a the u-dependent time derivative of u.

            Fully implicit: A.u' - b(u) = 0
            implicit/explicit: A.u' = b(u)
            Fully explicit: u' = A^(-1).b(u)

        Corresponding to these are seven functions that can be
        provided to PETSc.TS: implicitIF and implicitIJ calculate the
        LHS A.u' - b and its Jacobian for the fully implicit
        form. Arguments:

        t: the time.
        u: a petsc4py.PETSc.Vec containing the state vector
        udot: a petsc4py.PETSc.Vec containing the u' (not used)
        shift: a real number -- see PETSc Ts documentation.
        J, B: matrices in which the Jacobian shift*A - Jacobian(b(u)) are left. 
        """
        logTS('implicitIJ entered, shift = ', shift)
        u.assemble()
        self.derivs.Jacobian(u, t=t, out=self.kJ)
        self.kJ.setOption(
            petsc4py.PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR,
            False
        )
        self.kJ.scale(-1.0)
        self.kJ.shift(shift)
        self.kJ.assemble()
        J.assemble()
        self.kJ.copy(J)
        J.assemble()
        logTS('J.getDiadonal().array', J.getDiagonal().array)
        if J != B:
            B.assemble()
            J.copy(B)
            B.assemble()
        return True
