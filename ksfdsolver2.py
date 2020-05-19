#! /usr/bin/env python3
#
import petsc4py
import sys
import os
import re
import copy
import collections
import numpy as np
import sympy as sy
import h5py
import dill
import pickle
from mpi4py import MPI
from argparse import Namespace, RawDescriptionHelpFormatter
from KSFD import (KSFDException, Grid, TimeSeries, random_function,
                  LigandGroups, ParameterList, Parser,
                  default_parameters, SolutionParameters,
                  SpatialExpression, Derivatives, implicitTS)
from KSFD.ksfddebug import log
import KSFD

def logMAIN(*args, **kwargs):
    log(*args, system='MAIN', **kwargs)

lig_help = """
Use --showparams to see ligand and user-defined parameters
"""

def parameter_help(param_list=default_parameters, add_help=lig_help):
    help = 'Parameters:\n'
    for t,d,h in param_list:
        help += t + '=' + str(d) + ' -- ' + h + '\n'
    help += '''
You may define additional user parameters for use in rho0 or sources.
These should be of type float (e.g. 'k0=10.0' rather than 'k0=10')\n
'''
    help += lig_help
    return help    

def parse_commandline(args=None):
    commandlineArguments = Namespace()
    parser = Parser(
        description='Solve Keller-Segel PDEs',
        epilog=parameter_help(),
        formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument('--randgridnw', type=int,
                        help='# divisions in grid width for random rho0')
    parser.add_argument('--randgridnh', type=int,
                        help='# divisions in grid height for random rho0')
    parser.add_argument('--randgridnd', type=int,
                        help='# divisions in grid depth for random rho0')
    parser.add_argument('--cappotential', choices=['tophat', 'witch'],
                        default='tophat',
                        help='potential function for capping rho')
    parser.add_argument('--save',
                        help='filename prefix in which to save results')
    parser.add_argument('--check',
                        help='filename prefix for checkpoints')
    parser.add_argument('--resume',
                        help='resume from last point of a TimeSeries')
    parser.add_argument('--restart',
                        help='restart (i.e., with t=t0) from last point of a TimeSeries')
    parser.add_argument('--showparams', action='store_true',
                        help='print all parameters')
    parser.add_argument('--noperiodic', action='store_true',
                        help='no periodic boundary conditions')
    parser.add_argument('--onestep', action='store_true',
                        help='exit after one step (maxsteps=0)')
    parser.add_argument('--solver', default='petsc',
                        help='linear solver')
    parser.add_argument('--seed', type=int, default=793817931,
                        help='random number generator seed')
    parser.add_argument('--source', type=str, action='append',
                        default=[], help='source function for rho, U_1_1, ...')
    parser.add_argument('--decay', type=str, action='append',
                        default=[], help='decay rate of parameter')
    parser.add_argument('--slope', type=str, action='append',
                        default=[], help='slope of parameter increase')
    parser.add_argument('params', type=str, nargs='*',
                        help='parameter values')
    commandlineArguments = parser.parse_args(
        args=args,
        namespace=commandlineArguments
    )
    return commandlineArguments

def in_notebook():
    try:
        cfg = get_ipython()
        if cfg.config['IPKernelApp']:
            return(True)
    except NameError:
        return(False)
    return(False)

from signal import (signal, NSIG, SIGHUP, SIGINT, SIGPIPE, SIGALRM,
                    SIGTERM, SIGXCPU, SIGXFSZ, SIGVTALRM, SIGPROF,
                    SIGUSR1, SIGUSR2, SIGQUIT, SIGILL, SIGTRAP,
                    SIGABRT, SIGFPE, SIGBUS, SIGSEGV,
                    SIGSYS)

def signal_exception(signum, frame):
    raise KeyboardInterrupt('Caught signal ' + str(signum))

def catch_signals(signals=None):
    """catch all possible signals and turn them into exceptions"""
    terminators = set([
        SIGHUP,
        SIGINT,
        SIGPIPE,
        SIGALRM,
        SIGTERM,
        SIGXCPU,
        SIGXFSZ,
        SIGVTALRM,
        SIGPROF,
        SIGUSR1,
        SIGUSR2,
        SIGQUIT,
        SIGILL,
        SIGTRAP,
        SIGABRT,
        SIGFPE,
        SIGBUS,
        SIGSEGV,
        SIGSYS,
    ])
    if not signals:
        signals = terminators
    for sig in signals:
        try:
            signal(sig, signal_exception)
        except:
            pass

def decode_sources(sargs, ps, grid):
    ligands = ps.groups.ligands()
    from KSFD import find_duplicates, KSFDException
    nligands = ps.nligands
    sources = ['0.0'] * (nligands + 1)
    keys = [ arg.split('=', maxsplit=1)[0] for arg in sargs ]
    dups = find_duplicates(keys)
    if dups:
        raise KSFDException(
            'duplicated sources: ' + ', '.join(dups)
        )
    names = ['rho'] + [ lig.name() for lig in ligands ]
    for k in keys:
        if k not in names:
            raise KSFDException(
                'unknown function: ' + k
            )
    for name in keys:
        snum = keys.index(name)
        fnum = names.index(name)
        sarg = sargs[snum]
        k,val = sarg.split('=', maxsplit=1)
        sources[fnum] = SpatialExpression(ps, grid, val)
    for i,src in enumerate(sources):
        if isinstance(src, str):
            sources[i] = SpatialExpression(ps, grid, src)
    return sources
        
def initial_values(
        clargs,
        grid,
        params
):
    """
    Compute initial values of the functions rho, U1, ..., Un.

    Required positional argument:
    clargs: The commandline arguments. (A Namespace.)
    grid: A KSFD.Grid. The grid on which the solutions are to be
        computed, with nligands+1 dofs.
    params: The SolutionParameters onject containing the parameters

    Returns a 2-tuple: a single PETSc.Vec initialized with initial
    values, followed by the time. This will always be t0, except when
    resuming a previous solution, in which case it will be the time of
    the point at which the solution is being resumed.
    """
    resuming = clargs.resume or clargs.restart
    if resuming:
        return resume_values(clargs, grid, params)
    else:
        return start_values(clargs, grid, params)

def resume_values(clargs, grid, ps):
    lastvec = grid.Vdmda.createGlobalVec()
    #
    # This needs to open MPI, s1r0, or matching s<size>r<rank>
    #
    resuming = clargs.resume or clargs.restart
    cpf = TimeSeries(resuming, grid=grid, mode='r')
    tlast = cpf.sorted_times()[-1]
    if clargs.resume:
        t = tlast
    else:
        t = ps.t0
    #
    # This may need to retrieve only a slice
    #
    values = cpf.retrieve_by_time(tlast)
    lastvec.array = values.reshape(lastvec.array.shape, order='F')
    lastvec.assemble()
    logMAIN('lastvec.array', lastvec.array)
    logMAIN('t', t)
    return lastvec,t

def start_values(clargs, grid, ps):
    rnx = clargs.randgridnw if clargs.randgridnw else ps.nwidth//4
    rny = clargs.randgridnh if clargs.randgridnh else ps.nheight//4
    rnz = clargs.randgridnd if clargs.randgridnd else ps.ndepth//4
    rgrid = Grid(dim=ps.dim,
                 width=ps.width, height=ps.height,
                 depth=ps.depth,
                 nx=rnx,
                 ny=rny,
                 nz=rnz,
                 dof=1
        )
    murho0 = ps.params0['Nworms']/(ps.width**ps.dim)
    sigma = ps.params0['srho0']
    randrho = random_function(grid, randgrid=rgrid, mu=murho0,
                              sigma=sigma)
    rra = randrho.array.reshape(grid.Slshape, order='F')
    vec = grid.Vdmda.createGlobalVec()
    va = vec.array.reshape(grid.Vlshape, order='F')
    svec = grid.Sdmda.createGlobalVec()
    if ps.params0['rho0']:
        vrho0 = SpatialExpression(ps, grid, ps.params0['rho0'])(
            out=(va[0],)
        )
    else:
        va[0] = 0.0
    va[0] += rra
    U0names = ['U0' + lig.name()[1:] for lig in ps.groups.ligands()]
    for dof, lig in enumerate(ps.groups.ligands()):
        name = 'U0' + lig.name()[1:]
        if name in ps.params0 and ps.params0[name]:
            U0 = SpatialExpression(ps, grid, ps.params0[name])(
                out=(va[dof+1],)
            )
        else:
            va[dof+1] = va[0]*float(lig.s/lig.gamma)
    vec.assemble()
    return vec,ps.t0
               
               
def main(*args):
    if args:
        args = list(args)
    else:
        args = sys.argv
    commandlineArguments = parse_commandline(args[1:])
    petsc4py.init(args=(args[0:1] + commandlineArguments.petsc))
    logMAIN('commandlineArguments.petsc', commandlineArguments.petsc)
    #
    # The following must not be done until petsc4py.init has been
    # called.
    #
    from petsc4py import PETSc
    catch_signals()
    comm = MPI.COMM_WORLD
    periodic = not commandlineArguments.noperiodic
    if not periodic:
        raise KSFDException(
            '--periodic=false not implements'
        )
    ps = SolutionParameters(commandlineArguments)
    logMAIN('list(ps.groups.ligands())',
            list(ps.groups.ligands()))
    np.random.seed(commandlineArguments.seed)
    if (commandlineArguments.showparams):
        for n,p,d,h in prmms.params0.params():
            print(
                '{n}={val} -- {h}'.format(n=n, val=p(), h=h)
            )
        if not in_notebook():
            sys.exit()
    logMAIN('ps.params0', ps.params0)
    grid = Grid(
        dim=ps.dim,
        dof=ps.nligands+1,      # rho, ligands
        width=ps.width, height=ps.height, depth=ps.depth,
        nx=ps.nwidth, ny=ps.nheight, nz=ps.ndepth
    )
    sources = decode_sources(commandlineArguments.source, ps, grid)
    logMAIN('sources', sources)
    vec0, t = initial_values(commandlineArguments, grid, ps)
    if commandlineArguments.save:
        tseries = TimeSeries(
            basename=commandlineArguments.save,
            grid=grid,
            mode='w'
        )
        tseries.info['commandlineArguments'] = dill.dumps(
            commandlineArguments,
            protocol=0
        )
        tseries.info['SolutionParameters'] = dill.dumps(ps, recurse=True,
                                                        protocol=0)
        tseries.info['sources'] = dill.dumps(sources, protocol=0)
        tseries.flush()
    else:
        tseries = None
    vec0.assemble()
    v0a = vec0.array.reshape(grid.Vlshape, order='F')
    lvec0 = grid.Vdmda.createLocalVec()
    derivs = Derivatives(ps, grid, sources=sources, u0=vec0)
    UJacobian_arrays = derivs.UJacobian_arrays(t=ps.t0)
    grid.Vdmda.globalToLocal(vec0, lvec0)
    lv0a = lvec0.array.reshape(grid.Vashape, order='F')
    logMAIN('lv0a[:]', lv0a[:])
    options = PETSc.Options()
    options.setValue('ts_max_snes_failures', 100)
    resuming = commandlineArguments.resume or commandlineArguments.restart
    if commandlineArguments.onestep:
        truemaxsteps = 0
    else:
        truemaxsteps = ps.params0['maxsteps']
    ts = implicitTS(derivs,
                    t0 = t,
                    restart = not bool(resuming),
                    rtol = ps.params0['rtol'],
                    atol = ps.params0['atol'],
                    dt = ps.params0['dt'],
                    tmax = ps.params0['tmax'],
                    maxsteps = truemaxsteps)
    logMAIN('ts', str(ts))
    ts.setMonitor(ts.printMonitor)
    logMAIN('printMonitor set')
    if commandlineArguments.save:
        saveMonitor, closeMonitor = ts.makeSaveMonitor(
            timeseries=tseries
        )
        ts.setMonitor(saveMonitor)
        logMAIN('saveMonitor set')
    if commandlineArguments.check:
        ts.setMonitor(
            ts.checkpointMonitor,
            (),
            {'prefix': commandlineArguments.check}
        )
        logMAIN('checkpointMonitor set')
    try:
        logMAIN('calling ts.solve', ts.solve)
        ts.solve()
    except KeyboardInterrupt as e:
        print('KeyboardInterrupt:', str(e))
    except Exception as e:
        print('Exception:', str(e))
        einfo = sys.exc_info()
        sys.excepthook(*einfo)
    if commandlineArguments.save:
        closeMonitor()
        tseries.close()
        logMAIN('saveMonitor closed')
    ts.cleanup()
    tseries.close()
    if MPI.COMM_WORLD.rank == 0:
        print("SNES failures = ", ts.getSNESFailures())

if __name__ == "__main__" and not in_notebook():
    # execute only if run as a script
    main()
