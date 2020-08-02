#! /usr/bin/env python3
#
"""Solve Finite difference solution of Keller-Segel PDE system

Reference: http://hdl.handle.net/10012/15480

The reference is my PhD thesis. It describes the mathematical problem
and also describes results from a DG finite element solution. That
code for the DG solution is in repository KSDG
(https://github.com/leonavery/KSDG). I don't recommend using
that. KSFD is better-designed and easier to use. 

This script, ksfdsolver2.py, is the most important script in the
repo. It is the one you actually run to carry out a numerical
solution. There are some ancillary scripts: tsmerge.py, moviemaker1.py,
and h5images.py that are useful for manipulating and analyzing the
results. I also incldue in the repo some csh scripts, which may be
useful in themslves, but are probably more useful as indications of
how to use the python scripts.

ksfdsolver2.py is controlled by command-line arguments. The arguments
are many and complex, so I always type them into a file,
e.g. options29n, which can be referenced as follows:

    python ksfdsolver2.py @options29n

Files included by indirection in this way may contain comments (as in
python, the comment character is '#'). The file is parsed by the
python shlex module, so you can quote argument to include special
characters such as spaces. I have included in the repo some such
options files, with comments explaining how they work.

Command line arguments are of three types. First, there are PETSc
options. These are framed as follows:

    python ksfdsolver2.py [args] --petsc [petsc options] -- [args]

You may omit the terminating '--' if no arguments follow. PETSc
options are crucial for selectign the nitty-gritty numerical methods
used, e.g. selecting a time stepper or a Krylov solver. Google PETSc
manual and see the manual for more information.

The other two types of arguments are options and parameters. options
have the form '--option=value'. (There are no short-form options.)
Parameters have the form 'symbol=value'. Parameter values are numbers
or real number-valued functions of space and time represented by sympy
expressions. The general intention is that parameters describe the
mathematical problem to be solved and options tell ksfdsolver2 how to
go about solving it. You may define new parameters simply by including
an argument 'param_name=value'. However, for historical reasons some
of the mathematical description of the problem is still controlled by
options. (The --source option is an example.)

List of options:

--cappotential=tophat: This controls the choice of the Vrho component
  of the potential function. There are two choices, tophat and
  witch. This option will probably eventually (soon?) be replaced with
  a rho_potential parameter.

--save=<prefix>: Save the solution in a TimeSeries using the indicated
  prefix. If ksfdsolver2 is run sequentially, the TimeSeries will be
  stored in a single file <prefix>s1r0.h5. (The containing directory
  will be created if necessary.) If run in parallel with MPI, the
  files will either be named <prefx>MPI.h5 or
  <prefix>s<size>r<rank>.h5, where <size> is the number of parallel
  peocesses, depending on whether your h5py python module has been
  built with MPI support. I have never had the privilege of running
  with an MPI-enabled h5py, so the MPI support has never been tested,
  and you should assume it is full of bugs. If your TimeSeries is
  stored in a sequance of files, you will probabyl need to use
  tsmerg.py to merge them into a single file before analyzing
  them. (This I *have* used and tested.)

--check=<prefix>: Checkpoint at each time step. A checkpoitn is a
  TimeSeries containing only a single time point with name
  <prefix>_<n>_, where <n> is the deciaml number of the time step
  (starting at 0). 

--resume=<prefix>: Resume an interrupted solution from a
  checkpoint, e.g. '--resume=checks/checks109/options109a_459_'. You
  may also resume from a solution file created using
  --save. ksfdsolver2 will resume from the last time poitn in this
  case. Thus, in principle checkpointing with --check is
  unnecessary. There is, however, a danger in relying only on
  --save. If your process crashes while the solution file is open, the
  HDF5 files may be left in an unusable state. I therefore always use
  both --save and --check. A broken solution file may be reconstructed
  by merging checkpoint files with tsmerge. By default, a resumed
  solution begins with t0, dt, and lastvart retrvieved from the
  checkpoint (or solution) file. These may be overridden by explicitly
  settign these parameters on the command line.

--restart=<prefix>: This is like --resume, except that t0, dt, and
  lastvart defaults are not taken from the checkpoint file. 

--series_retries=<n>: This option controls retries when opening a
  TimeSeries (i.e., a checkpoint or a solution file). I added this
  option because on the HPC cluster I was using, the /scratch
  filesystem occasionally freezes for a few minutes. If this happend
  when one of my jobs was trying to create a checkpoint, an exception
  was thrown, thus interrupting a solution that might still have days
  to run. If you use, for instance, --series_retries=10, then whn the
  open fails ksfdsolver2 will pause for --series_retry_interval
  seconds, then try again. This will happen ten times before an
  exception is thrown. The default, --series_retries=0, disables
  retries. 

--series_retry_interval=<n>: Number of seconds to pause between
  TimeSeries open retries. Defaults to 60.

--showparams: This option causes ksfdsolver2 to list all parameters
  and their values, then quit immedaitely without solving the PDE
  system.

--noperiodic: At present this option simply cause an exceptio nto be
  thrown. KSFD corruently only knows how to solve on a rectangular
  domain with periodic boundary conditions. The options is reserved
  for solution with Neumann boudnary condition,s should I ever get
  around to writing that.

--onestep: This causes ksfdsolver2 to take a single time step, then
  exit. This is useful mainly to force the computation and compilation
  of all the ufuncs that will be necessary to solve the problem. Yo
  ucan do this sequentially, then run ksfdsolver2 in parallel, knwoing
  that it will not have to fork processes to build ufuncs.

--seed=<n>: A number with which to seed the random number
  generator. The default is (for no particular reason) 793817931. 

--source=<field>=<expression>. This option (which really ought to be a
  parameter) modifies the PDEs by adding a source term to the
  RHS. <field> may be 'rho' or a ligand name (see Ligands below),
  e.g. 'U_1_1'. <expression> is a sympy expression, which may depend
  on t and x (in one dimension, , x and y (in two, or t, x, y, and z
  (in three). 

Parameters:
There is no fixed list of parameters, since you may define a new
parameter called <name> simply by including <name>=<value> as as
argument. (Here <name> stands for a a valid python symbol name. Python
reserved words such as "lambda' may not be used.) The <value> may be
any sympy expression, and it may depend on t and other
parameters. Certain parameters may also depend on x, y, and z. Each
parameter value must resolve to a number. ksfdsolver2 figures out
which order to evaluate your parameters in to get a number. For
instance, if you say:

    'variance_timing_function=ceiling(vtfreq*log(Max(t, start), base))
    vtfreq=2
    start=1
    base=10

then variance_timing_function will become the expression
ceiling(2*log(Max(t, 1), 10), a function of t that can be evaluated to
a number at any time. This particular VTF will inject variance at t=1,
t=sqrt(10), t=10, and twice every decade thereafter, and can be
adjusted with the paramters vtfreq, start, base.

If there are syntax errors or cyclic dependencies (e.g. p1=2*p2,
p2=2*p1) then ksfdsolver2 will throw an extremely uninformative
exception (something I ought to fix). 

Although you may define parameters at will, certain parameters
(detailed below) have special meaning. Also, although any parameter
may in principle be made a function of time, certain parameters are
used at time t0 to set up the problem and never looked at again. You
will onyl be confusing yourself if yo uset width=1+0.001*t. This will
not cause the width of the domain to increase with time, sicne the
domain dimensions are fixed. In the followign listing, parameters are
listed with their default values

Parameters fixed at time t0:
degree=3: Minimum order of the finite difference approximations used.
dim=1: Number of spatial dimensions. Because of PETSc limitations,
    this must be 1, 2, or 3.
nwidth=8: Number of grid points in the x dimension.
nheight=8: Numebr of gird points in the y dimension.
ndepth=8: Number of grid points in the z dimension.
nelements=8: This is a shortcut to set nwidth, nheight, and ndepth
    simultaneously.
randgridnw=0: Number of grid points in the x dimension for the grid
    used to generate a random initial function. The value 0 means to
    use nwidth/4. See the thesis for explanation of how random
    functions are generated.
randgridnh=0: Like randgridnw, but for the y dimension.
randgridnd=0: Like randgridnw, but for the z dimension.
width=1.0: The x dimension of the domain extends from 0 to width.
height=1.0: The y dimension of the domain extends from 0 to height.
depth=1.0: The z dimension of the domain extends from 0 to depth.
maxsteps=1000: Maximum number of time steps.
tmax=200000: Time at which to finish finish.
t0=0.0: Time at beginning of solution.
dt=0.001: First time step.
rtol=1e-5: tolerance for relative error. (This is used with the PETSc
    basic timestep adapter).
atol=1e-5: tolerance for absolute error. (This is used with the PETSc
    basic timestep adapter).

Initial condition parameters:
Nworms=0.0: Total number of worms (i.e., the integral of rho over the
    entire domain.) This number is divided by the measure of the
    domain and added to rho0. (This parameter is no longer useful. Use
    rho0.)
rho0=9000.0: Can be a function of space.
srho0: standard deviation of rho(t0). This may be a function of space.
U0_<g>_<l>='': This is the initial concentration of ligand <l> in
    group <g>. Can be a function of space. If left blank, i.e. the
    default, U0_<g>_<l> is set at each point to
    rho*s_<g>_<l>/gamma_<g>_<l> at this point. (Note that this is not
    typically rho0*s_<g>_<l>/gamma_<g>_<l> because rho differs from
    rho0 but whatever random noise has been added to rho0 to get rho.) 

Variance injection:
A defect in the original model was that, although noise could be
injected at time zero in the form of random variance of the initial
condition, the solutions was deterministic thereafter. In reality, of
course, noise is constantly generated in the worm system because of
the relative small number of animals and Poisson variation. I
therefore added the facility for add noise at any time. This is till
quite crude. I attach to every grid point an independent geometric
Brownian motion such that the variance of log(rho) increases linearly
with time. This turns out to be computationally expensive, not because
the injection of noise itself is expensive, but because the time
stepper must take small steps after noise is injected. Thus, if you
actually inject noise at every time step, solution proceeds
slowy. Noise injection is controlled by the following parameters:

lastvart=0.0: Time of last variance injection. This parameter is
    updated as solution proceeds. It is also saved in checkpoints and
    will be set from the checkpoint unless overriddedn by an explicit
    command line parameter.
variance_rate=0.0: If variance is injected at time t, the amount
    injected is variance_rate*(t-lastvart). The value 0 disables the
    noise injection system. 
variance_timing_function=t/variance_interval: When the tim stepper
    steps to time t, VTF(t)-VTF(lastvart) is calculated. If the
    difference is 1 or more, i.e. if VTF has increased by at least 1,
    then variance is injected. Note that this doesn't mean that
    variance will be injected every variance_timing_interval
    seconds. For instance, if the time stepper take a step > VTI sec,
    there will be only one injection at the end of the time step. (The
    size of the injection will, however, scale with the actual time
    interval.) 
variance_interval=100: With the default variance_timing_function, this
    is the interval between noise injections, more or less.
conserve_worms=False: If True, rho will be scaled after
    variance injecttion such that the total number of worms (i.e., the
    sum of rho over all grid points) remains constant.

The nosie injection system is a work in progress and may change.

Parameters that may usefully be time-dependent:
Umin=1e-7: If ligand concentration at a point falls below Umin, it will
    be set to Umin.
rhmin=1e-7: If rho at a point falls below rhomin, it will be set to
    rhomin.
rhomax=28000: used in computing Vrho. See thesis.
cushion=2000: used in computing Vrho. See thesis.
maxscale=2.0: used in computing Vrho. See thesis.
s2=5.56e-4: Confusingly, this is the parameter called sigma in the
    thesis. 
CFL_safety_factor=0.0: Controls an alternative to PETSc TSAdapt for
    adaptive time stepping. The CFL condition is that the time step
    must be smaller than the half the stencil width divided by the
    absolute value of velocity. If CFL_safety_factor <= 0, this is
    ignored. But if CFL_safety_factor >= 0, the time step will be
    constrained to less than 

         CFL_safety_factor*Dx/max(abs(v(x)))

    where v(x) is the x-component of the worm velocity at x and Dx is
    the distance in the x direction from the center of each stencil to
    its furthest point. The same calculation is done for y and z
    dimensions if present, and the timestep is constrained to be the
    minimum of all of them. 

    CFL_safety_factor may be used in conjunction with TSAdapt. The
    step size will be the minimum of the step sizes determiedn by the
    two methods.

Ligands and ligand parameters:
The architecture of chemical signals is somewhat complex, for reasons
that seemed good at the time I developed it. Liagnds come in
groups. (A group may and usually does contain only a single ligand.)
Within a group each ligand has a weight weight_<g>_<l>, a secretion rate
s_<g>_<l>, a diffusion constant D_<g>_<l>, and a decay rate
gamma<g>_<l>, where <g> is the group number and <l> the ligand number
with that group. The ligand itself has the name U_<g><l>. (Unusually
for python, I start the numbering at 1, not 0.) Each group has
parameters alpha<g> and beta<g>. To compute the potential due to a
ligand group, first compute the weighted sum of the concentrations of
the ligands in that group. The weights, obviously, are the
weight_<g>_<l> parameters. Calling that weighted sum U_<g>, the
potential is -beta<g>*log(alpha_<g>+U_<g>). It would make sense for
that potential to be a parameter, but that is not yet
implemented. Finally, to get the potential due to all the signals, add
up the potential of the groups. The parameters thus are

ngroups=1: The number of ligand groups
nligands_<g>=21: number of ligands in group <g>
alpha_<g>=1.0: As above.
beta_<g>=1.0: As above.
weight_<g>_<l>=1.0: As above.
s_<g>_<l>=1.0: As above
gamma_<g>_<l>=1.0: As above.
D_<g>_<l>=1.0: As above.

Any of alpha, beta, weight, s, gamma, and D may depend on time. ngroup
and nligands<g> are setup parameters only read at time t0.

Environment variables:
KSFDDEBUG: This variable enables the printing to stdout of very
detailed debugging information. The value should be either ALL or a
colon-separated list of subsystems. The available subsystems are MAIN
(this script), RANDOM (the random function generator), SYM (the
sympy-based computer-algebra system), SERIES (the system that saves a
time series in a file), TS (the time-stepper), and UFUNC (the system
that writes sympy functions to C code and compiles it to numpy
ufuncs). KSFDDEBUG=MAIN:TS would cause the emission of debuggign info
from the MAIN and TS subsystems. KSFDDEBUG=ALL, of course causes all
subsystems to emit debugging info. The output is voluminous, probabyl
useful only if you redirect it to a file. 

AUTOWRAP_SCRATCH: If the value of this variable is set and points to a
directory, it will be used as the location of a cache in which
compiled sympy expression will be stored. You should definitely take
advantage of this.
"""
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
                  SpatialExpression, Derivatives, implicitTS,
                  Generator)
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
    parser.add_argument('--series_retries', type=int, default=0,
                        help='# retries to open TimeSeries')
    parser.add_argument('--series_retry_interval', type=int, default=60,
                        help='time (s) between open retries')
    parser.add_argument('--mpiok', action='store_true',
                        help='use parallel HDF5')
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
    sources = [0.0] * (nligands + 1)
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
    cpf = TimeSeries(
        resuming,
        grid=grid,
        mpiok=clargs.mpiok,
        mode='r',
        retries=clargs.series_retries,
        retry_interval=clargs.series_retry_interval
    )
    tlast = cpf.sorted_times()[-1]
    dtparams = [ p for p in clargs.params if p.startswith('dt=') ]
    lastvartparams = [ 
        p for p in clargs.params if p.startswith('lastvart=') 
    ]
    if clargs.resume:
        t = tlast
        if dtparams:          # there was an explicit dt param
            ps.params0['dt'] = float(dtparams[0][3:])
        elif 'dt' in cpf.info:
            ps.params0['dt'] = float(cpf.info['dt'][()])
        elif len(cpf.sorted_times()) >= 2:
            ps.params0['dt'] = tlast - cpf.sort_times()[-2]
        else:
            pass                  # leave default dt unchanged.
        if lastvartparams:      # there was an explicit lastvart param
            ps.params0['lastvart'] = float(lastvartparams[0][9:])
        elif 'lastvart' in cpf.info:
            ps.params0['lastvart'] = float(cpf.info['lastvart'][()])
        elif len(cpf.sorted_times()) >= 2:
            ps.params0['lastvart'] = tlast - cpf.sort_times()[-2]
        else:
            ps.params0['lastvart']=t
    else:
        t = ps.t0
        if lastvartparams:
            ps.params0['lastvart'] = float(lastvartparams[0][9:])
        else:
            ps.params0['lastvart'] = ps.t0
    #
    # This may need to retrieve only a slice
    #
    values = cpf.retrieve_by_time(tlast)
    values = values.copy(order='F')
    cpf.close()
    lastvec.array = values.reshape(lastvec.array.shape, order='F')
    lastvec.assemble()
    logMAIN('lastvec.array', lastvec.array)
    logMAIN('t', t)
    return lastvec,t

def start_values(clargs, grid, ps):
    rnx = (ps.params0['randgridnw'] if ps.params0['randgridnw'] else
           ps.nwidth//4)
    rny = (ps.params0['randgridnh'] if ps.params0['randgridnh'] else
           ps.nheight//4)
    rnz = (ps.params0['randgridnd'] if ps.params0['randgridnd'] else
           ps.ndepth//4)
    rgrid = Grid(dim=ps.dim,
                 width=ps.width, height=ps.height,
                 depth=ps.depth,
                 nx=rnx,
                 ny=rny,
                 nz=rnz,
                 dof=1
        )
    values0 = ps.values0
    murho0 = values0['Nworms']/(ps.width**ps.dim)
    sigma = values0['srho0']
    rvals = rgrid.Sdmda.createGlobalVec()
    rva = rvals.array.reshape(rgrid.Slshape, order='F')
    if sigma == 0.0:
        rva = murho0
    else:
        SpatialExpression(ps, rgrid, sigma)(
            out=(rva,)
        )
        rvals.assemble()
        rng = Generator.get_rng()
        stn_sample = rng.normal(size=rva.shape)
        rva *= stn_sample
        rva += murho0
    rvals.assemble()
    randrho = random_function(grid, randgrid=rgrid, vals=rvals)
    rra = randrho.array.reshape(grid.Slshape, order='F')
    vec = grid.Vdmda.createGlobalVec()
    va = vec.array.reshape(grid.Vlshape, order='F')
    svec = grid.Sdmda.createGlobalVec()
    if values0['rho0']:
        vrho0 = SpatialExpression(ps, grid, values0['rho0'])(
            out=(va[0],)
        )
    else:
        va[0] = 0.0
    va[0] += rra
    U0names = ['U0' + lig.name()[1:] for lig in ps.groups.ligands()]
    for dof, lig in enumerate(ps.groups.ligands()):
        name = 'U0' + lig.name()[1:]
        if (
            name in values0 and
            values0[name] is not None and
            values0[name] is not False and
            values0[name] != ''
        ):
            U0 = SpatialExpression(ps, grid, values0[name])(
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
    kgen = Generator(seed=commandlineArguments.seed, comm=comm)
    if (commandlineArguments.showparams):
        for n,p,d,h in ps.params0.params():
            print(
                '{n}={val} -- {h}'.format(n=n, val=p(), h=h)
            )
        if not in_notebook():
            sys.exit()
    grid = Grid(
        dim=ps.dim,
        dof=ps.nligands+1,      # rho, ligands
        width=ps.width, height=ps.height, depth=ps.depth,
        nx=ps.nwidth, ny=ps.nheight, nz=ps.ndepth
    )
    sources = decode_sources(commandlineArguments.source, ps, grid)
    logMAIN('sources', sources)
    vec0, t = initial_values(commandlineArguments, grid, ps)
    logMAIN('ps.params0', ps.params0)
    logMAIN('ps.values0', ps.values0)
    if commandlineArguments.save:
        tseries = TimeSeries(
            basename=commandlineArguments.save,
            grid=grid,
            mode='w',
            mpiok=commandlineArguments.mpiok,
            retries=commandlineArguments.series_retries,
            retry_interval=commandlineArguments.series_retry_interval,
        )
        tseries.info['commandlineArguments'] = np.string_(dill.dumps(
            commandlineArguments,
            protocol=0
        ))
        tseries.info['SolutionParameters'] = np.string_(
            dill.dumps(ps, recurse=True, protocol=0)
        )
        tseries.info['sources'] = np.string_(
            dill.dumps(sources, protocol=0)
        )
        tseries.info['dt'] = float(ps.params0['dt'])
        if 'lastvart' in ps.params0:
            tseries.info['lastvart'] = float(ps.params0['lastvart'])
        tseries.flush()
    else:
        tseries = None
    vec0.assemble()
    v0a = vec0.array.reshape(grid.Vlshape, order='F')
    lvec0 = grid.Vdmda.createLocalVec()
    derivs = Derivatives(ps, grid, sources=sources, u0=vec0)
    # UJacobian_arrays = derivs.UJacobian_arrays(t=ps.t0)
    grid.Vdmda.globalToLocal(vec0, lvec0)
    lv0a = lvec0.array.reshape(grid.Vashape, order='F')
    logMAIN('lv0a[:]', lv0a[:])
    options = PETSc.Options()
    options.setValue('ts_max_snes_failures', 100)
    resuming = commandlineArguments.resume or commandlineArguments.restart
    if commandlineArguments.onestep:
        truemaxsteps = 1
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
            {
                'prefix': commandlineArguments.check,
                'mpiok': commandlineArguments.mpiok
            }
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
    try:
        vec0.destroy()
    except:
        pass
    try:
        lvec0.destroy()
    except:
        pass
    if MPI.COMM_WORLD.rank == 0:
        print("SNES failures = ", ts.getSNESFailures())
    try:
        PETSc._finalize()
    except PETSc.Error:
        pass
    #
    # This Finalize call produces the message:
    # "Attempting to use an MPI routine after finalizing",
    # however, it I don't do this, I get a SIGSEGV or PETSc.Error at
    # exit.
    #
    MPI.Finalize()

if __name__ == "__main__" and not in_notebook():
    # execute only if run as a script
    main()
