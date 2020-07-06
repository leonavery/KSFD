#!/usr/bin/env python3
"""Class for extracting information about a PDE solution.

Expected usage:
    from KSFD import Solution

    soln = Solution(prefix)
    values = soln.project(t)
    coords = soln.coords()
    params = soln.params(t)

General ideas: First, you should need only to pass a prefix at
creation time. The prefix is like a filename, but various things are
appended to it to get the names of the actual files in which the
solution is stored. (E.g., prefix + '_ts.h5' is the name of the
TimeSeries file in which numerical results are stored.) You shouldn't
need to pass in any other information -- all necessary information
about the problem and solution are extracted from the files. Some of
this information is made available as member attributes,
e.g. soln.timeSeries is the TimeSeries. Information requiring more
analysis is extracted through functions. Typically such a function
will take time as an argument.

Second, this module is intended to be used as an interface module with
Mathematica. Therefore, functions will typically return simple types
that can be converted cleanly to Mathematica types.

Note: the Solution class is not MPI-safe. Run as a single process only.

"""

import sys
import os
import re
import copy
import dill, pickle
import numpy as np
import sympy as sy
import networkx as nx
import h5py
import collections
try:
    from .ksfdargparse import default_parameters
    from .ksfdtimeseries import TimeSeries
    from .ksfdexception import KSFDException
    from .ksfdmakesolver import makeKSFDSolver
    from .ksfdligand import ParameterList, LigandGroups
    from .ksfdsym import Derivatives, safe_sympify
except ImportError:
    from ksfdargparse import default_parameters
    from ksfdtimeseries import TimeSeries
    from ksfdexception import KSFDException
    from ksfdmakesolver import makeKSFDSolver
    from ksfdligand import ParameterList, LigandGroups
    from ksfdsym import Derivatives, safe_sympify


class SolutionParameters:
    """
    This class just serves as a single place to collect information
    about parameters. The most important members are:

    params0: a mappable giving the initial (time t0) values of
        parameters.
    values0: a mappable giving the initial (time t0) values of
        parameters. This differs from params0 in that all symbols that
        are resolvable at time t0 have been substituted with their
        numeric values, whereas the values in params0 may still be
        sympy expressions with parameters represented by free
        symbols. Thus, if you want a number, values0 is the member to
        use. (It is actually just ps.values() stored away for
        convenisnce.)
    groups: a LigandGroups object detailing the ligands.
    Vparams: LigandGroups object containing parameters for use in
        computing V.
    V: a function to compute potential
    funcs: A dict containing as values functions for computing all
        parameters as a function of time.
    tdfuncs: A dict containing as values functions for computing
        time-dependent parameters as a function of time. This is a
        subset of funcs.
    constants: a Dict containing the values of all parametsrs that
        don't vary in time.

    In addition to these main members, there are several members that
        allow convenient access to important individual parameters:
    t0: The time at which the simulation is to start.
    dim: Integer dimension of the space (1, 2, or 3).
    degree: order of the polynomial approximation.
    nwidth, nheight, ndepth: Integer dimensions of the grid. How many
        of these are meangful depends on dim. For instance, if dim ==
        2, only nwidth and nheight are significant.
    width, height, depth: float dimensions of the space. Also
        depending on dim.
    nligands: number of ligands. The number of PDE dofs is
        nligands+1.
    rhomax, cushion, maxscale: The (time t0) values of the
        corresponding parameters of Vrho.

    This class is MPI-safe -- it does nothign that would be affaected
    by running parallel.
    """

    def __init__(
            self,
            clargs              # Command Line Arguments
    ):
        self.clargs = clargs
        self.groups = LigandGroups(clargs)
        self.params0 = ParameterList(default_parameters)
        self.t0 = self.params0['t0']
        self.params0['t'] = self.t0
        self.params0.add(self.groups.params())
        self.cparams = ParameterList() # command-line params
        self.cparams.decode(clargs.params, allow_new=True)
        self.params0.decode(clargs.params, allow_new=True)
        if 'sigma' in self.cparams:
            if 's2' in self.cparams:
                raise KSFDException(
                    's2 and sigma cannot both be specified'
                )
            self.params0['sigma'] = self.cparams['sigma']
            self.params0['s2'] = self.params0['sigma']**2/2
        if 's2' in self.cparams:
            self.params0['sigma'] = sy.sqrt(2.0 * self.cparams['s2'])
            self.params0['s2'] = self.params0['sigma']**2/2
        if not 'nwidth' in self.cparams:
            self.params0['nwidth'] = self.params0['nelements']
        if not 'nheight' in self.cparams:
            self.params0['nheight'] = self.params0['nelements']
        if not 'ndepth' in self.cparams:
            self.params0['ndepth'] = self.params0['nelements']
        self.nwidth = self.params0['nwidth']
        self.nheight = self.params0['nheight']
        self.ndepth = self.params0['ndepth']
        self.groups.fourier_series()
        self.params0.add(self.groups.params()) # in case Fourier added any
        self.Vgroups = copy.deepcopy(self.groups)
        self.Vparams = ParameterList(default_parameters) # for evaluating V
        self.Vparams.add(self.Vgroups.params())
        self.width = self.params0['width']
        self.height = self.params0['height']
        self.depth = self.params0['depth']
        self.dim = self.params0['dim']
        self.degree = self.params0['degree']
        self.nligands = self.groups.nligands()
        self.rhomax = self.params0['rhomax']
        self.cushion = self.params0['cushion']
        self.t0 = self.params0['t0']
        self.maxscale = self.params0['maxscale']
        self.pfuncs()
        self.values0 = self.values()
        self.constants = collections.OrderedDict()
        for k,v in self.values0.items():
            if k not in self.tdfuncs:
                self.constants[k] = v
        def Vfunc(Us, params={}):
            self.Vparams.update(params)  # copy params into ligands
            return self.Vgroups.V(Us)    # compute V
        def Vtophat(rho, params={}):
            tanh = sy.tanh((rho - params['rhomax'])/params['cushion'])
            return params['maxscale'] * params['s2'] * (tanh + 1)
        def Vwitch(rho, params={}):
            tanh = sy.tanh((rho - params['rhomax'])/params['cushion'])
            return (params['maxscale'] * params['s2'] *
                    (tanh + 1) * (rho / params['rhomax'])
            )
        Vcap = Vwitch if self.clargs.cappotential == 'witch' else Vtophat
        def V2(Us, rho, params={}):
            return Vfunc(Us, params=params) + Vcap(rho, params=params)
        self.V = V2

    def __getstate__(self):
        """
        Make SolutionParameters pickleable
        """
        return self.clargs

    def __setstate__(self, clargs):
        self.__init__(clargs)

    def values(self, t=None):
        """
        Values of all parameters at time t as dict

        If not provided, t defaults to self.t0
        """
        if t is None:
            t = self.t0
        return collections.OrderedDict([
            (name, func(t)) for name,func in self.funcs.items()
        ])

    #
    # Parameters not to translate into symbolic form
    #
    non_symbolic_params = [ re.compile(nsp) for nsp in [
        'degree',
        'dim',
        'nelements',
        'nwidth',
        'nheight',
        'ndepth',
        'width',
        'Nworms',
        'ngroups',
        'nligands_\d+',
        'maxsteps',
        'rtol',
        'atol',
        'series_\d+_\d+',
        'rho0',
        'U0_\d+_\d+'
    ]]

    def param_symbols(self):
        """
        Mapping from param names (as strs) to sympy symbols

        Parameters that are important to the internal working of the
        Ligands functions are mapped to numeric values (from params0)
        rather than symbols.
        """
        psyms = collections.OrderedDict()
        for name in self.funcs.keys():
            blocked = False
            for nsp in self.non_symbolic_params:
                if re.fullmatch(nsp, name):
                    blocked = True
                    psyms[name] = self.params0[name]
            if not blocked:
                psyms[name] = sy.Symbol(name)
        return psyms

    def constant_symbols(self):
        """
        Mapping from param names (as strs) to sympy symbols

        Parameters that are important to the internal working of the
        Ligands functions are mapped to numeric values (from params0)
        rather than symbols.
        """
        psyms = collections.OrderedDict()
        for name in self.constants.keys():
            blocked = False
            for nsp in self.non_symbolic_params:
                if re.fullmatch(nsp, name):
                    blocked = True
                    psyms[name] = self.params0[name]
            if not blocked:
                psyms[name] = sy.Symbol(name)
        return psyms

    def time_dependent_symbols(self):
        """
        Returns a dict that maps constants to number, but
        time-dependent parameters to symbols.
        """
        tds = collections.OrderedDict(self.values0)
        for name in self.tdfuncs:
            tds[name] = sy.Symbol(name)
        return tds

    def pfuncs(self):
        """Create functions for parameters
            
        pfuncs returns a tuple of two dicts mapping param names (from
        params0) to functions. Each function has the call signature
        func(t, params={}) and returns the value of the parameter at
        time t. In fact, the functions created by pfuncs all ignore
        params -- it is there only so that the function can be called
        with a params argument without an exception being thrown.

        pfuncs creates a mapping from parameter 't' to the identity
        function, which returns t.

        The return tuple is (funcs, tdfuncs). The dict funcs contains
        a function for every parameter in params0. tdfuncs contains
        function only for time-dependent parameters. There will always
        be at least one of these: 't'. 

        pfuncs begins with a topological sort of the paramters,
        producing an ordred list in which a parameter depends only on
        parameters that occur earlier in the list. This will fail
        with a NetworkX exception if there are cyclic dependencies
        (e.g. p1=2*p2, p2=2*p1).

        pfuncs then proceeds to substitute the expressions for earlier
        expression into later expressions. The result will be a
        numerical value for each parameter, a vlue as a function of
        time, or a value that is a function of time and space. The
        latter two classes are used to make tdfuncs.

        Eventually I plan to extend this to allow expressions (potentials)
        that depend on rho and ligand fields. The required ligand
        parameter is included for this purpose, but is not currently
        used.
        """
        clargs = self.clargs
        t0 = self.t0
        params0 = self.params0
        pgraph = nx.DiGraph()
        leaves = set(sy.symbols('t x y z')[:self.dim+1])
        keys = set(params0.keys()).difference(map(str, leaves))
        pgraph.add_nodes_from(keys)
        for p1,v1 in params0.items():
            if isinstance(v1, str):
                v1 = safe_sympify(v1)
            if (
                v1 is None or
                isinstance(v1, bool) or
                isinstance(v1, int) or
                isinstance(v1, float)
            ):
                continue
            for p2 in v1.free_symbols.difference(leaves):
                pgraph.add_edge(str(p2), p1)
        order = nx.topological_sort(pgraph)
        done = collections.OrderedDict()
        funcs = {}
        tdfuncs = {}
        for k in order:
            pt = params0[k]
            isnum = (
                pt is None or pt == '' or
                isinstance(pt, bool) or
                isinstance(pt, int) or
                isinstance(pt, float)
            )
            if not isnum:
                pt = pt.subs(done)
            done[k] = pt
            pta = pt.free_symbols if not isnum else set()
            if not pta:
                pt0 = pt.evalf() if not isnum else pt
                def func(t, params={}, p0=pt0):
                    return p0
                funcs[str(k)] = func
            elif pta == set([sy.Symbol('t')]):
                lpt = sy.lambdify(sy.Symbol('t'), pt, 'numpy')
                def func(t, params={}, l0=lpt):
                    return l0(t)
                funcs[str(k)] = func
                tdfuncs[str(k)] = func
            else:
                def func(t, params={}, s0=pt):
                    return s0.subs({'t': t})
                funcs[str(k)] = func
                if sy.Symbol('t') in pt.free_symbols:
                    tdfuncs[str(k)] = func
        def identity(t, params={}):
            return(t)
        funcs['t'] = identity
        tdfuncs['t'] = identity
        self.funcs = funcs
        self.tdfuncs = tdfuncs
        return (funcs, tdfuncs)

class Solution():
    def __init__(self,
                 prefix,
        ):
        """Access a KSFD solution.

        Required argument:
        prefix: This is the prefix for the names of the files in which
        the solution is stored. (Typically, it would be the value of
        the --save option to ksfdsolver<d>.py.
        """
        self.prefix = os.path.expanduser(prefix)
        self.prefix = os.path.expandvars(self.prefix)
        self.timeSeries = TimeSeries(prefix, mode='r')
        self.grid = self.timeSeries.grid
        self.commandlineArguments = dill.loads(
            self.timeSeries.info['commandlineArguments'][()]
        )
        self.solutionParameters = dill.loads(
            self.timeSeries.info['SolutionParameters'][()]
        )
        self.sources = dill.loads(
            self.timeSeries.info['sources'][()]
        )
        self.tstimes = self.timeSeries.sorted_times()
        self.tmin, self.tmax = self.tstimes[0], self.tstimes[-1]
        self.derivatives = Derivatives(
            ps=self.solutionParameters,
            grid=self.grid,
            sources=self.sources
        )

    #
    # short forms of some of the most useful members
    #
    @property
    def ps(self):
        return self.solutionParameters

    @property
    def tseries(self):
        return self.timeSeries

    @property
    def clargs(self):
        return self.commandlineArguments

    @property
    def derivs(self):
        return self.derivatives

    @property
    def ligands(self):
        return self.ps.Vgroups.ligands()

    def params(self, t):
        pd = collections.OrderedDict(
            [(name, self.param_funcs[name](t, params=self.params0))
             for name in self.param_names]
        )
        return pd

    def load(self, t):
        """Load solution for time t."""
        self.vec = self.tseries.retrieve_by_time(t)
        
    def images(self, t=None):
        self.load(t)
        assert self.vec.shape == self.grid.globalVshape
        self.ims = self.vec
        return self.ims

    
def main():
    try:
        prefix = sys.argv[1]
        soln = Solution(prefix)
    except IndexError:
        pass

if __name__ == "__main__":
    # execute only if run as a script
    main()
                
