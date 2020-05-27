#! /usr/bin/env python3
#
"""
Functions and classes for manipulation and evaluation of symbolic
expressions.
"""
import keyword
import re
import sympy as sy
from numbers import Number
import numpy as np
import uuid, os, tempfile
import collections
import itertools
import math
from mpi4py import MPI
import petsc4py
MPIINT = MPI.INT64_T
PetscInt = 'int32'              # dtype for PETSc Ve and Mat indexes
from sympy.utilities.codegen import make_routine, C99CodeGen
try:
    from .ksfddebug import log
    from .ksfdufunc import (UfuncifyCodeWrapperMultiple, ufuncify,
                            UFUNC_MAXARGS)
    from .ksfdexception import KSFDException
    from .ksfdmat import getMat
except ImportError:
    from ksfddebug import log
    from ksfdufunc import (UfuncifyCodeWrapperMultiple, ufuncify,
                            UFUNC_MAXARGS)
    from ksfdexception import KSFDException
    from ksfdmat import getMat

from dogpile.cache import make_region
if 'AUTOWRAP_SCRATCH' in os.environ:
    cachefilename = os.path.join(
        os.environ['AUTOWRAP_SCRATCH'],
        'ksfdsym_cache.dbm'
    )
    cache_region = make_region().configure(
        'dogpile.cache.dbm',
        arguments = {
            'filename': cachefilename
        }
    )
else:
    from dogpile.cache.backends.null import NullBackend
    cache_region = make_region().configure(
        'dogpile.cache.null'
    )

def logSYM(*args, **kwargs):
    log(*args, system='SYM', **kwargs)

def safe_sympify(exp):
    """
    Does what sympify does, except that it checks the string for
    reserved keywords and raises an exception if there are any.

    sympify itself raises an exception in this case, but the message
    is not informative -- it merely reports a syntax error.
    """
    if isinstance(exp, str):
        wordre = r'\b\w+\b'
        words = re.finditer(wordre, exp)
        for word in words:
            if word.group() in keyword.kwlist:
                raise ValueError(
                    'expression contains keyword {kw}'.format(
                        kw=word.group()
                    )
                )
    return sy.sympify(exp)

def cartesian_product(*arrays, order='F'):
    ndim = len(arrays)
    out = np.stack(np.meshgrid(*arrays), axis=-1).reshape(-1, ndim)
    if order == 'C':
        ixs = np.flip(np.arange(len(arrays)))
        out = out[:, ixs]
    return out

def spatial_expression(
    expr,
    grid,
    params,
    vec=None,
    ufunc_dir=None,
    name='spatial'
):
    """
    Evaluate a symbolic expression for every point on a Grid.

    Required positional arguments:
    expr: the sympy expression to be evaluated.
    grid: the KSFD.Grid on which exp is to be evaluated

    Optional keyword argument:
    params={}: a mappable mapping symbols to floating-point
        values. The keys may be either strs or sympy symbols. The
        values should be numbers.
    vec=None: A PETSc.Vec (typically allocated with
        grid.Sdmda.createGlobalVec()) in which to place the result. (If
        this argument is not supplied, spatial_expression will call
        grid.Sdmda.createGlobalVec() to allocate one. (In either case,
        the Vec is returned as the function value.)
    ufunc_dir=None: The directory in which ufuncs are to be built. A
        temporary directory is used if none is specified.
    name='spatial': name of the function to construct

    Returns : a PETSc Vec containing the values. This Vec is obtained
    with grid.Sdmda.createGlobalVec().

    exp may be any sympy expression containing free symbols 'x', 'y',
    and 'z' as well as symnbols whose values are provided in
    params. It should be something that sympy can evaluate to a float
    when numeric values are given for all free symbols.
    """
    sexpr = safe_sympify(expr).subs(params)
    dim = grid.dim
    xnames = sy.symbols('x y z')[:dim]
    if not sexpr.free_symbols.issubset(xnames):
        raise KSFDEXception(
            'unknown symbols {syms} in initial value {expr}'.format(
                syms = sexpr.free_symbols.difference(set(xnames)),
                expr = expr
            )
        )
    coords = grid.coordsNoGhosts
    rfunc = ufuncify(xnames, sexpr, name=name, verbose=True)
    args = [coords[i] for i in range(dim)]
    if vec is None:
        vec = grid.Sdmda.createGlobalVec()
    vecarray = vec.array.reshape(grid.Slshape, order='F')
    rfunc(*args, out=vecarray)
    vec.assemble()
    return vec

class Derivatives:
    """
    Computes time derivatyives and spatial derivatives of fields of
    the Keller-Segel PDEs. 
    """
    def __init__(self, ps, grid, sources=None, u0=None):
        """
        Required positional parameters:
        ps: A SolutionParameters object specifyign the parameters of
            the problem to be solved.
        grid: The grid on which the problem is to be solved

        Optional keyword arguments:
        sources=None: A list of SpatialExpression objects. Source term
            to be added to the time derivatives. The default, None, is
            equivalent to 0.0 for all sources.
        u0=None: Initial values vector. (0 if None.)
        """
        self.ps = ps
        self.grid = grid
        if sources is None:
            self.sources = [SpatialExpression('0.0')] * (ps.nligands + 1)
        else:
            self.sources = sources
        if u0 is None:
            self.u0 = self.grid.Vdmda.createGlobalVec()
            self.u0.zeroEntries()
        else:
            self.u0 = u0
        self.dim = grid.dim
        self.xnames = sy.symbols('x y z')[:self.dim]
        self.sw = grid.stencil_width
        self.n_stencil_points = 1 + 2*self.sw*self.dim
        ssubs, sts, xcs = self.field_stencil_symbols('rho')
        sts[:, -1] = 0
        self.stencil_subs = ssubs
        self.stencils = sts
        self.xcoords = xcs
        for dm1,lig in enumerate(self.ps.Vgroups.ligands()):
            dof = dm1 + 1
            name = lig.name()
            ssubs, sts, xcs = self.field_stencil_symbols(name)
            sts[:, -1] = dof
            self.stencil_subs.update(ssubs)
            self.stencils = np.append(self.stencils, sts, axis=0)
            self.xcoords = np.append(self.xcoords, xcs, axis=0)
        ssubs, sts, xcs = self.field_stencil_symbols('G')
        sts[:, -1] = self.ps.nligands + 1
        self.stencil_subs.update(ssubs)
        self.stencils = np.append(self.stencils, sts, axis=0)
        self.xcoords = np.append(self.xcoords, xcs, axis=0)
        self.stencil_sym_nums = self.stencil_sym_numbers()

    #
    # The following attributes support just-in-time construction of
    # ufuncs, allowing for faster startup, which is convenient for
    # debugging.
    #
    @property
    def dUdt_ufs(self):
        if not getattr(self, '_dUdt_ufs', None):
            key = 'dUdt_' + str(self.ps.time_dependent_symbols())
            cached = cache_region.get(key=key)
            if cached:
                self._dUdt_ufs = [
                    StencilUfunc(self, expressions=exps,
                                 out_stencils=oss)
                    for exps,oss in cached
                ]
            else:
                self._dUdt_ufs = self.dUdt_ufuncs()
                value = [
                    (suf.expressions, suf.out_stencils)
                    for suf in self._dUdt_ufs
                ]
                cache_region.set(
                    key=key,
                    value=value
                )
        return self._dUdt_ufs
        
    @property
    def Guf(self):
        if not getattr(self, '_Guf', None):
            key = 'Guf_' + str(self.ps.time_dependent_symbols())
            cached = cache_region.get(key=key)
            if cached:
                (exps, oss) = cached
                self._Guf = StencilUfunc(self, expressions=exps,
                                         out_stencils=oss)
            else:
                self._Guf = self.Gufunc()
                cache_region.set(
                    key=key,
                    value=(self._Guf.expressions,
                           self._Guf.out_stencils)
                )
        return self._Guf

    @property
    def divrhogradGuf(self):
        if not getattr(self, '_divrhogradGuf', None):
            key = 'divrhogradGuf_' + str(self.ps.time_dependent_symbols())
            cached = cache_region.get(key=key)
            if cached:
                (exps, oss) = cached
                self._divrhogradGuf = StencilUfunc(self,
                                                   expressions=exps,
                                                   out_stencils=oss)
            else:
                self._divrhogradGuf = self.divrhogradGufunc()
                cache_region.set(
                    key=key,
                    value=(self._divrhogradGuf.expressions,
                           self._divrhogradGuf.out_stencils)
                )
        return self._divrhogradGuf

    @property
    def drhodt_ufs(self):
        if not getattr(self, '_drhodt_ufs', None):
            key = 'drhodt_ufs_' + str(self.ps.time_dependent_symbols())
            cached = cache_region.get(key=key)
            if cached:
                self._drhodt_ufs = [
                    StencilUfunc(self, expressions=exps,
                                 out_stencils=oss)
                    for exps,oss in cached
                ]
            else:
                self._drhodt_ufs = self.drhodt_ufuncs()
                value = [
                    (suf.expressions, suf.out_stencils)
                    for suf in self._drhodt_ufs
                ]
                cache_region.set(
                    key=key,
                    value=value
                )
        return self._drhodt_ufs

    @property
    def Jrhoufs(self):
        if not getattr(self, '_Jrhoufs', None):
            key = 'Jrhoufs_' + str(self.ps.time_dependent_symbols())
            cached = cache_region.get(key=key)
            if cached:
                self._Jrhoufs = [
                    StencilUfunc(self, expressions=exps,
                                 out_stencils=oss)
                    for exps,oss in cached
                ]
            else:
                self._Jrhoufs = self.rhoJacobian_ufuncs()
                value = [
                    (suf.expressions, suf.out_stencils)
                    for suf in self._Jrhoufs
                ]
                cache_region.set(
                    key=key,
                    value=value
                )
        return self._Jrhoufs

    def field_stencil_symbols(self, function):
        """
        Create stencil symbols for a function

        Required positional arguments:
        function: This identifies the functions for which stencil symbols
            are to be created. It may either be a sympy Function or a str
            whose value is the name of the function.

        Returns a tuple (stencil_substitutions, stencils, xcoords):
        stencil_substitutions: A collections.OrderedDict mapping
            expressions of the form function(x, [y, [z]]) to symbols of
            the form _s_function_number. To get a list of the symbols
            order by number, use stencilsubtitutions.values(). The first
            symbol, _s_function_0 will always correspond to function(0, [0,
            [0]]).
        stencils: This is intended for use as the col_offsets
            argument of KSFD.ksfdMat.setValuesJacobian. It is an int
            ndarray of shape (ns, 4), where ns = 1 +
            2*grid.dim*grid.stencil_width is the number of points in
            the stencil. The elements of stencils[n] are di, dj, dk,
            and c, where di, dj, and dk are the offset from stencil
            point i, j, k to that referenced by stencil_syms[n]. c
            will be 0 in all cases -- set it to the desired value with
            stencils[:,-1] = dof.
        xcoords: The coordinates (as a dtype=float ndarray) of
            each point in the stencil.
        """
        if isinstance(function, sy.Function):
            func = function
            name = function.name
        else:
            name = str(function)
            func = sy.Function(name)
        stencils = np.zeros((self.n_stencil_points, 4), dtype=int)
        xcoords = np.zeros((self.n_stencil_points, self.dim),
                           dtype=float)
        current = 1
        for i in range(self.dim):
            nxt = current
            for j in range(-self.sw, 0):
                stencils[nxt, i] = j
                xcoords[nxt, i] = j*self.grid.spacing[i]
                nxt += 1
            for j in range(1, self.sw+1):
                stencils[nxt, i] = j
                xcoords[nxt, i] = j*self.grid.spacing[i]
                nxt += 1
            current = nxt
        lfunc = func(*self.xnames)
        stencil_syms = sy.symarray('_s_'+name, self.n_stencil_points)
        ssubs = collections.OrderedDict(
            (func(*coord), stencil_sym) for
            stencil_sym,coord in zip(stencil_syms, xcoords)
        )
        return (ssubs, stencils, xcoords)

    def stencil_sym_numbers(self):
        """
        Return a dict mapping stencil symbols to their numbers
        """
        stencil_syms = list(self.stencil_subs.values())
        return collections.OrderedDict(
            zip(stencil_syms, itertools.count())
        )

    def diff_stencil(
            self,
            exp,
            order=1,
            subs=None
    ):
        """
        Finite-difference approximation of derivatives of an expression

        Required positional argument:
        exp: The sympy expression to differentiate. The spatial
            dependence of this function should be  via functions of
            space, e.g. rho(x, y) or U_1_1(x, y, z).

        Optional keyword arguments:
        order=1: The order of the derivatives to be taken.
        subs=self.stencil_subs: a mappable containing subsitutions
            for sympy functions, e.g. rho(0.0, 0.0): _s_rho_0. The
            first return argument of field_stencil_symbols is an
            appropriate substitution list for a single function.

        Returns a tuple of length self.dim whose elements are finite
        difference approximations of the derivatives (with respect to
        the spatial coordinates), in terms of the stencil symbols.
        """
        if subs is None:
            subs = self.stencil_subs
        originsubs = {
            x: sy.S(0.0) for x in self.xnames
        }
        ddxs = [None] * self.dim
        for i in range(self.dim):
            points = np.zeros((1 + 2*self.sw, self.dim))
            points[:2*self.sw] = self.xcoords[
                1+2*self.sw*i:1+2*self.sw*(i+1)
            ]
            di = exp.diff(self.xnames[i], order)
            di = sy.Derivative.as_finite_difference(
                di,
                points=points[:,i],
                x0=sy.S(0)
            )
            di = di.subs(originsubs)
            di = di.subs(subs)
            ddxs[i] = di
        return(tuple(ddxs))
            

    def laplacian(self, exp):
        """
        Build a sympy expression for the Laplacian of a spatial field

        Required positional parameters:
        exp: the sympy expression whose Laplacian is to be
            calculated. If this argument is a str, then
            exp(*self.xnames) is used. In this case stencil symbols
            will be constructed from the name exp.

        Returns laplacian_exp, a sympy expression for the Laplacian.
        """
        if isinstance(exp, str):
            name = exp
            exp = sy.Function(name)(*self.xnames)
            ssubs, _, _ = self.field_stencil_symbols(name)
        else:
            ssubs = self.stencil_subs
        ddxs = self.diff_stencil(exp, order=2, subs=ssubs)
        laplacian_exp = sum(ddxs)
        return laplacian_exp


    def laplacian_ufunc(self, exp):
        """
        Build ufunc to compute Laplacian of a field

        Required positional parameters:
        exp: the sympy expression whose Laplacian is to be
            calculated. If this is a str, the expression is
            exp(*self.xnames).
        
        Returns:
        a StencilUfunc object.
        """
        if isinstance(exp, str):
            name = exp
            exp = sy.Function(name)(*self.xnames)
        laplacian_exp = self.laplacian(exp)
        ufunc = StencilUfunc(self, laplacian_exp)
        return ufunc 
        

    def grad(self, exp):
        """
        Build a sympy expression for the gradient of a scalar field

        Required positional parameters:
        exp: the sympy expression whose gradient is to be
            calculated. If exp is a str, exp(*self.xnames) is used.

        Returns a grad_exp, a sympy column vector of dimension self.dim.
        """
        if isinstance(exp, str):
            name = exp
            exp = sy.Function(name)(*self.xnames)
            ssubs, _, _ = self.field_stencil_symbols(name)
        else:
            ssubs = self.stencil_subs
        grad_exp = sy.Matrix(self.diff_stencil(exp, subs=ssubs))
        return grad_exp

    def grad_ufunc(self, exp):
        """
        Build ufunc to compute gradient of a field

        Required positional parameters:
        exp: the sympy expression whose gradient is to be
            calculated. If exp is a str, the expression
            exp(*self.xnames) is used.
        
        Returns a numpy ufunc that computes the appropriate linear
            combinations of the stencil symbols to estimate the
            gradient. This ufunc has self.dim outputs, corresponding
            to the components of the gradient vector.
        """
        if isinstance(exp, str):
            name = exp
            exp = sy.Function(name)(*self.xnames)
            ssubs, _, _ = self.field_stencil_symbols(name)
            args = list(ssubs.values())
        else:
            ssubs = self.stencil_subs
            args = list(ssubs.values())
            args += list(
                exp.free_symbols.difference(args)
            )
        grad_exp = self.diff_stencil(exp, subs=ssubs)
        ufunc = ufuncify(args, grad_exp, name='grad', verbose=True)
        return ufunc


    def divergence(self, density, potential):
        """
        divergence of density times potential gradient

        The flux whose divergence is to be calculated is taken to be a
        density times the gradianet of a potential.

        Required positional parameters:
        density: the sympy expression for the density
        potential: the sympy expression for potential

        Either of these agrument may be a str. In that case, the
        expresson density(*self.xnames) or potential(*self.xnames) is
        used.

        Returns div_exp, a sympy expresion for div(density*grad(potential))
        """
        # if isinstance(density, str):
        #     dname = density
        #     density = sy.Function(density)(*self.xnames)
        #     dsubs, _, _ = self.field_stencil_symbols(dname)
        # else:
        #     dsubs = self.stencil_subs
        # if isinstance(potential, str):
        #     pname = potential
        #     potential = sy.Function(potential)(*self.xnames)
        #     psubs, _, _ = self.field_stencil_symbols(pname)
        # else:
        #     psubs = self.stencil_subs
        plaplacian = self.laplacian(potential)
        dgrad = self.grad(density)
        pgrad = self.grad(potential)
        originsubs = {
            x: sy.S(0.0) for x in self.xnames
        }
        if isinstance(density, str):
            d0 = sy.Symbol('_s_{density}_0'.format(density=density))
        else:
            d0 = density.subs(originsubs).subs(self.stencil_subs)
        div_exp = dgrad.dot(pgrad) + d0*plaplacian
        return div_exp

    def field_stencils(self, dof=1):
        """
        Return the stencils corresponding to the dof
        """
        npoints = self.n_stencil_points
        stencils = self.stencils[
            dof*npoints:(dof+1)*npoints
        ]
        return stencils

    def dUdt_exp(self, ligand, dof=1):
        """
        sympy expression for time derivative of a ligand concentration.

        Required positional arguments:
        ligand: a KSFD.Ligand instance describing the ligand.

        Optional keyword argument:
        dof=1: The dof corresponding to this ligand in the grid.

        Returns dUdt_exp, a sympy expression for the time derivative of ligand
        concentration. 
        """
        name = ligand.name()
        s, D, gamma = [
            sy.Symbol(
                '{n}_{g}_{l}'.format(
                    n=n,
                    g=ligand.groupnum,
                    l=ligand.ligandnum
                )
            ) for n in ['s', 'D', 'gamma']
        ]
        rho_stencil_sym = sy.Symbol('_s_rho_0')
        laplacian_exp = self.laplacian(name)
        dUdtexp = (
            -gamma*sy.Symbol('_s_{name}_0'.format(name=name)) +
            s*rho_stencil_sym +
            D*laplacian_exp
        )
        return dUdtexp

    def dUdt_ufuncs(self):
        """
        Build ufuncs to calculate dU/dt for each ligand

        Returns an array of length nligands of StencilUfunc objects
        for computing the time derivatives of each ligand.
        """
        ufuncs = [ StencilUfunc(self) ] * self.ps.nligands
        for lm1,lig in enumerate(self.ps.Vgroups.ligands()):
            dof = lm1 + 1
            dUdtexp = self.dUdt_exp(lig, dof=dof)
            ufunc = StencilUfunc(self, dUdtexp)
            ufuncs[lm1] = ufunc
        return ufuncs

    def UJacobian_arrays(self, t=None):
        """
        UJacobian_arrays constructs U rows of Jacobian

        Optional keyword argument:
        t=None: The time at which the Jacobian is to be
        evaluated. self.ps.t0 is used if this is not supplied.

        Returns:
        A list of three ndarrays: rows, col_offsets, values. These are
        meant to be used as the corresponding arguments for
        ksfdMat.setValuesJacobian. 
        """
        arrays = []
        ligands = list(self.ps.Vgroups.ligands())
        ssyms = list(self.stencil_sym_nums.keys())
        for lm1,lig in enumerate(ligands):
            dof, lig = lm1+1, lig
            dUdtexp = self.dUdt_exp(lig, dof=dof)
            row_args = [
                np.arange(*range) for range in self.grid._ranges
            ]
            row_args += [np.array([0])] * (3-self.dim)
            row_args += [np.array([dof])]
            rows = cartesian_product(*row_args)
            # ixs = np.arange(4)
            # ixs[:self.dim] = np.flip(np.arange(self.dim))
            # rows = rows[:, ixs]
            args = dUdtexp.free_symbols.intersection(ssyms)
            argnums = np.sort(
                np.array([ self.stencil_sym_nums[sym] for sym in args])
            )
            args = [ ssyms[n] for n in argnums ]
            vals = np.array([
                float(dUdtexp.diff(arg).subs(self.ps.values(t)))
                for arg in args
            ])
            vals = np.broadcast_to(vals, (len(rows), len(vals)))    
            arrays.append((
                rows,
                self.stencils[argnums],
                vals
            ))
        return arrays

    def rhoJacobian_arrays(self, fvec, t=None, out=None):
        """
        rhoJacobian_arrays constructs rho rows of Jacobian

        Optional keyword argument:
        t=None: The time at which the Jacobian is to be
            evaluated. self.ps.t0 is used if this is not supplied.
        out=None: If supplied, this should be a tuple of three ndarrays
            in which to store the results. (The purpsoe is to avoid
            uncessary freeing and reallocation of huge arrays on the
            heap.) Typically out would be a previous return from
            rhoJacobian_errays, used somethign like this:

        out=None
        while ...:
            out = derivs.rhoJacobian_arrays(fvec, t, out=out)
            ...
        del out                 # (or just let it go out of scope)

        Returns:
        A list of three ndarrays: rows, col_offsets, values. These are
        meant to be used as the corresponding arguments for
        ksfdMat.setValuesJacobian. 
        """
        if out is not None:
            rows, col_offsets, values = out
        fvec.assemble()         # just to be safe
        fva = fvec.array.reshape(self.grid.Vlshape, order='F')
        lfvec = self.grid.Vdmda.getLocalVec()
        self.grid.Vdmda.globalToLocal(fvec, lfvec)
        farr = lfvec.array.reshape(self.grid.Vashape, order='F')
        #
        # Build list of nonzero Jacobian elements (by stencil)
        #
        cos = self.stencils
        nonzero_offsets = np.zeros(len(cos), dtype=bool)
        stencil_nums = dict(
            (tuple(stencil), n) for n,stencil in enumerate(cos)
        )
        ncols = np.zeros(len(self.Jrhoufs), dtype=int)
        for i,suf in enumerate(self.Jrhoufs):
            ncols[i] = len(suf.out_stencils)
            for stencil in suf.out_stencils:
                nonzero_offsets[stencil_nums[tuple(stencil)]] = True
        cos = cos[nonzero_offsets]
        if out:
            col_offsets[:] = cos
        else:
            col_offsets = cos
        #
        # Build list of row stencils
        #
        logSYM('self.grid._ranges', self.grid._ranges)
        row_args = [
            np.arange(*range) for range in self.grid._ranges
        ]
        row_args += [np.array([0])] * (3-self.dim)
        row_args += [np.array([0])]
        if out:
            rows[:] = cartesian_product(*row_args)
        else:
            rows = cartesian_product(*row_args)
        # ixs = np.arange(4)
        # ixs[:self.dim] = np.flip(np.arange(self.dim))
        # rows = rows[:, ixs]
        #
        # Fill in the value array
        #
        valshape = self.grid.Slshape + (len(col_offsets),)
        if not out:
            values = np.zeros(valshape, order='F')
        else:
            values = values.reshape(valshape, order='F')
            values[:] = 0.0
        maxouts = np.max(ncols)
        outarrays = np.empty((maxouts,) + self.grid.Slshape)
        for suf in self.Jrhoufs:
            ncols = len(suf.out_stencils)
            outs = tuple(outarrays[i] for i in range(ncols))
            suf(farr, t=t, out=outs) # Jacobian elements from this ufunc
            for out, stencil in zip(outs, suf.out_stencils):
                stnum = stencil_nums[tuple(stencil)]
                values[..., stnum] += out
        values = values.reshape((len(rows), len(col_offsets)), order='F')
        self.grid.Vdmda.restoreLocalVec(lfvec)
        return [rows, col_offsets, values]

    def drhodt(self, fvec, t=None):
        """
        Compute the time derivative of density rho

        Required positional argument:
        fvec: global Vec containing the field values for which the
        derivative is to be computed. This should be created using
        self.grid.Vdmda.createGlobalVec(). It has nligands+2 dofs,
        rho, the ligands, and G. Although it is modified in the course
        of the computation (in particular, G is computed from the
        other fields), it is left unchanged on return.

        Optional keyword argument:
        t=None: The time at which the derivativbe is to be
            evaluated. self.ps.t0 if None

        Returns:
        a numpy array of shape self.grid.Slshape whose values are the
        time derivatives of rho at each point.
        """
        fvec.assemble()         # just to be safe
        fva = fvec.array.reshape(self.grid.Vlshape, order='F')
        lfvec = self.grid.Vdmda.getLocalVec()
        self.grid.Vdmda.globalToLocal(fvec, lfvec)
        farr = lfvec.array.reshape(self.grid.Vashape, order='F')
        drhodtva = self.drhodt_ufs[0](farr)
        for uf in self.drhodt_ufs[1:]:
            drhodtva += uf(farr)
        self.grid.Vdmda.restoreLocalVec(lfvec)
        return drhodtva

    def Jacobian(self, fvec, t=None, out=None, cache=True):
        """
        Compute the Jacobian of the time derivative

        Required positional argument:
        fvec: a PETSc.Vec object containgin the field vector at which
            the Jacobian is the be evaluated.

        Optional keyword argument:
        t=None: The time at which to evalute the Jacobian. self.ps.t0
            is used if this argumnet is not provided.
        out=None: a ksfdMat object in which the Jacobian is to be
            stored. (Typically this would be a previously returned
            Jacobian, e.g.

        out = None
        while ...:
            out = derivs.Jacobian(fvec, out=out)
        ...

        cache=True: This argument tells Jacobian whether to cache
            large arrays (such as the outputs of rhoJacobian_arrays)
            in an attribute for later reuse. It does not determien
            whether such cahced values are used in the current
            call. cached values, if available, are always
            used. However, if the cache argument is False, the cache
            values will nto be available in subsequent calls.
        """
        if out is None:
            J = self.grid.Vdmda.getMatrix()
            kJ = getMat(J)
            kJ.setOption(
                petsc4py.PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR,
                False
            )
            kJ.setUp()
        else:
            kJ = out
        kJ.zeroEntries()
        kJ.assemble()
        if hasattr(self, 'rJas'):
            rJas = self.rJas
        else:
            rJas = None
        UJas = self.UJacobian_arrays(t)
        for UJa in UJas:
            rows, col_offsets, values = UJa
            kJ.assemble()
            kJ.setValuesJacobian(
                rows,
                col_offsets,
                values,
                insert_mode=petsc4py.PETSc.InsertMode.ADD
            )
        rJas = self.rhoJacobian_arrays(fvec, t=t, out=rJas)
        rows, col_offsets, values = rJas
        kJ.setValuesJacobian(
            rows,
            col_offsets,
            values,
            insert_mode=petsc4py.PETSc.InsertMode.ADD
        )
        kJ.assemble()
        if cache:
            self.rJas = rJas
        else:
            if hasattr(self, 'rJas'):
                del self.rJas
        kJ.setOption(
            petsc4py.PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR,
            True
        )
        return kJ

    def dfdt(self, fvec, t=None, out=None):
        """
        Compute time derivative of field vector

        Required positional argument:
        fvec: The field values at which the time derivative is to be
            computed, as a PETSc.Vec.

        Optional keyword argument:
        t=None: The time at which the derivative is to be
            evaluated. self.ps.pt0 if None supplied.
        out=None: If supplied, this should be a PETSc.Vec in which the
            result is to be stored. If not supplied, the result is
            returned in a Vec created with
            self.grid.Vdmda.createGlobalVec(). 
        """
        fvec.assemble()
        lfvec = self.grid.Vdmda.getLocalVec()
        self.grid.Vdmda.globalToLocal(fvec, lfvec)
        farr = lfvec.array.reshape(self.grid.Vashape, order='F')
        rhomin = self.ps.params0['rhomin']
        farr[0] = np.maximum(farr[0], rhomin) # don't take logs of <=0
        if out is None:
            dfdt = self.grid.Vdmda.createGlobalVec()
        else:
            dfdt = out
        dfdt.assemble()
        dfdtarray = dfdt.array.reshape(self.grid.Vlshape, order='F')
        dfdtarray[0] = self.drhodt(fvec, t=t)
        src = self.sources[0](t)
        dfdtarray[0] += src
        for lm1,Uuf in enumerate(self.dUdt_ufs):
            fnum = lm1 + 1
            Uuf(farr, t=t, out=(dfdtarray[fnum],))
            self.sources[fnum](t, out=(src,))
            dfdtarray[fnum] += src
        del src
        self.grid.Vdmda.restoreLocalVec(lfvec)
        dfdt.assemble()
        return dfdt

    def field_functions(self, Us=None, rho=None):
        if Us is None:
            Us = [None]*self.ps.nligands
        else:
            Us = list(Us)
        if len(Us) != self.ps.nligands:
            raise ValueError(
                'Us must yield {nl} expressions'.format(
                    nl=self.ps.nligands
                )
            )
        for i,lig in enumerate(self.ps.Vgroups.ligands()):
            if Us[i] is None:
                Us[i] = sy.Function(lig.name())(*self.xnames)
        if rho is None:
            rho = sy.Function('rho')(*self.xnames)
        return Us, rho

    def V(self, Us=None, rho=None, params=None):
        """
        sympy expression for potential

        Optional keyword arguments:
        Us: a tuple (or other iterable) of nligands U. These can be
            sympy expressions or strs -- str will be converted to
            symbols. If not provided, functional expreesions like
            'U_1_1(x,y)' are constructed, where 'U_1_1' is replaced
            with the functions bearing names of the ligands, and 'x,
            y' with the appropriate (based on dim) spatial coordinate
            names.
        rho: a sympy expression or str for density. Defaults to a
            function call of the form 'rho(x, y)'.

        Returns a sympy expression for potential in terms of rho nad
        the Us and any time-dependent parameters.
        """
        Us, rho = self.field_functions(Us, rho)
        if params is None:
            params = self.ps.time_dependent_symbols()
        return self.ps.V(Us, rho, params=params)

    def G(self, Us=None, rho=None, params=None):
        Us, rho = self.field_functions(Us, rho)
        if params is None:
            params = self.ps.time_dependent_symbols()
        return (
            self.V(Us, rho, params=params) +
            params['s2']*sy.log(rho)
        )

    def Gufunc(self):
        """
        Build ufunc to compute free energy G
        """
        Unames = [
            '_s_{name}_0'.format(name=lig.name())
            for lig in self.ps.Vgroups.ligands()
        ]
        Us = [ sy.Symbol(name) for name in Unames ]
        rho = sy.Symbol('_s_rho_0')
        Gexp = self.G(Us, rho)
        Gufunc = StencilUfunc(self, Gexp)
        return Gufunc

    def divrhogradGufunc(self):
        drgG = self.divergence('rho', 'G')
        return StencilUfunc(self, [drgG])

    @property
    def drhodt_exp(self):
        if not getattr(self, '_drhodt_exp', None):
            self._drhodt_exp = self.make_drhodt_exp()
        return self._drhodt_exp

    def make_drhodt_exp(self):
        drgG = self.divergence('rho', 'G')
        Gsubs = {}
        for sym in drgG.free_symbols:
            name = str(sym)
            if not name.startswith('_s_G_'): continue
            n = name[5:]
            Us = [
                sy.Symbol('_s_{l}_{n}'.format(l=lig.name(), n=n))
                for lig in self.ps.Vgroups.ligands()
            ]
            rho = sy.Symbol('_s_rho_{n}'.format(n=n))
            Gsubs[sym] = self.G(Us, rho)
        drgG = drgG.subs(Gsubs)
        return drgG

    def drhodt_ufuncs(self):
        """
        Construct StencilUfuncs to compute drho/dt

        Returns a list of ufuncs to be called with the local vector
        array. drhodt is the sum of the outputs of these ufuncs. (In
        the simplest case this list will have length 1, and a single
        call will do the job. However, because of the ufunc limiton
        number of arguments, it may be necessary to expand the
        exprepssion and compute it in pieces.
        """
        maxargs = UFUNC_MAXARGS()
        npoints = self.n_stencil_points
        nfields = self.ps.nligands + 1
        drhodtin = self.drhodt_exp
        field_syms = (
            list(self.stencil_sym_nums.keys())[:npoints*nfields]
        )
        field_stencils = self.stencils[:npoints*nfields]
        drhodtout = []
        assert str(field_syms[0]) == '_s_rho_0'
        #
        # Try it the simple eway first
        #
        drhodtsuf = StencilUfunc(
            self,
            expressions=[drhodtin],
            out_stencils=field_stencils[0]
        )
        drhodtout = self.expanded_ufuncs(drhodtsuf)
        return drhodtout

    def rhoJacobian_ufuncs(self):
        """
        Construct StencilUfuncs to compute elements of the rho rows of
        the Jacobian.

        Most of the StencilUfuncs in the list compute more than one
        Jacobian element. They are pooled to the extent possible to
        enable the maximum extent of CSE optimization by the C
        compiler. (In principle, it would be most efficient to pruce a
        single ufunc for all the Jacobian elements, but the limti of
        32 arguments to a single ufunc makes this impossible.) The
        Jacobian elements computed by each StencilUfunc are  indictaed
        by its out_stencils attribute.

        Some Jacobian elements (_s_rho_0 especially) may require
        multiple ufuncs to compute them. In this case the Jacobian
        element is the sum of that computed by all the StencilUfuncs
        that have that out_stencils element.
        """
        maxargs = UFUNC_MAXARGS()
        npoints = self.n_stencil_points
        nfields = self.ps.nligands + 1
        drhodt = self.drhodt_exp
        field_syms = (
            list(self.stencil_sym_nums.keys())[:npoints*nfields]
        )
        field_stencils = self.stencils[:npoints*nfields]
        Jufuncsin = [ None ] * npoints*nfields
        for i in range(npoints*nfields):
            Jufuncsin[i] = StencilUfunc(
                self,
                drhodt.diff(field_syms[i]),
                out_stencils=field_stencils[i]
            )
        Jufuncsout = []
        #
        # Handle _s_rho_0 separately
        #
        assert str(field_syms[0]) == '_s_rho_0'
        assert np.all(
            Jufuncsin[0].out_stencils[0] == np.array([0, 0, 0, 0])
        )
        Jufuncsout = self.expanded_ufuncs(Jufuncsin[0])
        start = 1
        while start < len(Jufuncsin):
            merge = Jufuncsin[start]
            if merge.nargs >= maxargs:
                Jufuncsout += self.expanded_ufuncs(Jufuncsin[start])
                start += 1
                continue
            for nxt in range(start+1, len(Jufuncsin)):
                newmerge = merge.copy()
                success = newmerge.merge(Jufuncsin[nxt]) < maxargs
                if not success: break
                merge = newmerge
            Jufuncsout.append(merge)
            if success: break
            assert nxt < len(Jufuncsin)
            start = nxt
        return Jufuncsout

    def expanded_ufuncs(self, suf):
        """
        Create one or more ufuncs whose outputs sum up to exp.
        """
        maxargs = UFUNC_MAXARGS()
        ufuncsout = []
        if suf.nargs < maxargs: # don't need to expand
            return [ suf ]
        for exp in suf.expressions:
            eexp = sy.expand(exp)
            assert isinstance(eexp, sy.Add)
            ufuncsin = [
                StencilUfunc(self, term, out_stencils=suf.out_stencils)
                for term in eexp.args
            ]
            start = 0
            while start < len(ufuncsin):
                sum = ufuncsin[start]
                for nxt in range(start+1, len(ufuncsin)):
                    newsum = sum.copy()
                    success = newsum.add(ufuncsin[nxt]) < maxargs
                    if not success: break
                    sum = newsum
                assert nxt < len(ufuncsin)
                start = nxt
                ufuncsout.append(sum)
                if success: break
        return ufuncsout
        

class StencilUfunc:
    """
    StencilUfunc -- ufunc that operates on stencil values

    This class encapsulates a numpy ufunc with the information
    necessary to build and call it. The functions ufuncs needed by
    KSFD.Derivatives all have a common structure (beyond the
    commonalities shared by all ufuncs). They all operate on spatial
    fields. Each field is a separate dof of a PETSc Vec. There are
    nligands+2 fields. The 0'th field is always worm density rho. This
    is followed by a field for each ligand. The final field is
    reserved for free energy G. 

    Most of the ufuncs compute finite difference approximations of
    spatial derivatives for points owned by the process on which they
    run. The derivatives of a field F at x are calculated as a linear
    combination of values at x and of points near to x in the
    grid. The points contributing to F'(x) and F''(x) are called the
    stencil. The requisite linear combinations can be computed across
    all the x owned by the local process by passing as arguments array
    slices displaced from center by the appropriate number of grid
    points. Each such displaced array is designated by a stencil
    array, aleways of length 4, e.g. [-1, 0, 0, 1]. This particular
    stencil array specifies points displaced from x by 1 grid spacing
    left in dimensions 0, and with no displacement in other
    dimensions. Elements 1 and 2 are ignored in a one-dimensional
    problem, and element 2 is ignored in a 2D problem. Element 3 is
    the number of the dof. In this case, e.g., it would be the first
    ligand.

    Each stencil array is designated by a sympy symbol of the form
    _s_<field>_<n>, where <field> is the name of the field and <n> the
    decimal number of the stencil point. For instance, _s_rho_0,
    _s_U_1_1_1, and _s_G_2 are stencil symbols. (<n> == 0 is always
    the center point, [0, 0, 0, dof].) Each stencil symbol has a
    number which can be found on the dict
    derivs.stencil_sym_nums. derivs.stencils is an ndarray whose rows
    are rthe corresponding stencil arrays.

    The ufuncs handle by this class take as arguments stencil arrays,
    as detailed above, and time-dependent parameters. They may produce
    multiple outputs.

    Key members:
    expressions: a list of sympy expressions whose values are the
        outputs of the ufunc. This attribute is read/write. 
    inputs: a list a sympy symbols representing the input arguments of
        the ufunc. These are the free symbols of the expression, order
        so that stencil symbols precede time-dependent parameters.
    nargs: Number of arguments. This is just len(expression) +
        len(inputs)
    ufunc: The actual ufunc, if it has been built.

    Key methods:
    merge(): Merges two StencilUfuncs into a single one whose
        expressions are the concatenated expression of the two.
    add(): Adds the ufunc passed as arbgument to that on which it is
        called. Like merge, this method may throw a TooManyAruments
        exception.
    build(): Build the ufunc that does the computation.
    __call__: Call the ufunc. This takes a single required argument,
        an array obtained as grid.Vdmda.getVecArray() containingg the
        local field values (including ghost points). The outputs are
        returned as a list or via the usual ufunc out optional
        argument. If necessary, build is called to build the ufunc.
    """
    
    def __init__(
        self,
        derivs,
        expressions=[],
        out_stencils=[]
    ):
        """
        Required positional argument:
        derivs: The KSFD.Derivatives object that contains the
        information about this problem.

        Optional keyword argument:
        expressions=[]: The list of expression whose values are to be
            the outputs of the ufunc. May be set after creation.
        out_stencils=[]: This argument may be used to associate each
            expression with a stencil.
        """
        self.derivs = derivs
        self.expressions = expressions
        self.out_stencils = out_stencils

    @property
    def expressions(self):
        return self._expressions

    @expressions.setter
    def expressions(self, exps):
        if not isinstance(exps, list):
            exps = [ exps]
        self._expressions = [
            exp.subs(self.derivs.ps.time_dependent_symbols())
            for exp in exps
        ]
        self.out_stencils = []
        self.construct_arguments()

    @property
    def out_stencils(self):
        return self._out_stencils

    @out_stencils.setter
    def out_stencils(self, oss):
        if len(oss) > 0 and oss.ndim < 2:
            oss = np.array([ oss ], dtype=int)
        if len(oss) != 0 and len(oss) != len(self.expressions):
            raise ValueError(
                'Number of out_stencils and expression must match'
            )
        self._out_stencils = oss

    @property
    def nargs(self):
        return len(self.expressions) + len(self.inputs)

    @property
    def inputs(self):
        return self._inputs

    @property
    def stencils(self):
        return self._stencils

    @property
    def ufunc(self):
        return self._ufunc

    def construct_arguments(self):
        exps = self.expressions
        args = set()
        for exp in exps:
            args.update(exp.free_symbols)
        sym_nums = np.array([
            self.derivs.stencil_sym_nums[sym]
            for sym in args
        ], dtype=int)
        sym_nums.sort()
        sym_list = list(self.derivs.stencil_sym_nums.keys())
        stencil_syms = [ sym_list[num] for num in sym_nums ]
        # for sym in self.derivs.stencil_sym_nums.keys():
        #     if sym in args:
        #         stencil_syms.append(sym)
        # stencil_syms = list(args.intersection(
        #     set(self.derivs.stencil_sym_nums.keys())
        # ))
        self._stencils = np.array([
            self.derivs.stencils[self.derivs.stencil_sym_nums[sym]]
            for sym in stencil_syms
        ], dtype=int)
        tdsyms = args.difference(stencil_syms)
        unknowns = tdsyms.difference(sy.symbols(
            list(self.derivs.ps.params0.keys())
        ))
        if unknowns:
            raise ValueError(
                'unknown symbol(s) {unknowns}'.format(
                    unknowns=list(unknowns)
                )
            )
        self._inputs = list(stencil_syms) + list(tdsyms)

    def __str__(self):
        return str(self.expressions)

    def __repr__(self):
        return repr(self.expressions)

    def __call__(
        self,
        array,
        t=None,
        out=None,
        where=True,
        **kwargs
    ):
        return self.call(array, t=t, out=out, where=where, **kwargs)

    def call(
        self,
        array,
        t=None,
        out=None,
        where=True,
        **kwargs
    ):
        """
        Call a ufunc on stencil values and time-dependent parameters

        Required positional arguments:
        array: A vector local array containing the fields that will
            supply the input values. This array is typically
            constructed with grid.Vdmda.createLocalVec().

        Optional keyword arguments:
        t=self.ps.t0: The time at which the ufunc is to be evaluated.
        out=None, where=True, **kwargs. These have the usual meanings
            for ufuncs.

        Stencil symbols are looked up and array sliced so as to
            provide a (typically) off-centered slice, sontaining ghost
            values at the edges. In this way a ufunc is able to
            compute a combination of stencil values for every point in
            the grid in parallel. ps.values(t) is used to get the
            values of any time-dependent parameters.
        """
        if t is None:
            t = self.derivs.ps.t0
        pvalues = self.derivs.ps.values(t)
        ufargs = [
            self.derivs.grid.stencil_slice(stencil, array)
            for stencil in self.stencils
        ]
        for arg in self.inputs[len(self.stencils):]:
            ufargs.append(pvalues[arg])
        return self.ufunc(*ufargs, out=out, where=where, **kwargs)

    def build(self):
        self._ufunc = ufuncify(
            self.inputs,
            self.expressions,
            name='StencilUfunc',
            verbose=True
        )

    @property
    def ufunc(self):
        if not getattr(self, '_ufunc', None):
            self.build()
        return self._ufunc

    def merge(self, suf):
        """
        Merge two StencilUfuncs.

        Required positonal argument:
        suf: The StencilUfunc to be merged with this StencilUfunc. The
        merge is done in place: self is modifed to incldue all the
        expressions it had before and the new ones from suf. If a
        ufunc had been built, it is invalidated.

        Returns self.nargs, the number of arguments after the merge.
        """
        if (
            len(self.out_stencils) == len(self.expressions) and 
            len(suf.out_stencils) == len(suf.expressions)
        ):
            out_stencils = np.append(self.out_stencils,
                                     suf.out_stencils, axis=0)
        else:
            out_stencils = np.array([], dtype=int)
        self.expressions = self.expressions + suf.expressions
        self._out_stencils = out_stencils
        if hasattr(self, '_ufunc'):
            del self._ufunc
        return self.nargs

    def add(self, suf):
        """
        Add two Stencil ufuncs

        Required positional arguemnf:
        suf: the StencilUfunc to be added to this one. The two
            StencilUfuncs must have the same number of
            expressions. The new StencilUfunc has as its expressions
            the sums of the input expressions

        Returns self.nargs, the number of arguments after the add.
        """
        out_stencils = self.out_stencils
        if len(self.expressions) != len(suf.expressions):
            raise ValueError(
                'two StencilUfuncs to be added must have same' +
                'number of expressions'
            )
        exps = self.expressions.copy()
        for i,exp in enumerate(suf.expressions):
            exps[i] += exp
        self.expressions = exps
        self.out_stencils = out_stencils
        return self.nargs

    def copy(self):
        out = StencilUfunc(self.derivs)
        out._expressions = self._expressions
        out._inputs = self._inputs
        out._out_stencils = self._out_stencils
        if hasattr(self, '_ufunc'):
            out._ufunc = self._ufunc
        return out

class SpatialExpression:
    """
    SptailExpression -- evaluate a function of space

    This class encapsulates a numpy ufunc with the information
    necessary to build and call it. The function is defined by a sympy
    expression (or str that is sympified) of x, y, and z (or the
    subset appropriate to the dimension). It may also depend on t, or
    any command line parameters (which may themselves depnd on t). 

    The SpatialExpression object is suppleid a KSFD.Grid when created,
    and this defines x, y, and z at the time of evaluation. A func is
    created by the build() method, and called by the call() method
    with appropriate arguments. call() will call build() to build the
    ufunc if this has not already been done. There is also a __call__
    method, so that the SpatialExpression may be called directly -- it
    delegates to call().

    Key members:
    expression: a sympy expression whose values is to be computed as a
        function of space. This attribute is read/write. 
    inputs: a list a sympy symbols representing the input arguments of
        the ufunc. These are the free symbols of the expression,
        ordered so that x, y, z precede time-dependent parameters.
    ufunc: The actual ufunc, if it has been built.

    Key methods:
    build(): Build the ufunc that does the computation.
    __call__: Call the ufunc. This takes a single required argument,
        an array obtained as grid.Vdmda.getVecArray() containingg the
        local field values (including ghost points). The outputs are
        returned as a list or via the usual ufunc out optional
        argument. If necessary, build is called to build the ufunc.
    """
    
    def __init__(
        self,
        ps,
        grid,
        expression='0.0',
    ):
        """
        Required positional argument:
        ps: The SolutionParameters Object describing the problem
        grid: The KSFD.Grid object.

        Optional keyword argument:
        expressions='0.0': The symp-y expression to be evaluated.

        The values of ps, grid, and expression are avilable as
        properties ps, grid, expression, and parameters. ps and grid
        are read-only. expression may be assigned to after object
        creation.
        """
        self._ps = ps
        self._grid = grid
        self.expression = expression

    @property
    def ps(self):
        return self._ps

    @property
    def grid(self):
        return self._grid

    @property
    def expression(self):
        return self._expression

    @expression.setter
    def expression(self, exp):
        exp = safe_sympify(exp)
        self._expression = exp.subs(
            self.ps.time_dependent_symbols()
        )
        self.construct_arguments()

    @property
    def inputs(self):
        return self._inputs

    @property
    def ufunc(self):
        return self._ufunc

    def construct_arguments(self):
        dim = self.grid.dim
        exp = self.expression
        args = set()
        args.update(exp.free_symbols)
        syms = sy.symbols('x y z')[:dim]
        syms = syms + (sy.Symbol('t'),)
        tdsyms = args.difference(syms)
        unknowns = tdsyms.difference(sy.symbols(
            list(self.ps.time_dependent_symbols().keys())
        ))
        if unknowns:
            raise ValueError(
                'unknown symbol(s) {unknowns}'.format(
                    unknowns=list(unknowns)
                )
            )
        self._inputs = list(syms) + list(tdsyms)

    def __call__(
        self,
        t=None,
        out=None,
        where=True,
        **kwargs
    ):
        return self.call(t=t, out=out, where=where, **kwargs)

    def call(
        self,
        t=None,
        out=None,
        where=True,
        **kwargs
    ):
        """
        Evaluate a spatial expression on its grid

        Optional keyword arguments:
        t=self.ps.t0: The time at which the ufunc is to be evaluated.
        out=None, where=True, **kwargs. These have the usual meanings
            for ufuncs.

        Returns a numpy array of shape self.grid.Slshape.
        """
        if t is None:
            t = self.ps.t0
        pvalues = self.ps.values(t)
        coords = self.grid.coordsNoGhosts
        ufargs = [
            coords[i] for i in range(self.grid.dim)
        ]
        for arg in self.inputs[self.grid.dim:]:
            ufargs.append(pvalues[str(arg)])
        return self.ufunc(*ufargs, out=out, where=where, **kwargs)

    def build(self):
        self._ufunc = ufuncify(
            self.inputs,
            self.expression,
            name='SpatialExpression',
            verbose=True
        )

    @property
    def ufunc(self):
        if not getattr(self, '_ufunc', None):
            self.build()
        return self._ufunc

    def __str__(self):
        return str(self.expression)

    def __repr__(self):
        return repr(self.expression)

    def copy(self):
        out = SpatialExpression(self.ps, self.grid)
        out._expression = self._expression
        out._inputs = self._inputs
        if hasattr(self, '_ufunc'):
            out._ufunc = self._ufunc
        return out

    def __getstate__(self):
        state = dict(
            _ps=self.ps,
            _grid=self.grid,
            _expression=self.expression,
            _inputs=self.inputs,
            _ufunc=None
        )
        return state

    def __setstate__(self, state):
        for k,v in state.items():
            setattr(self, k, v)
