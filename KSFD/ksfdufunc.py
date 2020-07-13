"""
Subclass UfuncifyCodeWrapper to correctly return multiple results
"""
from __future__ import print_function, division

import sys
import os
import sympy as sy
import shutil
import uuid, re
import tempfile
from subprocess import STDOUT, CalledProcessError, check_output
from string import Template
from warnings import warn
from mpi4py import MPI
try:
    from .ksfddebug import log
    # from .ksfdsym import safe_sympify
except ImportError:
    from ksfddebug import log
    # from ksfdsym import safe_sympify

def logUFUNC(*args, **kwargs):
    log(*args, system='UFUNC', **kwargs)

def UFUNC_MAXARGS():
    return 32

from dogpile.cache import make_region
if 'AUTOWRAP_SCRATCH' in os.environ:
    cachefilename = os.path.join(
        os.environ['AUTOWRAP_SCRATCH'],
        'ufunc_cache.dbm'
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

from sympy.core.cache import cacheit
try:                            # sympy 1.5 needed this: 1.6 barfs on it
    from sympy.core.compatibility import range, iterable
except ImportError:
    pass
from sympy.core.function import Lambda
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy, Symbol
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.utilities.codegen import (make_routine, get_code_generator,
                                     OutputArgument, InOutArgument,
                                     InputArgument, CodeGenArgumentListError,
                                     Result, ResultBase, C99CodeGen)
from sympy.utilities.lambdify import implemented_function
from sympy.utilities.decorator import doctest_depends_on
import sympy.utilities.autowrap
from sympy.utilities.autowrap import *

_doctest_depends_on = {'exe': ('f2py', 'gfortran', 'gcc'),
                       'modules': ('numpy',)}


#################################################################
#                           UFUNCIFY                            #
#################################################################

#
# The following are copied from autowrap.py.
#
from sympy.utilities.autowrap import (_ufunc_top, _ufunc_outcalls,
                                      _ufunc_body, _ufunc_bottom,
                                      _ufunc_init_form, _ufunc_setup)
# _ufunc_top = sympy.utilities.autowrap._ufunc_top
# _ufunc_outcalls = sympy.utilities.autowrap._ufunc_outcalls
# _ufunc_body = sympy.utilities.autowrap._ufunc_body
# _ufunc_bottom = sympy.utilities.autowrap._ufunc_bottom
# _ufunc_init_form = sympy.utilities.autowrap._ufunc_init_form
# _ufunc_setup = sympy.utilities.autowrap._ufunc_setup

_ufuncMultiple_outcalls = Template("${funcname}(${call_args});")            


class UfuncifyCodeWrapperMultiple(UfuncifyCodeWrapper):
    """Wrapper for Ufuncify returning multiple results"""

    def __init__(self, *args, comm=MPI.COMM_WORLD, **kwargs):

        super(UfuncifyCodeWrapperMultiple, self).__init__(*args, **kwargs)
        self.comm = comm
        #
        # Want to be able to cache these functions for long
        # times. Thus we need unique names for functions, modules, and
        # directories. The following is used to build these name.
        #
        idstr = str(uuid.uuid4())
        idstr = 'id_' + re.sub(r'\W', '_', idstr)
        self._uuid = idstr

    #
    # Ovverride these CodeWrapper methods to provide unique names
    @property
    def filename(self):
        return '{filename}_{id}_{count}'.format(
            filename=self._filename,
            id=self._uuid,
            count=CodeWrapper._module_counter
        )

    @property
    def module_name(self):
        return '{filename}_{id}_{count}'.format(
            filename=self._module_basename,
            id=self._uuid,
            count=CodeWrapper._module_counter
        )

    def dump_c(self, routines, f, prefix, funcname=None):
        """Write a C file with python wrappers

        This file contains all the definitions of the routines in c code.

        Arguments
        ---------
        routines
            List of Routine instances
        f
            File-like object to write the file to
        prefix
            The filename prefix, used to name the imported module.
        funcname
            Name of the main function to be returned.
        """
        if len(routines) != 1:
                msg = 'only one routine allowed in UfuncifyCodeWrapperMultiple'
                raise ValueError(msg)
        functions = []
        function_creation = []
        ufunc_init = []
        module = self.module_name
        include_file = "\"{0}.h\"".format(prefix)
        top = _ufunc_top.substitute(include_file=include_file, module=module)

        name = funcname

        # Partition the C function arguments into categories
        # Here we assume all routines accept the same arguments
        r_index = 0
        py_in, py_out = self._partition_args(routines[0].arguments)
        n_in = len(py_in)
        n_out = len(py_out)

        # Declare Args
        form = "char *{0}{1} = args[{2}];"
        arg_decs = [form.format('in', i, i) for i in range(n_in)]
        arg_decs.extend([form.format('out', i, i+n_in) for i in range(n_out)])
        declare_args = '\n    '.join(arg_decs)

        # Declare Steps
        form = "npy_intp {0}{1}_step = steps[{2}];"
        step_decs = [form.format('in', i, i) for i in range(n_in)]
        step_decs.extend([form.format('out', i, i+n_in) for i in range(n_out)])
        declare_steps = '\n    '.join(step_decs)

        # Call Args
        form = "*(double *)in{0}"
        call_args = ', '.join([form.format(a) for a in range(n_in)])
        form = "(double *)out{0}"
        call_args = (
            call_args + ', ' +
            ', '.join([form.format(a) for a in range(n_out)])
        )

        # Step Increments
        form = "{0}{1} += {0}{1}_step;"
        step_incs = [form.format('in', i) for i in range(n_in)]
        step_incs.extend([form.format('out', i, i) for i in range(n_out)])
        step_increments = '\n        '.join(step_incs)

        # Types
        n_types = n_in + n_out
        types = "{" + ', '.join(["NPY_DOUBLE"]*n_types) + "};"

        # Docstring
        docstring = '"Created in SymPy with Ufuncify"'

        # Function Creation
        function_creation.append("PyObject *ufunc{0};".format(r_index))

        # Ufunc initialization
        init_form = _ufunc_init_form.substitute(module=module,
                                                funcname=name,
                                                docstring=docstring,
                                                n_in=n_in, n_out=n_out,
                                                ind=r_index)
        ufunc_init.append(init_form)

        outcalls = [_ufuncMultiple_outcalls.substitute(
            call_args=call_args, funcname=routines[0].name)]

        body = _ufunc_body.substitute(module=module, funcname=name,
                                      declare_args=declare_args,
                                      declare_steps=declare_steps,
                                      call_args=call_args,
                                      step_increments=step_increments,
                                      n_types=n_types, types=types,
                                      outcalls='\n        '.join(outcalls))
        functions.append(body)

        body = '\n\n'.join(functions)
        ufunc_init = '\n    '.join(ufunc_init)
        function_creation = '\n    '.join(function_creation)
        bottom = _ufunc_bottom.substitute(module=module,
                                          ufunc_init=ufunc_init,
                                          function_creation=function_creation)
        text = [top, body, bottom]
        f.write('\n\n'.join(text))

    def wrap_code(self, routines, helpers=None):
        """
        MPI-aware override of the parent function.

        This version compiles the module in rank 0 only, then
        broadcasts import info to other processes.

        If the environment variable 'AUTOWRAP_SCRATCH' is defined,
        each module is buitls in a new ly created directory named
        $AUTOWRAP_SCRATCH/<uuid>, where <uuid> is a universally unique
        ID created with the uuid module. All information needed to
        import the ufunc is cached, so that subsequently it doesn't
        need to be rebuilt.

        If AUTOWRAP_SCRATCH is not defined, the ufunc is built in a
        directory autowrap/<uuid> in the current directory, and is not
        cached. (Really, if would make more sense to use tempdir, and
        perhaps I will do that at some point, but it is a bit
        problematic with MPI.)
        """
        if self.filepath:
            warn('filepath {filepath} ignored'.format(
                filepath=self.filepath
            ))
        helpers = helpers or []
        workdir = module_name = funcname = None
        if self.comm.rank == 0:
            (workdir, module_name, funcname) = self.create_module(
                routines, helpers
            )
        workdir = self.comm.bcast(workdir, root=0)
        module_name = self.comm.bcast(module_name, root=0)
        funcname = self.comm.bcast(funcname, root=0)
        logUFUNC('(workdir, module_name, funcname) sent/rcvd',
                (workdir, module_name, funcname))
        sys.path.append(workdir)
        mod = __import__(module_name)
        sys.path.remove(workdir)
        return self._get_wrapped_function(mod, funcname)

    @cache_region.cache_on_arguments()
    def create_module(
            self,
            routines,
            helpers
    ):
        """
        Create and write ufunc.
        
        Creates a module for the ufunc defined by routines and
        helpers. Returns tuple of strs (workdir, module_name, funcname)
        that can be used to import the ufunc.
        """
        if 'AUTOWRAP_SCRATCH' in os.environ:
            workdir = os.path.join(
                os.environ['AUTOWRAP_SCRATCH'],
                self._uuid
            )
        else:
            workdir = os.path.join(
                os.curdir,
                'autowrap',
                self._uuid
            )
        workdir = os.path.realpath(workdir)
        logUFUNC('creating workdir', workdir)
        os.makedirs(workdir, exist_ok=True)
        funcname = 'wrapped_{id}_{count}'.format(
            id=self._uuid,
            count=CodeWrapper._module_counter
        )
        logUFUNC('funcname', funcname)
        module_name = self.module_name
        logUFUNC('module_name', module_name)
        oldwork = os.getcwd()
        try:
            os.chdir(workdir)
            sys.path.append(workdir)
            self._generate_code(routines, helpers)
            self._prepare_files(routines, funcname)
            self._process_files(routines)
        finally:
            CodeWrapper._module_counter += 1
            os.chdir(oldwork)
        return (workdir, module_name, funcname)

def ufuncify(
    args,
    expr,
    verbose=False,
    helpers=None,
    name=None
):
    """
    Convenient wrapper to ufuncify an expression.

    Required arguments:
    args: list of the input arguments of the function to be built, as
        sympy symbols. 
    expr: The expression to be evaluated. This may be either a single
        sympy expression, in which case a single-output ufunc will be
        built. You may also supply a list of sympy expressions, or any
        other iterator yielding sympy expressions. In this case the
        ufunc will have multiple outputs, in the order given in the
        list. The expressions can be strs, which will be sympified.

    Optional keyword arguments. 
    verbose=False
    helpers=None

    See documentation for sympy.utilities.autowrap.ufuncify.
    """
    args = list(args)           # make sure it's a list
    if name is None:
        name = str(uuid.uuid4())
        name = 'id_' + re.sub(r'\W', '_', name)
    if isinstance(expr, sy.Basic) or isinstance(expr, str):
        exprs = [ expr ]
    else:
        exprs = list(expr)
    for n,e in enumerate(exprs):
        if type(e) is str:
            exprs[n] = sy.sympify(e)
    for e in exprs:
        if not e.free_symbols.issubset(args):
            raise KSFDEXception(
                'unknown symbols {syms} in expression {expr}'.format(
                    syms = e.free_symbols.difference(set(args)),
                    expr = e
                )
            )
    outs = [
        sy.Symbol('_out{n}'.format(n=n)) for n in range(len(exprs))
    ]
    eqs = [
        sy.Eq(out, e) for out,e in zip(outs, exprs)
    ]
    alist = args + outs
    routine = make_routine(
        name=name,
        expr=eqs,
        argument_sequence=alist,
        language='C99'
    )
    wrapper = UfuncifyCodeWrapperMultiple(
        comm=MPI.COMM_WORLD,
        generator=C99CodeGen('ufuncify'),
        flags=[],
        verbose=verbose
    )
    ufunc = wrapper.wrap_code([routine])
    return ufunc

