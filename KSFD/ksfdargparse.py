# Note: it is key that this module not import petsc4py. When petsc4py
# is initialized, it snarfs up all the arguments in sys.argv, which,
# if this parser is being used, will probably include non-PETSc
# arguments.
#
import sys
import shlex
from argparse import ArgumentParser, SUPPRESS

default_parameters = [
    ('degree', 3, 'order of finite difference approximations'),
    ('dim', 1, 'spatial dimensions'),
    ('nelements', 8, 'number grid poimnts in each dimension'),
    ('nwidth', 8, 'number grid points in width'),
    ('nheight', 8, 'number grid points in height'),
    ('ndepth', 8, 'number grid points in depth'),
    ('randgridnw', 0, 'random grid with'),
    ('randgridnh', 0, 'random grid height'),
    ('randgridnd', 0, 'random grid depth'),
    ('width', 1.0, 'width of spatial domain'),
    ('height', 1.0, 'height of spatial domain'),
    ('depth', 1.0, 'depth of spatial domain'),
    ('CFL_safety_factor', 0.0, 'CFL upper bound on timestep'),
    ('conserve_worms', '', 'enforce conservation of worms'),
    ('variance_rate', 0.0, 'rate of increase in random rho variance'),
    ('variance_interval', 100.0, 'frequency of increase in random rho variance'),
    ('variance_timing_function', 't/variance_interval', 'when to inject noise'),
    ('Umin', 1e-7, 'minimum allowed value of U'),
    ('rhomin', 1e-7, 'minimum allowed value of rho'),
    ('rhomax', 28000, 'approximate max value of rho'),
    ('cushion', 2000, 'cushion on rho'),
    ('maxscale', 2.0, 'scale of cap potential'),
    ('s2', 5.56e-4, 'random worm movement (sigma^2/2)'),
    ('Nworms', 0.0, 'total number of worms'),
    ('srho0', '90.0', 'standard deviation of rho(0)'),
    ('rho0', '9000.0', 'C++ string function for rho0, added to random rho0'),
    ('U0_1_1', '', 'C++ string function for U0_1_1'),
    ('ngroups', 1, 'number of ligand groups'),
    ('nligands_1', 1, 'number of ligands in group 1'),
    ('alpha_1', 1500.0, 'alpha for ligand group 1'),
    ('beta_1', 5.56e-4, 'beta for ligand group 1'),
    ('s_1_1', 0.01, 's for ligand group 1, ligand 1'),
    ('gamma_1_1', 0.01, 'gamma for ligand group 1, ligand 1'),
    ('D_1_1', 1e-6, 'D for ligand group 1, ligand 1'),
    ('maxsteps', 1000, 'maximum number of time steps'),
    ('t0', 0.0, 'initial time'),
    ('dt', 0.001, 'first time step'),
    ('tmax', 200000, 'time to simulate'),
    ('rtol', 1e-5, 'relative tolerance for step size adaptation'),
    ('atol', 1e-5, 'absolute tolerance for step size adaptation'),
]

class Parser(ArgumentParser):
    """An argument parser for KSFD scripts

    KSFD.Parser is used like argparse.ArgumentParser -- you create one,
    call the add_argument method to add arguments to it, then call the
    parse_args methods to parse the command line. However, it also
    extracts PETSc commandline arguments. The syntax for passing these is:

    program --petsc petscarg1 petscarg2 ...
    program myarg1 ... --petsc petscarg1 petscarg2 ... -- myarg2 myarg3

    The list of PETSc arguments is terminated by a '--' ot its end

    parse_args returns a Namespace object that contains the results of
    parsing your object in the usual way. In addition, it will contain
    the name 'petsc', whose value is a list of strs that can be passed
    to petsc4py.init in order to set defaults for that subsystem.
    """

    subsystems = ['petsc']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, fromfile_prefix_chars='@',
                         allow_abbrev=False, **kwargs)
        #
        # The following is just for the sake of the help
        # message. '--petsc' will be stripped before arguments are
        # passed to the parser, so these options will not usually be
        # activated.
        #
        self.add_argument('--petsc', action='append', default=SUPPRESS,
                          help='PETSc subsystem arguments: \
                                terminate with --, \
                                --petsc -help for help')

    def convert_arg_line_to_args(self, arg_line, comment_char='#'):
        """Override the function in argparse to handle comments"""
        return shlex.split(arg_line, comments=True)
        cpos = arg_line.find(comment_char)
        if cpos >= 0:
            arg = arg_line[:cpos].strip()
        else:
            arg = arg_line.strip()
        if arg:
            return([arg])
        else:
            return([])
    
    def parse_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]
        #
        # ***** warning *****
        # Here I call a private method of ArgumentParser to handle
        # file indirection.
        #
        args = self._read_args_from_files(args)
        sargs = [[]] * len(self.subsystems)
        for s, subsystem in enumerate(self.subsystems):
            while('--' + subsystem in args):
                f = args.index('--' + subsystem)
                try:
                    e = args.index('--', f + 1)
                except ValueError:
                    e = len(args)
                sargs[s] += args[f+1:e]
                args[f:e+1] = []
        ns = super().parse_args(args)
        for s, subsystem in enumerate(self.subsystems):
            setattr(ns, subsystem, sargs[s])
        return ns
        

def main():
    parser1 = Parser('KSFD Parser')
    parser1.add_argument('-1', '--opt1', action='store_true')
    parser1.add_argument('-2', '--opt2')
    print(parser1.parse_args(
        ['--petsc', '-help', '--', '-2', 'filename']
    ))
    print(parser1.parse_args())

if __name__ == '__main__':
    main()
