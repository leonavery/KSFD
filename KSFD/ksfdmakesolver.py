"""Factory function for making KSDGSolver, handles multiligand and periodic."""

try:
    from .ksfdexception import KSFDException
except ImportError:
    from ksfdexception import KSFDException

#
# Really just a stub now. May eventually become useful if there are
# multiple Solver types.
#
def makeKSFDSolver(*args, **kwargs):
    return(KSFDSolverPeriodic(*args, **kwargs))
            
# from .ksfdsolver import KSFDSolver
