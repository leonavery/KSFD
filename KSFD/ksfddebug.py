import os
from mpi4py import MPI

def log(*args, system = 'KSFD', **kwargs):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    systems = set(os.getenv('KSFDDEBUG', default='').split(':'))
    if system in systems or 'ALL' in systems:
        print('{system}, rank={rank}:'.format(system=system, rank=rank), *args, flush=True, **kwargs)
