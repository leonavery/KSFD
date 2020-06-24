#!/usr/bin/env python3
"""
tsmerge does two separate tasks: merging and gathering TimeSeries
files. 

The need to merge arises when a PDE solution is interrupted and
resumed. This results in the creation of two or more TimeSeries
covering (typically) disjoint time intervals. The merge combines the
distinct TimeSeries into a single TimeSeries housed in a single file
that contains all the time points of the merged series.

The need to gather arises when a solution is run in multiple processes
in the MPi environment. When this is done with a build of HDF5 that
doesn't support parallel HDF5, it is necessary to save the results
from each process in distinct files. This form is invonvenient for
analysis, and also for resuming an interrupted process. The gather
combines the results of all the files representing a single TimeSeries
into a single file. 

tsmerge takes two types of commandline arguments: the output file
prefix, specified with the -o or --outfile option, and a list of input
file prefixes. The output is stored in a file named <prefix>s1r0.h5,
where >prefix> is the prefix specified with the --outfile
option. Input prefixes are of two types. For a series stored in a
single file (that would typically be named either <inprefix>s1r0.h5 of
<inprefix>MPI.h5), the commandline argument is just the prefix. If,
however, one of the TimeSeries to be merged is stored in process files
that need to be gathered, there is a special syntax:
<inprefix>s<n>@. For instance, 'bases4@' would be used for a series
with basename base stored in four files bases4r0.h5, bases4r1.h5,
bases4r2.h5, and bases4r3.h5.
"""
import sys
from argparse import ArgumentParser
from KSFD import TimeSeries, Gatherer, KSFDException
import numpy as np
from mpi4py import MPI
import petsc4py

def main():
    comm = MPI.COMM_WORLD
    if comm.size > 1:
        raise KSFDException(
            'tsmerge must be run sequentially.'
        )
    petsc4py.init()
    parser = ArgumentParser(description='Merge time series',
                            allow_abbrev=True)
    parser.add_argument('-o', '--outfile',
                        help='merged file basename')
    parser.add_argument('-s', '--start', type=float, default=0.0,
                        help='start time')
    parser.add_argument('-e', '--end', type=float,
                        help='end time')
    parser.add_argument('infiles', nargs='+', help='files to merge')
    parser.add_argument('-v', '--verbose', action='count')
    clargs = parser.parse_args()
    times = np.empty((0), dtype=float)
    steps = np.empty((0), dtype=int)
    files = np.empty((0), dtype=int)
    grid = None
    for f,name in enumerate(clargs.infiles):
        if clargs.verbose > 0:
            print('collecting times from {name}'.format(name=name),
                  flush=True)
        g = Gatherer(name)
        if grid is None:
            grid = g.grid
        st = g.sorted_times()
        if clargs.verbose > 1:
            print('times:', st, flush=True)
        steps = np.append(steps, g.steps())
        times = np.append(times, st)
        files = np.append(files, np.full_like(st, f, dtype=int))
        g.close()
    out = TimeSeries(clargs.outfile, grid, comm=MPI.COMM_SELF, mode='w')
    # order = times.argsort()
    # files = files[order]
    # steps = steps[order]
    # times = times[order]
    k = 0
    del out.tsf['/info']
    for f,name in enumerate(clargs.infiles):
        if clargs.verbose > 0:
            print('collecting data from {name}'.format(name=name),
                  flush=True)
        fmatch = files == f
        # forder = order[fmatch]
        fsteps = steps[fmatch]
        ftimes = times[fmatch]
        g = Gatherer(name)
        if '/info' not in out.tsf:
            g.tsf.copy(
                source='/info',
                dest=out.tsf,
                name='/info',
                shallow=False
            )
        for s in g:
            if clargs.verbose > 0:
                print(str(s.tsf), flush=True)
            for k,t in zip(fsteps,ftimes):
                if t < clargs.start or (clargs.end and t > clargs.end):
                    continue
                vals = s.retrieve_by_number(k)
                if clargs.verbose > 1:
                    print('point {k}, time {t}'.format(k=k, t=t),
                          flush=True)
                out.store_slice(s.ranges, vals, t)
        g.close()
    out.close()
    
if __name__ == "__main__":
    # execute only if run as a script
    main()

