#!/usr/bin/env python3

import sys
import os
import h5py
import sympy as sy
import json
import datetime
import petsc4py
from warnings import warn
from argparse import Namespace
from KSFD import (KSFDException, Parser, Solution)
import numpy as np
import matplotlib.pyplot as plt

def parse(args=sys.argv[1:]):
    parser = Parser(description='Create HDF5 files from time series')
    parser.add_argument('-p', '--prefix',
                        help='solution file prefix')
    parser.add_argument('--steps', action='store_true',
                        help='use actual time steps')
    parser.add_argument('-s', '--start', type=float, default=0.0,
                        help='start time')
    parser.add_argument('-e', '--end', type=float,
                        help='end time')
    parser.add_argument('-n', '--nframes', type=int, default=3001,
                        help='number frames')
    parser.add_argument('-v', '--verbose', action='count')
    parser.add_argument('frameprefix', help='prefix for HDF5 files')
    clargs = parser.parse_args(args)
    return clargs

def main():
    clargs = parse()
    petsc4py.init(clargs.petsc)
    soln = Solution(clargs.prefix)
    tmin, tmax = soln.tmin, soln.tmax
    start = clargs.start
    end = clargs.end if clargs.end else tmax
    n = clargs.nframes
    if clargs.steps:
        frname = 'step'
        times = [ t for t in soln.tstimes if t >= start and t <= end ]
    else:
        frname = 'frame'
        times = np.linspace(start, end, num=n)
    for k,t in enumerate(times):
        if t < start:
            continue
        if t > end:
            break
        images = soln.images(t)
        params=soln.ps.values(t)
        for key,val in params.items():
            if isinstance(val, sy.Float):
                params[key] = float(val)
            elif isinstance(val, sy.Integer):
                params[key] = int(val)
        h5fname = clargs.frameprefix + '_' + frname + '%05d'%k + '.h5'
        if clargs.verbose:
            print('saving %s %d, t= %7g, %s'%(frname, k, t, h5fname))
        h5img = h5py.File(h5fname, 'w')
        h5img['t'] = t
        h5img['images'] = soln.images(t).copy(order='C')
        h5img['params'] = json.dumps(params)
        h5img.close()
    
if __name__ == "__main__":
    # execute only if run as a script
    main()

