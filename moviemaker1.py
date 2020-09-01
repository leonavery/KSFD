#!/usr/bin/env python3

import sys
import os
import datetime
import petsc4py
from warnings import warn
from argparse import Namespace
from KSFD import (KSFDException, Parser, Solution)
import numpy as np
import matplotlib.pyplot as plt

def parse(args=sys.argv[1:]):
    parser = Parser(description='Create movie frames from a time series')
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
    parser.add_argument('-w', '--width', type=float, default=0.0,
                        help='image width (default based on # subspaces)')
    #
    # Abbreviation -h would conflict with --help
    #
    parser.add_argument('-t', '--height', type=float, default=5.0,
                        help='image height')
    parser.add_argument('--vmax', type=float, default=None,
                        help='max value plotted')
    parser.add_argument('--vmin', type=float, default=None,
                        help='min value plotted')
    parser.add_argument('-d', '--dpi', type=int, default=150,
                        help='image height')
    parser.add_argument('-v', '--verbose', action='count')
    parser.add_argument('-c', '--nocolorbar', action='store_true',
                        help="don't plot colorbars")
    parser.add_argument('--names', type=str,
                        help='comma-separated subspace names (for labeling plots)')
    parser.add_argument('--label', type=str, default='t',
                        help='parameter with which to label plots')
    parser.add_argument('--format_time', type=str, default='t',
                        help='format a time label')
    parser.add_argument('-ss', '--subspace', action='append',
                        default=[], help="subspaces to plot")
    parser.add_argument('frameprefix', help='prefix for frame images')
    clargs = parser.parse_args(args)
    return clargs

defplotopts=dict(
    colorbar=True,
    subspaces=[0, 1],
    label='t',
    tformat='t',
)

def plot_curves(t, soln, opts=defplotopts):
    dim = soln.grid.dim
    zmin = ymin = xmin = 0.0
    xmax = soln.grid.bounds[0]
    if dim > 1:
        ymax = soln.grid.bounds[1]
    if dim > 2:
        zmax = soln.grid.bounds[2]
    coords = soln.grid.coordsNoGhosts
    nplots = len(opts['subspaces'])
    names = opts['names']
    images = soln.images(t)
    height = opts['height']
    if opts['width'] > 0.0:
        width = opts['width']
    else:
        width = 4.0*nplots + 2.0*(nplots-1)
    fig = plt.figure(1, figsize=(width,height), dpi=opts['dpi'])
    currplot = 1
    fig.clf()
    params=soln.ps.values(t)
    try:
        labelval = params[opts['label']]
    except KeyError:
        labelval = t
    if opts['label'] == opts['tformat']:
        ti = datetime.timedelta(seconds=int(np.round(labelval)))
        label= opts['label'] + ' = ' + str(ti)
    else:
        label = '%s = %.4g'%(opts['label'], labelval)
    for name,subspace in zip(names, opts['subspaces']):
        title = "%s\n%s"%(name, label)
        ra = fig.add_subplot(1, nplots, currplot, label=title)
        fmin, fmax = (
            np.min(images[subspace]),
            np.max(images[subspace])
        )
        if opts['vmin'] is not None:
            vmin = max(fmin, opts['vmin'])
        else:
            vmin = fmin
        if opts['vmax'] is not None:
            vmax = min(fmax, opts['vmax'])
        else:
            vmax = fmax
        if dim == 1:
            clipped = images[subspace]
            if opts['vmin'] is not None:
                clipped = np.maximum(clipped, opts['vmin'])
            if opts['vmax'] is not None:
                clipped = np.minimum(clipped, opts['vmax'])
            p = plt.plot(coords[0], images[subspace])
            plt.title(title)
        elif dim == 2:
            p = plt.imshow(
                np.transpose(images[subspace]),
                extent=(xmin, xmax, ymin, ymax),
                vmin=vmin,
                vmax=vmax,
                origin='lower',
                cmap='viridis',
                interpolation='none'
            )
            plt.title(title)
            if opts['colorbar']:
                plt.colorbar()
        else:
            raise KSFDException("can only plot 1 or 2 dimensions")
        plt.xlabel('(%7g, %7g)'%(fmin,fmax), axes=ra)
        currplot += 1
    return(fig)

def decode_subspace(ss):
    try:
        ret = int(ss)
    except ValueError:
        ret = str(ss)
    return ret

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
    nsubspaces = soln.grid.dof
    subspaces = [ decode_subspace(ss) for ss in clargs.subspace ]
    if subspaces == []:
        subspaces = list(range(nsubspaces))
    names = list(['y'+str(i) for i in subspaces])
    if clargs.names:
        nopt = clargs.names.split(',')
        if len(nopt) < len(names):
            names[:len(nopt)] = nopt
        else:
            names = nopt
    plotopts = dict(
        colorbar=not clargs.nocolorbar,
        subspaces=subspaces,
        names=names,
        label=clargs.label,
        tformat=clargs.format_time,        
        width=clargs.width,
        height=clargs.height,
        dpi=clargs.dpi,
        vmin=clargs.vmin,
        vmax=clargs.vmax,
    )
    for k,t in enumerate(times):
        if t < start:
            continue
        if t > end:
            break
        fig = plot_curves(t, soln, opts=plotopts)
        frame = clargs.frameprefix + '_' + frname + '%05d'%k + '.png'
        if clargs.verbose:
            print('plotting %s %d, t= %7g, %s'%(frname, k, t, frame))
        fig.savefig(frame)
    
if __name__ == "__main__":
    # execute only if run as a script
    main()

