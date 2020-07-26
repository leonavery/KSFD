#!/usr/bin/env python3
"""
lastcheckpoint is a simple kludge to identify and print out the name
of the last checkpoint TimeSeries produced by a previous run of
ksfdsolver2.py. Typical usage is 

    export LASTCHECK=`python lastcheckpoint.py checks/checks115/options115e`

The argument (checks/checks115/options115e in the example) should be
the value of the --check options to ksfdsolver.

lastcheckpoint looks up all files whose names match the regular
expression <prefix>_[0-9]*_s[0-9]r[0-9].h5. It chooses
the maximum checkpoint number (the checkpoint number is the first
[0-9]* in the above regular expression). Calling this <mcn>, the
corresponding TimeSeries prefix is <prefix>_<mcn>_. This is written to
stdout. Alternatively, the -g/--gather options tells last checkpoint
to write <preix>_<mcn>_s<rank>@, where <rank> is the number that
follows s in the regular expression. If rank has multiple values in
the last checkpoint filenames, the largest numbr is used.
"""
import sys, glob, os, re
from argparse import Namespace
from KSFD import Parser
import numpy as np

def parse_commandline(args=None):
    # clargs = Namespace()
    parser = Parser(description='Find last KSFD solution checkpoint')
    parser.add_argument('-g', '--gather', action='store_true')
    parser.add_argument('-v', '--verbose', action='count')
    parser.add_argument('prefix', nargs=1, help='checkpoint prefix')
    clargs = parser.parse_args(args=args)
    return clargs

def main():
    clargs = parse_commandline()
    prefix = clargs.prefix[0]
    fnames = glob.glob(prefix + '_*_s*r*.h5')
    plen = len(prefix)
    fends = [
        fname[plen:] for fname in fnames if fname.startswith(prefix)
    ]
    cpre = re.compile(r'_([0-9]*)_s([0-9]*)r([0-9]*)\.h5')
    csr = np.array([
        list(map(int, cpre.fullmatch(fend).groups()))
        for fend in fends if cpre.fullmatch(fend)
    ], dtype=int)
    mcn = np.max(csr[:,0])
    checkpoint = prefix + '_' + str(mcn) + '_'
    mcn_fends = csr[csr[:,0] == mcn]
    size = np.max(mcn_fends[:,1])
    if clargs.gather:
        print(checkpoint + 's' + str(size) + '@')
    else:
        print(checkpoint)
        
    
if __name__ == '__main__':
    main()
