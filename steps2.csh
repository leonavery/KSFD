#!/bin/tcsh -vx
#
set sol=$1
set images=images/images$sol
set prefix=solutions/options$sol
set iname=$images/options$sol
set moviemaker=moviemaker1.py
set subspaces='0'
set ssnames = 'worms'
set python=`which python`
set width=10.0
set height=10.0
set dpi=150

mkdir -p $images
rm $images/*step*.png
gtime -v $python $moviemaker -v -w $width -t $height --subspace=$subspaces --steps --names=$ssnames -p $prefix "$iname"

