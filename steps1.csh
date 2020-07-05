#!/bin/tcsh -vx
#
set sol=$1
set images=images/images"$sol"steps
set prefix=solutions/options$sol
set iname=$images/options$sol
set moviemaker=moviemaker1.py
set ssnames = 'worms,attractant,repellent'
set python=`which python`

mkdir -p $images
rm $images/*step*.png
gtime -v gnice -n 19 $python $moviemaker -v --steps --names=$ssnames -p $prefix "$iname"

bell
