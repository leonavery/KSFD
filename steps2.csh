#!/bin/tcsh -vx
#
set sol=$1
set images=images/images$sol
set prefix=solutions/options$sol
set iname=$images/options$sol
set moviemaker=moviemaker1.py
set subspaces='0'
set ssnames = 'worms'
set vmax=30000
set width=10.0
set height=10.0
set dpi=150

#
# The following depends on the user having an alias definition file
# called .aliases in their home directory that defines an alias 'ksfd'
# that sets up the correct virtualenv.
#
source $HOME/.aliases
ksfd
set python=`which python`

mkdir -p $images
rm $images/*step*.png
gtime -v $python $moviemaker -v -w $width -t $height --subspace=$subspaces --steps --names=$ssnames -p $prefix "$iname"

