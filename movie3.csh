#!/bin/tcsh -vx
#
set sol=$1
set images=images/images$sol
set prefix=solutions/options$sol
set iname=$images/options$sol
set movie=movies/options$sol.mp4
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
rm $images/*frame*.png
gtime -v gnice -n 19 $python $moviemaker -v -n 3001 -e 200000 --vmax=$vmax -w $width -t $height --subspace=$subspaces --names=$ssnames -p $prefix "$iname"
ffmpeg -y -framerate 15 -pattern_type glob -i "$iname"'_frame*.png' -pix_fmt yuv420p -vf scale=w=-2:h=0 $movie </dev/null

bell
