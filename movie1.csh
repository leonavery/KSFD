#!/bin/tcsh -vx
#
set sol=$1
set images=images/images$sol
set prefix=solutions/options$sol
set iname=$images/options$sol
set movie=movies/options$sol.mp4
set moviemaker=moviemaker1.py
set ssnames = 'worms,attractant,repellent'
set python=`which python`

mkdir -p $images
rm $images/*frame*.png
gtime -v gnice -n 19 $python $moviemaker -v -n 1501 -e 10000 --names=$ssnames -p $prefix "$iname"a
gtime -v gnice -n 19 $python $moviemaker -v -n 1501 -s 10000 -e 200000 --names=$ssnames -p $prefix "$iname"b
rm "$iname"b_frame00000.png
ffmpeg -y -framerate 15 -pattern_type glob -i "$iname"'[ab]_frame*.png' -pix_fmt yuv420p -vf scale=w=-2:h=0 $movie </dev/null

bell
