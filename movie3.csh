#!/bin/tcsh -vx
#
set sol=$1
set images=images/images$sol
set prefix=solutions/options$sol
set iname=$images/options$sol
set movie=movies/options$sol.mp4
set moviemaker=moviemaker1.py
set subspaces='1'
set width=10.0
set height=10.0
set dpi=150
set ssnames = 'worms'
set python=$HOME/KSFD/bin/python

mkdir -p $images
rm $images/*frame*.png
gtime -v gnice -n 19 $python $moviemaker -v -n 1501 --subspace=$subspaces -w $width -t $height -d $dpi -e 10000 --names=$ssnames -p $prefix "$iname"a
gtime -v gnice -n 19 $python $moviemaker -v -n 1501 --subspace=$subspaces -w $width -t $height -d $dpi -s 10000 -e 200000 --names=$ssnames -p $prefix "$iname"b
rm "$iname"b_frame00000.png
ffmpeg -y -framerate 15 -pattern_type glob -i "$iname"'[ab]_frame*.png' -pix_fmt yuv420p -vf scale=w=-2:h=0 $movie

bell
