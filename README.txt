KSFD is a package for fintie difference solutions of a version of the
Keller-Segel PDEs. The guts of the code are in the package KSFD, which
lives in the subdirectory KSFD. In addition, this root directory
contains some script meant ot be used by the user. The principle one
is ksfdsolver2.py. KSFD is designed to work efficiently in parallel,
with MPI. A typical workflow:

mkdir solutions
mkdir checks
mpiexec -n 16 python ksfdsolver2.py @options84
python tsmerge.py solutions/options83s16@ -o solutions/options84s16
csh movie2.csh 84s16

Explanation: ksfdsolver2.py is entreily controlled by commandline
options. Sicne these are typicalyl long and complex, I write them in a
text file, which is loaded by indirection (e.g. '@options84'). The
command above runs ksfdsolver2.py in 16 parallele processes. To run
this on computecanada's graham cluster, I actually used

sbatch options84s16.sh

The shell script options84s16.sh is included. 

Now, unfortunately, the version of h5py I was able to install was not
built for MPI. Thus, when ksfdsolver3.py ran, it stores the results
in 16 HDF5 files called solutions/options84as16r0.h5,
solutions/options84as16r1.h5, ..., solutions/options84as16r16.h5. tThe
tsmerge.py commandline combines all these results into a single HDF5
file called solutions/options84s16s1r0.h5. Finally, the csh script
movie2.csh (which yo uwill probably have to edit to fit you
environment and preferences) uses moviemaker2.py to create a series of
images from the merged solution and ffmpeg to combine them into a
movie. 

Although I have not written a systematic reference, most of the
submodules and major functions have detailed docstrings.
