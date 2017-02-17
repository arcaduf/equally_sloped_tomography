from __future__ import division , print_function
import os


os.chdir( '../algorithm/' )


##  TEST 1
command = 'python run_est.py -Di ../data/ -i shepp_logan_pix0256_ang0100_pseudo.DMP -Do ../data/ -o shepp_logan_pix0256_ang0100_pseudo_est_reco.DMP -g 1 -r 2 -p'
print( '\nTEST 1:\nIterative EST reconstruction of sinogram with projections at pseudopolar angles\n' )
print( '\n', command , '\n' )
os.system( command )


##  TEST 2
command = 'python run_est.py -Di ../data/ -i shepp_logan_pix0256_ang0100_equi.DMP -Do ../data/ -o shepp_logan_pix0256_ang0100_equi_est_reco.DMP -g 0 -r 2 -p'
print( '\nTEST 2:\nIterative EST reconstruction of sinogram with projections at equally angularly spaced angles\n' )
print( '\n', command , '\n' )
os.system( command )


