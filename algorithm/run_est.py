#################################################################################
#################################################################################
#################################################################################
#######                                                                   #######
#######                    RUN EQUALLY SLOPED TOMOGRAPHY                  #######
#######                                                                   #######
#######        Author: Filippo Arcadu, arcusfil@gmail.com, 01/07/2014     #######
#######                                                                   #######
#################################################################################
#################################################################################
#################################################################################




####  PYTHON MODULES
from __future__ import division,print_function
import time
import datetime
import argparse
import sys
import os
import numpy as np




####  MY PYTHON MODULES
sys.path.append( '../common/' )
import my_image_io as io
import my_image_display as dis
import my_image_process as proc




####  MY EST LIBRARY
import pyest




####  MY FORMAT VARIABLES
myfloat = np.float32




##########################################################
##########################################################
####                                                  ####
####             GET INPUT ARGUMENTS                  ####
####                                                  ####
##########################################################
##########################################################

def getArgs():
    parser = argparse.ArgumentParser(description='Run Equally Sloped Tomography',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-Di', '--pathin', dest='pathin', default='./',
                        help='Specify path to input data')    
    
    parser.add_argument('-i', '--sino', dest='sino',
                        help='Specify name of input sinogram')
    
    parser.add_argument('-Do', '--pathout', dest='pathout',
                        help='Specify path to output data') 
    
    parser.add_argument('-o', '--reco', dest='reco',
                        help='Specify name of output reconstruction')
    
    parser.add_argument('-g', '--geometry', dest='geometry',default='0',
                        help='Specify projection geometry;'
                             +' -g 0 --> equiangular views in [0,180)'
                             +' -g 1 --> equally sloped views in [0,180)'
                             +' -g angles.txt --> a list of angles (in degrees)')
    
    parser.add_argument('-c', '--center' , dest='ctr', type=myfloat,
                        help='Centre of rotation (default: center of the image);'
                              + ' -1 ---> search for the center of rotation')
    
    parser.add_argument('-p',dest='plot', action='store_true',
                        help='Display check-plots during the run of the code')

    parser.add_argument('-r',dest='reco_proc', type=np.int, default=1,
                        help='-r 1  --->  PCG-IPPFT; -r 2  --->  Iterative with constraints')    
    
    args = parser.parse_args()
    
    ##  Exit of the program in case the compulsory arguments, 
    ##  are not specified
    if args.sino is None:
        parser.print_help()
        print('ERROR: Input sinogram name not specified!')
        sys.exit()  
    
    return args




##########################################################
##########################################################
####                                                  ####
####                SAVE RECONSTRUCTION               ####
####                                                  ####
##########################################################
##########################################################

def saveReco( reco , pathin , args ):
    ##  Get output directory
    if args.pathout is None:
        pathout = pathin
    else:
        pathout = args.pathout

    if os.path.exists( pathout ) is False:
        print('\nOutput directory ', pathout,' does not exist  --->  created!')
        os.makedirs( pathout )

    print('\nOutput directory:\n', pathout)  


    ##  Save output file
    filename = args.sino
    filename = filename[:len(filename)-4]
    filename += '_est_rec.DMP'
    filename = pathout + filename
    io.writeImage( filename , reco )




##########################################################
##########################################################
####                                                  ####
####                       MAIN                       ####
####                                                  ####
##########################################################
##########################################################

def main():
    ##  Initial print
    print('\n')
    print('########################################################')
    print('#############   EQUALLY SLOPED TOMOGRAPHY  #############')
    print('########################################################')
    print('\n')


    ##  Get the startimg time of the reconstruction
    time1 = time.time()



    ##  Get input arguments
    args = getArgs()


    
    ##  Get input directory
    pathin = args.pathin
    
    if pathin[len(pathin)-1] != '/':
        pathin += '/'

    if os.path.exists( pathin ) is False:
        sys.exit('\nERROR: input directory ', pathin,' does not exist!')

    print('\nInput directory:\n', pathin)



    ##  Get input sinogram
    sinofile = pathin + args.sino
    sino = io.readImage( sinofile )
    nang, npix = sino.shape

    print('\nSinogram to reconstruct:\n', sinofile)
    print('Number of projection angles: ', nang)
    print('Number of pixels: ', npix)



    ##  Display sinogram
    if args.plot is True:
        dis.plot( sino , 'Input sinogram' )



    ##  Getting projection geometry  
    ##  Case of equiangular projections distributed in [0,180)
    if args.geometry == '0':
        print('\nDealing with equiangular views distributed in [0,180)')
        angles = np.arange( nang ).astype( myfloat )
        angles[:] = ( angles * 180.0 )/myfloat( nang )

    ##  Case of pseudo polar views
    elif args.geometry == '1':
        print('\nDealing with equally sloped views in [0,180)')
        angles , dump1 , dum2 = pyest.create_est_views( nang )
        angles *= 180.0 / np.pi

    ##  Case of list of projection angles in degrees
    else:
        geometryfile = pathin + args.geometry
        print('\nReading list of projection angles: ', geometryfile)
        angles = np.fromfile( geometryfile , sep="\t" )

    print('\nProjection angles:\n', angles)



    ##  Set center of rotation axis
    if args.ctr == None:
        ctr = 0.0
        print('\nCenter of rotation axis placed at pixel: ', npix * 0.5)  
    elif args.ctr == -1:
        ctr = proc.searchCtrRot( sino , None , 'a' )
        print('\nCenter of rotation axis placed at pixel: ', ctr)
        sino = proc.sinoRotAxisCorrect( sino , ctr )
    else:
        ctr = args.ctr
        print('\nCenter of rotation axis placed at pixel: ', ctr)
        sino = proc.sinoRotAxisCorrect( sino , ctr ) 



    ##  Get inverse procedure
    if args.reco_proc == 1:
        proc = args.reco_proc
        print('\nSelected inverse procedure: PCG-IPPFT')

    elif args.reco_proc == 2:
        proc = args.reco_proc
        print('\nSelected inverse procedure: iterative procedure with constraints')   



    ##  Reconstruction with EQUALLY SLOPED TOMOGRAPHY
    print('\nPerforming EST reconstruction ....')
    time_rec1 = time.time()
    reco = pyest.est_tomo( sino , angles , proc )
    time_rec2 = time.time()
    print('\n.... reconstruction done!')



    ##  Display reconstruction    
    dis.plot( reco , 'Reconstruction' )


    
    ##  Save reconstruction
    saveReco( reco , pathin , args )

    
    
    ##  Time elapsed for the reconstruction
    time2 = time.time()
    print('\nTime elapsed for the back-projection: ', time_rec2-time_rec1 )
    print('Total time elapsed: ', time2-time1 )


    
    print('\n')
    print('##############################################')
    print('####   EQUALLY SLOPED TOMOGRAPHY DONE !   ####')
    print('##############################################')
    print('\n')




##########################################################
##########################################################
####                                                  ####
####               CALL TO MAIN                       ####
####                                                  ####
##########################################################
##########################################################  

if __name__ == '__main__':
    main()
