###########################################################
###########################################################
####                                                   ####
####         EQUALLY SLOPED TOMOGRAPHY LIBRARY         ####
####                                                   ####
####    A. Filippo , arcusfil@gmail.com , 01/07/2014   ####
####                                                   ####
###########################################################
###########################################################




####  PYTHON LIBRARIES
from __future__ import division , print_function
import sys
import numpy as np
import scipy
import scipy.fftpack




####  MY FORMAT VARIABLE
myFloat = np.float32
myComplex = np.complex64 




####  CONSTANTS
TRACER_1 = -2e14    ##  TRACER_1  --->  value tracing the missing samples  
TRACER_2 = -3e14    ##  TRACER_2  --->  value tracing the samples outside
                    ##                  the resolution circle




###########################################################
###########################################################
####                                                   ####
####               CREATE PSEUDO-POLAR ANGLES          ####
####                                                   ####
###########################################################
###########################################################

def create_est_views( n ):
    if n % 4 != 0:
        sys.exit('\n\tError inside create_est_views:'
                 +'\n\t  number of views is not divisible by 4 !\n')
    
    nh = int( n * 0.5 ) 
    nq = int( n * 0.25 )
    nq3 = nq * 3

    pseudo_angles = np.zeros( n , dtype=myFloat )
    pseudo_alphas = np.zeros( n , dtype=myFloat )
    pseudo_indeces = np.zeros( n , dtype=int )
    index = np.arange( nq + 1 , dtype=int )

    pseudo_angles[0:nq+1] = np.arctan( 4 * index[:] / myFloat(n) )
    pseudo_angles[nh:nq:-1] = np.pi/2 - pseudo_angles[0:nq]    
    pseudo_angles[nh+1:] = np.pi - pseudo_angles[nh-1:0:-1]

    pseudo_alphas[0:nq] = 1.0 / np.cos( pseudo_angles[0:nq] )
    pseudo_alphas[nq3:] = 1.0 / np.cos( pseudo_angles[nq:0:-1] )
    pseudo_alphas[nq:nq3] = 1.0 / np.sin( pseudo_angles[nq3:nq:-1] )    

    pseudo_indeces[n-1::-1] = np.arange( n , dtype=int )
    pseudo_indeces[:] = np.roll( pseudo_indeces , - ( nq3 - 1 ) )

    #pseudo_indeces[:nq] = np.arange( nq3 , n , dtype=int )
    #pseudo_indeces[nq:] = np.arange( 0 , nq3 , dtype=int )  
    
    return pseudo_angles , pseudo_alphas , pseudo_indeces




###########################################################
###########################################################
####                                                   ####
####       FRACTIONAL FOURIER TRANSFORM CENTERED       ####
####                                                   ####
###########################################################
###########################################################

def frft_ctr( x , alpha ):
    n = len( x )
    frft = np.empty( n , dtype=myComplex )

    index = np.arange( n )
    im = complex(0,1)
    frft[:] = x * ( np.cos( np.pi*index*alpha ) + im * np.sin( np.pi*index*alpha ) )

    index = np.concatenate( ( index , np.arange(-n,0,1) ) , axis=0 )
    arg = np.pi * alpha * index *index / myFloat( n )
    factor = np.cos( arg ) - im * np.sin( arg )
    
    frft = np.concatenate( ( frft , np.zeros( n , dtype=myComplex ) ) , axis=0 )
    frft[:] = frft * factor
    
    frft[:] = np.fft.ifft( np.fft.fft( frft )* np.fft.fft( np.conjugate( factor ) ) )
    frft[:] = frft * factor
    frft = frft[0:n]
    index = np.arange( -myFloat( n ) * 0.5 , myFloat( n ) * 0.5 , 1 )
    arg = np.pi * index * alpha
    frft[:] = frft * ( np.cos( arg )+ im * np.sin( arg ) )

    return frft




###########################################################
###########################################################
####                                                   ####
####            PSEUDO POLAR FOURIER TRANSFORM         ####
####                                                   ####
###########################################################
###########################################################

def ppft( f ):
    if f.ndim != 2:
        sys.exit('ERROR: input for forwardPseudoPolarFT is not a 2D array!')
    nx,ny = f.shape
    if nx != ny or nx % 2:
        n = int( np.ceil(max(nx,ny)/2.) * 2 )
    else: n = nx
    f_out = np.zeros( ( 2*n , 2*n ) , dtype=myComplex )
    f_aux = np.zeros( ( 2*n , n ) , dtype=myComplex )

    ##  Vertical hourglass
    f_aux[n/2:3*n/2,:] = f[:,:]      
    f_aux[:,:] = np.roll( f_aux , n , axis=0 )      
    f_aux[:,:] = scipy.fftpack.fft( f_aux , 2*n , axis=0 , overwrite_x=True )      
    f_aux[:,:] = scipy.fftpack.fftshift( f_aux , axes=0 )
    for i in range(-n,n):
        f_out[:n,i+n] = frft_ctr( f_aux[i+n,:] , -i/myFloat( n ) )

    ##  Horizontal hourglass
    f_aux[:,:] = 0
    f_aux = f_aux.reshape( n , 2*n )
    f_aux[:,n/2:3*n/2] = f[:,:]
    f_aux[:,:] = np.roll( f_aux , n , axis=1 )
    f_aux[:,:] = 2 * n * scipy.fftpack.ifft( f_aux , 2*n , axis=1 , overwrite_x=True )
    f_aux[:,:] = scipy.fftpack.fftshift( f_aux , axes=1 )  
    for i in range( -n , n ):
        f_out[n:,i+n] = frft_ctr( f_aux[:,i+n] , -i/myFloat(n) )
    f_out[:,:] = 1.0 /( np.sqrt(2) * n ) * f_out[:,:]

    return f_out




###########################################################
###########################################################
####                                                   ####
####        PSEUDO POLAR FOURIER TRANSFORM ADJOINT     ####
####                                                   ####
###########################################################
###########################################################

def ppft_adj( f ):
    if f.ndim != 2:
        sys.error('ERROR: input for adjointPseudoPolarFT is not a 2D array!')
    nx,ny = f.shape
    if nx != ny or nx % 2:
        n = int( np.ceil( max(nx,ny)/2. ) * 2 )
    else: n = nx
    n = int( n/2 )
    f_out = np.zeros( ( n , n ), dtype=myComplex )
    f_aux = np.zeros( ( 2*n , n ) , dtype=myComplex )

    ##  Vertical hourglass 
    f_aux[:,:] = np.transpose( f[:n,:] )
    for i in range( -n , n ):
        f_aux[i+n,:] = frft_ctr( f_aux[i+n,:] , i/myFloat( n ) )
    f_aux[:,:] = np.roll( f_aux , n , axis=0 )
    f_aux[:,:] = 2 * n * scipy.fftpack.ifft( f_aux , 2*n , axis=0 , overwrite_x=True )
    f_aux[:,:] = np.fft.fftshift( f_aux , axes=(0,) )
    f_out[:,:] = f_aux[n/2:3*n/2,:]

    ##  Horizontal hourglass
    f_aux = f_aux.reshape( n , 2*n )  
    f_aux[:,:] = f[n:,:]
    for i in range( -n , n ):
        f_aux[:,i+n] = frft_ctr( f_aux[:,i+n] , i/myFloat( n ) )
    f_aux[:,:] = np.roll( f_aux , n , axis=1 )
    f_aux[:,:] = scipy.fftpack.fft( f_aux , 2*n , axis=1 , overwrite_x=True )
    f_aux[:,:] = scipy.fftpack.fftshift( f_aux , axes=(1,) )

    f_out[:,:] = 1/(n*np.sqrt(2)) * ( f_out[:,:] + f_aux[:,n/2:3*n/2] ) 

    return f_out




###########################################################
###########################################################
####                                                   ####
####         CONJUGATE GRADIENT FOR INVERSE PPFT       ####
####                                                   ####
###########################################################
###########################################################

def ippft( f , niter=6 , eps=1e-8 ):
    ##  Check whether the input Pseudo Polar Grid is a 2D array
    ##  and it is a square matrix
    if f.ndim != 2:
        sys.error('ERROR: input for inversePseudoPolarFT is not a 2D array!') 
    if f.shape[0] != f.shape[1]:
        sys.error('WARNING: input for inversePseudoPolarFT is not a square matrix')

    
    ##  Convert input Pseudo Polar Grid to myComplex format
    f = f.astype( myComplex )

    ##  Get size of the output object
    n = int( 0.5 * f.shape[0] )

    ##  Create preconditioning matrix to decrease the
    ##  conditioning number of the Pseudo Polar Operator 
    M = np.array( np.sqrt( abs( np.arange(-n,n)/myFloat(n) ) ) )
    M[n] = np.sqrt( 1.0/myFloat(4*n) )
    M = np.ones( ( 2*n , 1 ) ) * M;
    M[:,:] = M[:,:]**2;

    ##  Initialization of the preconditioned conjugate method 
    xk = ppft_adj( np.multiply( M , f ) )
    pk = xk - ppft_adj( M * ppft( xk ) )
    rk = np.empty( ( n ,  n ) , dtype=myComplex )
    rk[:,:] = pk[:,:]
    f[:,:] = 0  
    ppAdj = np.zeros( ( n , n ) , dtype=myComplex )

    ##  Print setu-up parameters for the IPPFT 
    print('  PCG-IPPFT: stopping threshold: ' , eps )
    print('  PCG-IPPFT: max number of CG-iterations: ' , niter )
    
    ##  Start conjugate gradient iterations
    it = 0
    while it < niter:
        err = np.linalg.norm( pk )
        print('    Iter = ' , it , '   cost function = ' , err ) 
        if err > eps:
            f[:,:] = ppft( pk )
            f[:,:] = np.multiply( M , f )
            ppAdj[:,:] = ppft_adj( f )
            a0 = np.sum( np.abs(rk) * np.abs(rk) )
            a1 = np.sum( np.conjugate(pk) * ppAdj )
            a = a0 / myComplex( a1 )
            xk[:,:] += a*pk 
            rk[:,:] -= a * ppAdj 
            bb = np.sum( np.abs(rk) * np.abs(rk) )
            b = bb/a0
            pk[:,:] = rk + b * pk
        else: break
        it += 1

    ##  Final informative prints
    print('  PCG-IPPFT: number of CG-iterations used: ' , it )
    print('  PCG-IPPFT: cost function: ' , err )
    print('  .... PCG for inverse PPFT done!')  

    return xk




#############################################################
#############################################################
####                                                     ####
####             CALCULATE EST FOURIER SLICE             ####
####                                                     ####
#############################################################
############################################################# 

def calc_est_slice( proj , alpha ):
    ##  Get number of pixels
    npix = len( proj )
    npixh = int( npix * 0.5 )


    ##  Zero-pad projection
    fslice = np.zeros( 2 * npix , dtype=myComplex )
    fslice[ npixh : 3*npixh ] = proj
    #print('\nfslice inside:\n', fslice)


    ##  Apply centered fractional Fourier transform
    fslice[:] = frft_ctr( fslice , alpha )


    ##  Normalize transform
    fslice[:] *= 1.0 / ( myFloat( npix ) * np.sqrt( 2 ) )


    return fslice




#############################################################
#############################################################
####                                                     ####
#### FILL PSEUDO POLAR GRID WITH FRFT OF THE PROJECTIONS ####
####                                                     ####
#############################################################
############################################################# 

def fill_pseudo_polar_grid( sino , angles , n , eps=None  ):
    ##  Convert angles to radiants
    angles *= np.pi / 180.0


    ##  Get number of projection angles
    nang = len( angles )


    ##  Get initial number of pixels
    npix = n
    npixh = int( npix * 0.5 )

    
    ##  Check if number of pixels is even:
    ##  if yes, remove first pixel
    if npix % 2 == 0:
         sino = sino[:,1:]


    ##  Get number of pixels
    npix_op = sino.shape[1]


    ##  Get number of frequencies
    nfreq = 2 * npix
    nfreqh = int( 0.5 * nfreq )
    nfreqq = int( 0.25 * nfreq )
    nfreqq3 = int( 0.75 * nfreq ) 


    ##  Allocate memory for the Pseudo Polar Fourier grid
    ##  and fill it with -1 values
    ppgrid = np.zeros( ( nfreq , nfreq ) , dtype=myComplex )
    ppgrid[:,:] = TRACER_1


    ##  Allocate memory for the auxiliary array hosting each
    ##  single Fourier slice
    fslice = np.zeros( nfreq , dtype=myComplex )


    ##  Calculate list of pseudo polar angles, the related alpha values the 
    ##  related index to map the projections into the pseudo polar grid
    pseudo_angles , pseudo_alphas , pseudo_indeces = create_est_views( nfreq )


    ##  Threshold to filter out projection whose view is too far
    ##  from the nearest equally sloped view
    if eps is None:
        eps = 2 * np.pi / 180.0


    ##  Loop on the projection angles to find the nearest equally
    ##  sloped view, apply fractional fourier transform centered
    ##  and assign the transformed projection to the right location
    ##  in the pseudo polar grid
    #print('npixh-(npix_op-1)/2 = ', npixh-(npix_op-1)/2 )
    #print('npixh+(npix_op-1)/2 = ', npixh+(npix_op-1)/2 )
    proj_aux = np.zeros( npix , dtype=myFloat )

    for i in range( nang ):
        angle = angles[i]
        angle_diff = np.abs( pseudo_angles - angle ) 
        ind = np.argwhere( angle_diff == np.min( angle_diff ) )

        
        ##  Compute the Fourier slice
        if pseudo_indeces[ind] <= nfreqq:
            proj_aux[ npixh-(npix_op-1)/2 : npixh+(npix_op-1)/2+1 ] = sino[i,:]
            fslice[:] = calc_est_slice( proj_aux , pseudo_alphas[ind] )

        else:
            proj_aux[ npixh-(npix_op-1)/2 : npixh+(npix_op-1)/2+1 ] = sino[i,::-1] 
            fslice[:] = calc_est_slice( proj_aux , pseudo_alphas[ind] )


        ##  Switch off all Fourier samples outside the resolution circle
        radius = int( ( npix - 1 ) * 1.0/pseudo_alphas[ind] )
        fslice[ : nfreqh-radius ] = TRACER_2
        fslice[ nfreqh + radius + 1 : ] = TRACER_2


        ##  Assign Fourier slice to the right location in the grid
        ppgrid[ pseudo_indeces[ind] , : ] = fslice


    return ppgrid




#############################################################
#############################################################
####                                                     ####
####              APPLY PHYSICAL CONSTRAINTS             ####
####                                                     ####
#############################################################
#############################################################

def phys_constraints( image , beta = 0.9 ):
    ind = np.argwhere( image < 0 )
    image[ind[:,0],ind[:,1]] *= beta
    return image




#############################################################
#############################################################
####                                                     ####
####              TOMOGRAPHIC RECONSTRUCTION             ####
####                                                     ####
#############################################################
#############################################################

def est_tomo( sino , angles , proc=1 ):
    npix = sino.shape[1]

    ##  Fill pseudo polar grid
    ppgrid = fill_pseudo_polar_grid( sino , angles , npix )
    nfreq = ppgrid.shape[0]
    npix = int( nfreq * 0.5 )


    
    ##  1) Use PCG-IPPFT to reconstruct the tomogram
    if proc == 1:
        ##  Set missing Fourier samples to zero
        ind = np.argwhere( ( ppgrid == TRACER_1 ) | ( ppgrid == TRACER_2 ) )
        ppgrid[ ind[:,0] , ind[:,1] ] = 0

        
        ##  Apply PCG-IPPFT
        reco = np.real( ippft( ppgrid ) )


    
    ##  2) Use iterative procedure with physical constraints
    elif proc == 2:
        ##  Set missing Fourier samples and the ones outside
        ##  the resolution circle to zero
        ind1 = np.argwhere( ( ppgrid != TRACER_1 ) & ( ppgrid != TRACER_2 ) ) 
        ind2 = np.argwhere( ppgrid == TRACER_1 )
        ind3 = np.argwhere( ppgrid == TRACER_2 )
        ppgrid[ ind2[:,0] , ind2[:,1] ] = 0 
        ppgrid[ ind3[:,0] , ind3[:,1] ] = 0 

        
        ##  Allocate memory for the pseudo polar grid to update
        ppgrid_up = ppgrid.copy()

        
        ##  Allocate memory for the reconstruction
        reco = np.zeros( ( npix , npix ) , dtype=myFloat )  

        
        ##  Set param for macro loop
        niter = 7;  it = 0;  eps = 1e-7;  err = 1e5
        den = np.sum( np.abs( ppgrid[ind1[:,0],ind1[:,1]] ) )

        
        ##  Macro loops
        while it < niter:
            ##  Get reconstruction at iteration it
            reco[:,:] = np.real( ippft( ppgrid_up ) )

            ##  Multiply negative pixels per beta
            reco[:,:] = phys_constraints( reco , beta = 0 )

            ##  Compute new pseudo polar grid
            ppgrid_up[:,:] = ppft( reco )

            ##  Calculate error in the Fourier space
            err = np.sum( np.abs( ppgrid_up[ind1[:,0],ind1[:,1]]
                                   - ppgrid[ind1[:,0],ind1[:,1]] ) ) / den

            if err < eps: break

            ##  Reassign Fourier slice corresponding to the transformed
            ##  sinogram projections with zeros outside the resolution circle
            ppgrid_up[ind1[:,0],ind1[:,1]] = ppgrid[ind1[:,0],ind1[:,1]]

            ##  Zero-out all the samples outside the resolution circle
            ppgrid_up[ind3[:,0],ind3[:,1]] = 0.0

            ##  Increment iteration
            it += 1

            ##  Print
            print('Macro iter. number: ', it,'  Fourier error: ', err)


    ##  Rotate it by 90 degrees
    reco[:,:] = np.rot90( reco  , -1 )    

    return reco
