from __future__ import division , print_function
import sys
import myEST as est
import numpy as np



## Test 1: create pseudo polar views 
def test1():
    n = 16
    nh = int( n * 0.5 )
    nq = int( n * 0.25 )
    nq3 = int( n * 0.75 )
    angles , alphas , indeces = est.create_est_views( n )

    tg_arr = np.tan( angles[:nq+1] )
    tg_diff1 = tg_arr[1:] - tg_arr[:nq]

    tg_arr = 1.0 / np.tan( angles[nq:nq3] )
    tg_diff2 = tg_arr[1:] - tg_arr[:nh-1]

    tg_arr = np.tan( angles[nq3:] )
    tg_diff3 = tg_arr[1:] - tg_arr[:nq-1] 

    
    print('\n\nTEST 1: CREATE PSEUDO POLAR VIEWS')
    print('\nn = ', n)
    print('\nangles in degrees:\n', angles * 180.0 / np.pi)
    print('\ntangent difference 1:\n', tg_diff1)
    print('\ntangent difference 2:\n', tg_diff2)
    print('\ntangent difference 3:\n', tg_diff3)   
    print('\nalphas:\n', alphas)
    print('\nindeces:\n', indeces)




## Test 2: centered fractional fourier transform 
def test2():
    array = np.array([ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ])
    alpha = 0.000713
    frft = est.frft_ctr( array , alpha )
    
    print('\n\nTEST 2: FRACTIONAL FOURIER TRANSFORM CENTERED')
    print('\narray:\n', array)
    print('\nalpha: ', alpha)
    print('\nfrft ctr:\n', frft)




## Test 3: pseudo polar fourier transform 
def test3():
    array = np.array([ [ 1 , 2 , 3 , 4 ],
                       [ 5 , 6 , 7 , 8 ],
                       [ 9 , 10 , 11 , 12 ],
                       [ 13 , 14 , 15 , 16 ]
                    ])

    print( array.shape )
    
    ppft = est.ppft( array )
    
    print('\n\nTEST 3: PSEUDO POLAR FOURIER TRANSFORM')
    print('\narray:\n', array)
    print('\nppft:\n', ppft)




## Test 4: pseudo polar fourier transform adjoint 
def test4():
    array = np.array([ [ 1 , 2 , 3 , 4 ],
                       [ 5 , 6 , 7 , 8 ],
                       [ 9 , 10 , 11 , 12 ],
                       [ 13 , 14 , 15 , 16 ]
                    ])

    ppft_adj = est.ppft_adj( array )
    
    print('\n\nTEST 4: PSEUDO POLAR FOURIER TRANSFORM ADJOINT')
    print('\narray:\n', array)
    print('\nppft:\n', ppft_adj)




## Test 5: CG for inverse pseudo polar fourier transform 
def test5():
    array = np.array([ [ 1 , 2 , 3 , 4 ],
                       [ 5 , 6 , 7 , 8 ],
                       [ 9 , 10 , 11 , 12 ],
                       [ 13 , 14 , 15 , 16 ]
                    ])

    ppft = est.ppft( array )
    ippft = est.ippft( ppft )
    
    print('\n\nTEST 5: CG FOR INVERSE PSEUDO POLAR FOURIER TRANSFORM')
    print('\narray:\n', array)
    print('\nppft:\n', ppft)
    print('\nippft:\n', ippft)




##  Test 6: calculate fourier slice
def test6():
    array = np.array([ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ])
    alpha = 0.987

    fslice = est.calc_est_slice( array , alpha )

    print('\n\nTEST 6: CALCULATE EST FOURIER SLICE')
    print('\narray:\n', array)
    print('\nalpha: ', alpha)
    print('\nfslice:\n', fslice)




##  Test 7: fill pseudo polar grid
def test7():
    array = np.array([ [ 1 , 2 , 3 , 4],
                       [ 5 , 6 , 7 , 8 ],
                       [ 9 , 10 , 11 , 12 ],
                       [ 13 , 14 , 15 , 16 ] ])
    
    angles = np.array([ 0 , 45.0 , 90.0 , 135.0 ])

    print('\n\nTEST 7: FILL PSEUDO POLAR GRID')
    print('\narray:\n', array)
    print('\nangles:\n', angles)

    ppgrid = est.fill_pseudo_polar_grid( array , angles , 4 )

    print('\nppgrid:\n', ppgrid)




##  Test 8: apply physical constraints
def test8():
    array = np.array([ [ 1 , -2 , 3 , 4],
                       [ 5 , 6 , 7 , 8 ],
                       [ 9 , 10 , 11 , -12 ],
                       [ 13 , -14 , 15 , 16 ] ] , dtype=np.float32)

    print('\n\nTEST 7: FILL PSEUDO POLAR GRID')
    print('\narray:\n', array)

    array = est.phys_constraints( array )

    print('\narray with constraints:\n', array) 




def main():
    ##  Run test 1
    if sys.argv[1] == '1':
        test1()

    ##  Run test 2  
    elif sys.argv[1] == '2':
        test2()

    ##  Run test 3  
    elif sys.argv[1] == '3':
        test3()

    ##  Run test 4  
    elif sys.argv[1] == '4':
        test4()

    ##  Run test 5  
    elif sys.argv[1] == '5':
        test5()

    ##  Run test 6  
    elif sys.argv[1] == '6':
        test6()

    ##  Run test 6  
    elif sys.argv[1] == '6':
        test6()

    ##  Run test 7  
    elif sys.argv[1] == '7':
        test7()

    ##  Run test 8  
    elif sys.argv[1] == '8':
        test8() 


    print('\n')



if __name__ == '__main__':
    main()

