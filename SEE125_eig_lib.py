import numpy as np


def testcreate(n,val2):
    """ Create a square matrix where 
        each element's value is the square root of the 
        previous elements'value increased by 1. 

        Parameters: 
                n -- the shape of the matrix will be (n,n) (int)
                val2 -- the square root of this number will be the first matrix 
                        element at row 0, column 0 (float, positive or zero)
        Returns: 
                numpy array of size (n,n)

    """
    assert val2>=0 # since we are taking the square root
    A = np.arange(val2,val2+n*n).reshape(n,n)
    return np.sqrt(A)




def power(A,kmax=6):
    """
    function power given in the book (code 4.6 p 180)
    """
    zs = np.ones(A.shape[0])
    qs = zs/np.linalg.norm(zs)
    for k in range(1,kmax):
        zs = A@qs
        qs = zs/np.linalg.norm(zs)
        print(k,qs)
    lam = qs@A@qs 
    return lam, qs # lam=eigenvalue, qs=eigenvector


def mag(xs):
    return np.sqrt(np.sum(xs*xs))


def qrdec(A):
    '''
    QR decomposition of the input matrix (page 191)
    Input:
        Numpy array representing a square matrix A (real-valued)

    Returns:
        Q: orthogonal matrix
        R: upper triangular matrix
    '''
    n = A.shape[0]
    Ap = np.copy(A)
    Q = np.zeros((n,n))
    R = np.zeros((n,n))
    for j in range(n):
        for i in range(j):
            R[i,j] = Q[:,i]@A[:,j]
            Ap[:,j] -= R[i,j]*Q[:,i]

        R[j,j] = mag(Ap[:,j])
        Q[:,j] = Ap[:,j]/R[j,j]
    return Q, R



def qrmet(inA,kmax=100):
    '''
    QR method (page 202)
    '''

    A = np.copy(inA)
    for k in range(1,kmax):
        Q, R = qrdec(A)
        A = R@Q
        #print(k, np.diag(A))
    
    qreigvals = np.diag(A)
    return qreigvals

def powercrit(A, ceps=1e-8, kmax=50): 
    ### BEGIN SOLUTION ###
    ### END SOLUTION ###
    
    zs = np.ones(A.shape[0])#the initial guess
    
    qs = zs/np.linalg.norm(zs)
    # This is the power method with the new convergence criterion    
    for k in range(kmax): # also keep a max of iteration
        zs = A@qs
        qskm1 = qs
        qs = zs/np.linalg.norm(zs)     
        crit = np.sum(np.linalg.norm(qs-qskm1)/np.linalg.norm(qs))
        crit_1 = np.sum(np.linalg.norm(qs+qskm1)/np.linalg.norm(qs))
        print(crit, qs)
        if crit<ceps or crit_1<ceps:
            lam = qs.T@A@qs # same as qs@A@qs, Python will transpose
            print('Convergence reached after {:d} iterations'.format(k+1))
            break
    else:     
        lam = qs = None
        print('Convergence failed')
    
    return lam, qs
