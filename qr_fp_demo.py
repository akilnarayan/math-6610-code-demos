# A demonstration of how floating-point arithmetic makes some algorithms
# unreliable.

import numpy as np
from numpy import dot
from numpy.linalg import norm

def gram_schmidt(A):
    """
    Computes the reduced QR decomposition of an input matrix A via
    classical Gram Schmidt. Assumes that A has full column rank.

    Output: matrices Q and R. If A is m x n, then Q has shape (m,n) and
    R has shape (n,n)
    """

    m,n = A.shape
    assert m >= n, "Input matrix doesn't have full column rank"

    Q = np.zeros([m,n])
    R = np.zeros([n,n])

    Q[:,0] = A[:,0]
    R[0,0] = norm(A[:,0])
    Q[:,0] /= R[0,0]

    for r in range(1, n):
        R[:r,r] = dot(Q[:,:r].T, A[:,r])

        # GS: orthogonalize A[:,r] against Q[:,0], ..., Q[:,r-1]
        Q[:,r] = A[:,r] - dot(Q[:,:r], R[:r,r])

        R[r,r] = norm(Q[:,r])
        assert R[r,r] > 0, "Input matrix doesn't have full column rank (numerically)"

        Q[:,r] /= R[r,r]

    return Q, R

def mod_gram_schmidt(A):
    """
    Computes the reduced QR decomposition of an input matrix A via
    modified Gram Schmidt. Assumes that A has full column rank.

    Output: matrices Q and R. If A is m x n, then Q has shape (m,n) and
    R has shape (n,n)
    """

    A = A.copy()
    m,n = A.shape
    assert m >= n, "Input matrix doesn't have full column rank"

    Q = np.zeros([m,n])
    R = np.zeros([n,n])

    for r in range(0, n):
        R[r,r] = norm(A[:,r])
        assert R[r,r] > 0, "Input matrix doesn't have full column rank (numerically)"

        Q[:,r] = A[:,r]/R[r,r]

        # Orthogonalize all future columns by column r
        R[r,(r+1):] = dot(A[:,(r+1):].T, Q[:,r])
        A[:,(r+1):] -= np.outer(Q[:,r], R[r,(r+1):])

    return Q, R


#A = np.random.rand(4, 3)

A = np.zeros([4,3])
eps = 1e-8
A[0,:] = 1.
A[1:,:] = eps*np.eye(3)

Qc,Rc = gram_schmidt(A)
Qm,Rm = mod_gram_schmidt(A)

# Test: A == QR and Q.T Q == I
