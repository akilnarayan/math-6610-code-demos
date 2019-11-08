import numpy as np
from numpy.linalg import norm

def rayleigh_quotient(A, v, normalized=True):
    """
    Computes the Rayleigh quotient of a matrix vector pair (A, v).
    """

    if not normalized:
        v = v/norm(v)

    return np.dot(v, np.dot(A, v))

def power_iteration_cycle(A, v):
    """
    Completes one cycle of power iteration using the input matrix A and
    vector v. Also returned is the value of the Rayleigh quotient
    associated to the updated vector, and the residual A*v - r(v)*v,
    where v is the updated vector and r is the Rayleigh quotient.
    """

    w = np.dot(A, v)
    v = w/norm(w)

    mu = rayleigh_quotient(A, v)
    err = norm(np.dot(A, v) - mu*v)

    return v, mu, err

def hotelling_deflation(A, mu, v):
    """
    Constructs a deflated matrix via Hotelling's method. This is
    unstable when iterated a large number of times.
    """

    v = v/norm(v)

    return A - mu * np.outer(v, v)

def orthogonalize(v, V):
    """
    Orthogonalizes v against the columns in V. Returns a unit-norm
    vector.
    """

    n = V.shape[1]
    if n < 1:
        return v/norm(v)
    else:
        Q,_ = np.linalg.qr(np.hstack((V, np.reshape(v, (v.size, 1)))))
        return Q[:,n]

def rayleigh_iteration_cycle(A, v, mu):
    """
    Completes one cycle of Rayleigh iteration using the input matrix A
    and vector v. Also returned is the value of the Rayleigh
    quotient associated to the updated vector, and the residual A*v -
    r(v)*v, where v is the updated vector and r is the Rayleigh
    quotient.
    """

    v = np.linalg.solve(A - mu*np.eye(v.size), v)
    v = v/norm(v)

    mu = rayleigh_quotient(A, v)
    err = norm(np.dot(A, v) - mu*v)

    return v, mu, err

def qr_iteration_cycle(A, mu):
    """
    Completes one cycle of the QR algorithm using the input matrix A.
    Also returned is an updated estimate of the eigenvalues of A,
    along with the vector error between the previous and new
    eigenvalue approximations.
    """

    Q, R = np.linalg.qr(A)
    A = np.dot(R, Q)
    lmbda = np.sort(np.diag(A))

    err = norm(lmbda - mu)/np.sqrt(A.shape[0])

    return A, lmbda, err
