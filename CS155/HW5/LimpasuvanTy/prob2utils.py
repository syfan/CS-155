# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

import numpy as np

def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    return eta * (reg * Ui - Vj * (Yij - np.dot(Ui,Vj)))

def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regulariz+ed loss function with
    respect to Vj multiplied by eta.
    """
    return eta * (reg * Vj - np.dot(Ui, np.transpose(Yij - np.dot(Ui, Vj))))
def get_err(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    total = reg * (np.linalg.norm(U) + np.linalg.norm(V))
    for i in np.random.permutation(len(Y)):
        value = Y[i]
        row = value[0] - 1
        column = value[1] - 1
        y = value[2]
        pred = np.dot(U[row], V[column])
        total += np.power(y - pred, 2)
    return total / 2

def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    U = np.random.rand(M, K) - 0.5
    V = np.random.rand(N, K) - 0.5
    zeroth_err = 0
    first_err = 0
    prev_err = 0
    curr_err = 0
    for q in range(max_epochs):
        for i in np.random.permutation(len(Y)):
            value = Y[i]
            row = value[0] - 1
            column = value[1] - 1
            y = value[2]
            U[row] = U[row] - grad_U(U[row], y, V[column], reg, eta)
            V[column] = V[column] - grad_V(V[column], y, U[row], reg, eta)
        if (q == 0):
            zeroth_err = get_err(U, V, Y, reg)
        elif (q == 1):
            first_err = get_err(U, V, Y, reg)
            prev_err = zeroth_err
        else:
            prev_err = curr_err
            curr_err = get_err(U, V, Y, reg)
            if(np.absolute((curr_err - prev_err)) / 
               np.absolute((first_err - zeroth_err)) < eps):
                break
        
        
    return (U, V, curr_err)