import numpy as np
from scipy.linalg import lu_factor, lu_solve


def matrix_inverse_helper(M):
    """
    Helper function to invert a matrix.
    Does currently use numpy for matrix inversion.
    Will use lu decomposition for matrix inversion in the future.
    Currently the lu decomposition is implemented using scipy functionality.
    :param M: matrix to invert
    :return: inverse of M
    """

    # use lu decomposition for matrix inversion
    lu, permutations = lu_factor(M)

    # now need to solve the linear system for whole bunch of inhomogenities
    identity_inhomo = np.identity(M.shape[0])
    return np.array([lu_solve((lu, permutations), identity_inhomo[:, idx]) for idx in range(M.shape[1])]).T


def matrix_inverse_second(mat):
    # lu, permutations = lu_factor(mat)
    # And What about pivoting now?
    # does not work currently with pivoting
    lu, uu = lu_decomposition_helper(mat)

    identity_inhomo = np.identity(mat.shape[0])
    return np.array([lu_solve_helper(lu, uu, identity_inhomo[:, idx]) for idx in range(mat.shape[1])]).T


def matrix_system_solve(mat, inhomogenity):
    lu, uu = lu_decomposition_helper(mat)
    return lu_solve_helper(lu, uu, inhomogenity)


def lu_decomposition():
    # Implementation should orient itself at the function parameters and return values of the implementation
    # of the scipy python package
    pass


def norm_helper(vec):
    return np.linalg.norm(vec)


def lu_decomposition_helper(M, pivot=False):
    """
    lu_decomposition_helper


    @author: Dominik Fischer
    @date: 2023-xx-xx

    Implementation is done during homeworks and exercises for the CP bachelor lecture by Marcus Petschlies.
    So this implementation is also oriented at provided example codes.
    This is more concerning w.r.t. the pivoting implementation which is close to an copy of the provided example code.
    Pivoting is used to reduce the effect of numerical cancelling errors when dividing by very small numbers.
    :param M: matrix to decompose
    :param pivot: whether to use pivoting or not
    :return: L, U decomposition of M
    """
    # Implementation ist von der Computerphysik-Vorlesung inspiriert (aber: Pivotisierung ergibt teilweise quascht)
    # using the crout algorithm
    l_temp = np.zeros(np.shape(M))
    u_temp = np.zeros(np.shape(M))
    if not pivot:
        for i in range(np.shape(M)[0]):
            # calculate the U row
            if i == 0:
                u_temp[i] = M[i]
            else:
                for k in range(np.shape(M)[1]):
                    u_temp[i, k] = M[i, k] - np.sum(l_temp[i, 0:i] * u_temp[0:i, k])

            if u_temp[i, i] == 0:
                print("[lu_decomp_pivot] Fehler, Matrix ist singulär\n")
                raise ValueError("Singular Matrix")

            # the main diagonal of L is always 1
            l_temp[i, i] = 1

            # calculate the L column
            for k in range(np.shape(M)[0]):
                l_temp[k, i] = (M[k, i] - np.sum(l_temp[k, 0:i] * u_temp[0:i, i])) / u_temp[i, i]

    else:
        # Pivotisierung um Signifikanz zu erhalten
        permutations = np.zeros(np.shape(M)[0], dtype=int)
        current_design_matrix = M.copy()
        for i in range(np.shape(M)[0]):
            # need first to look up the row with absolutely largest value
            absolute_vector = np.abs(current_design_matrix[i:, i])
            max_index = np.argmax(absolute_vector)
            pivot_value = current_design_matrix[i + max_index, i]
            permutations[i] = max_index
            current_design_matrix[i:, i] /= pivot_value
            if max_index != i:
                temp_row = current_design_matrix[i, :]
                current_design_matrix[i, :] = current_design_matrix[i + max_index, :]
                current_design_matrix[i + max_index, :] = temp_row

            # continue with the proper decomposition
            # calculate the U row
            if i == 0:
                u_temp[i] = current_design_matrix[i]
            else:
                for k in range(np.shape(M)[1]):
                    u_temp[i, k] = current_design_matrix[i, k] - np.sum(l_temp[i, 0:i] * u_temp[0:i, k])

            if u_temp[i, i] == 0:
                print("[lu_decomp_pivot] Fehler, Matrix ist singulär\n")
                raise ValueError("Singular Matrix")

            # the main diagonal of L is always 1
            l_temp[i, i] = 1

            # calculate the L column
            for k in range(np.shape(M)[0]):
                l_temp[k, i] = (current_design_matrix[k, i] - np.sum(l_temp[k, 0:i] * u_temp[0:i, i])) / u_temp[i, i]

        # last step: compensate for the permutations of the design matrix
        # for i in reversed(range(np.shape(permutations)[0])):
        #     if i != permutations[i]:
        #         try:
        #             temp_row = l_temp[i, :]
        #             l_temp[i, :] = l_temp[permutations[i], :]
        #             l_temp[permutations[i], :] = temp_row
        #         except IndexError:
        #             print(f"IndexError: {i} != {permutations[i]}")
        #             print(type(i), type(permutations[i]))
        #             raise IndexError
        return l_temp, u_temp, permutations

    return l_temp, u_temp


def lu_solve_helper(l_mat, u_mat, b, permutations=None):
    if u_mat is None:
        # must extract both the L matrix and the U matrix from the one provided matrix
        l = np.tril(l_mat)
        print("l")
        print(l)
        u = np.triu(l_mat)
        print("u")
        print(u)

    else:
        # both matrices are provided separately
        u = u_mat
        l = l_mat

    p = np.arange(l.shape[0]) if permutations is None else permutations

    y_vector = np.zeros(b.shape)
    x_vector = np.zeros(b.shape)
    # forward iteration for the solution with a triangular matrix
    for i in range(b.shape[0]):
        y_vector[i] = b[p[i]] - np.sum(l[i, 0:i] * y_vector[0:i])

    # backward iteration for the solution with a triangular matrix
    for i in reversed(range(b.shape[0])):
        x_vector[i] = (y_vector[i] - np.sum(u[i, (i + 1):] * x_vector[(i + 1):])) / u[i, i]

    return x_vector
