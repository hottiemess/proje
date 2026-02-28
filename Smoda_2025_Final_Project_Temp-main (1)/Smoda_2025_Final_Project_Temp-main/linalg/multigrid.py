import queue

import numpy as np

from linalg.matrix import lu_decomposition_helper
from linalg.splitting import relaxed_jacobi_iteration


def restriction_operator(size):
    set_value = [1, 2, 1]
    mat = np.zeros((size, size // 2))
    for i in range(size // 2):
        mat[i, i:i + 3] = set_value

    return 0.25 * mat


def restrict_lattice(residue):
    grain_lattice_size = np.shape(residue)[0]
    operator = restriction_operator(grain_lattice_size)
    return np.matvec(operator, residue), operator


def prolongate_lattice(advanced_residue, restriction=None):
    coarse_lattice_size = np.shape(advanced_residue)[0]
    if restriction is None:
        operator = np.transpose(restriction_operator(2 * coarse_lattice_size))
    else:
        operator = np.transpose(restriction)
    return np.matvec(operator, advanced_residue)


# TODO: Implementation is not completed (maybe wont work)
def reduce_design_matrix(M):
    # only operate on every second element
    reduced = np.zeros((np.floor(np.shape(M)[0] / 2), np.floor(np.shape(M)[1] / 2)))
    for i in range(np.shape(reduced)[0]):
        for j in range(np.shape(reduced)[1]):
            reduced[i, j] = M[2 * i + 1, 2 * j + 1]

    return reduced


def coarse_solver(A, r):
    assert isinstance(A, np.ndarray)
    assert isinstance(r, np.ndarray)
    # will use a LU decomposition to solve the linear system
    l, u = lu_decomposition_helper(A)
    y_vector = np.zeros(r.shape)
    x_vector = np.zeros(r.shape)
    # forward iteration for the solution with a triangular matrix
    for i in range(r.shape[0]):
        y_vector[i] = r[i] - np.sum(l[i, 0:i] * y_vector[0:i])

    # backward iteration for the solution with a triangular matrix
    for i in reversed(range(r.shape[0])):
        x_vector[i] = (y_vector[i] - np.sum(u[i, (i + 1):] * x_vector[(i + 1):])) / u[i, i]

    return x_vector


def two_grid_cycle(M, b, initial_guess, iterations=10, relaxation=1):
    # smoothing
    temp_result = relaxed_jacobi_iteration(M, b, initial_guess, iterations=iterations, relaxation=relaxation)
    residues = b - M @ temp_result
    coarse_residues, projection_mat = restrict_lattice(residues)

    # solve on the coarse grid
    coarse_residue_optimum = coarse_solver(reduce_design_matrix(M), coarse_residues)

    # interpolate on the fine grid
    fine_result = prolongate_lattice(coarse_residue_optimum, projection_mat)

    # correct the current solution
    return temp_result + fine_result


def multi_grid_cycle(M, b, initial_guess, pre_iterations=10, post_iterations=10, relaxation=1, n_grids=2):
    initial_result = np.array(initial_guess)
    grid_queue = queue.LifoQueue()
    design_queue = queue.LifoQueue()
    result_queue = queue.LifoQueue()
    current_inhom = b
    current_design_matrix = M
    for _ in range(n_grids):
        initial_result = relaxed_jacobi_iteration(current_design_matrix, current_inhom, initial_result,
                                                  iterations=pre_iterations, relaxation=relaxation)
        residues = b - current_design_matrix @ initial_result
        result_queue.put(initial_result)
        design_queue.put(current_design_matrix)
        current_design_matrix = reduce_design_matrix(current_design_matrix)
        current_inhom, projection_mat = restrict_lattice(residues)
        grid_queue.put(projection_mat)

    coarse_residue_optimum = coarse_solver(current_design_matrix, residues)

    # prolonge back to the other grids
    while not grid_queue.empty():
        projection_mat = grid_queue.get()
        current_inhom = prolongate_lattice(coarse_residue_optimum, projection_mat)
        current_design_matrix = design_queue.get()
        last_solution = result_queue.get()
        temp_result = last_solution + prolongate_lattice(coarse_residue_optimum, projection_mat)
        coarse_residue_optimum = relaxed_jacobi_iteration(current_design_matrix, current_inhom, temp_result,
                                                          iterations=post_iterations, relaxation=relaxation)

    return coarse_residue_optimum
