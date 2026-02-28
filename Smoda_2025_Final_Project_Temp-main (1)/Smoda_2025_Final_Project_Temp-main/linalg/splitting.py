import numpy as np

from linalg.matrix import matrix_inverse_helper, norm_helper


def general_splitting_iteration_step(diag, decompose, b, current):
    """
    author: Dominik Fischer
    Helper function to perform a single iteration step of a general splitting algorithm.
    Will use matrix formalism function bundled with numpy to perform only trivial/elementary matrix operations.
    :param diag: already inverted matrix of the matrix to split away from the design matrix of the linear equation system.
    :param decompose: second decomposed matrix
    :param b: inhomogeneity of the linear equation system
    :param current: current result vector of the iteration
    :return: tuple of a new result vector and the residual vector to it's predecessor
    """
    next_iteration = diag @ (b - np.matvec(decompose, current))
    residue = next_iteration - current
    return next_iteration, residue


def apply_splitting(M, **kwargs):
    """
    author Dominik Fischer
    Helper function to perform the splitting of the design matrix of a linear equation system.
    :param M: design matrix of the linear equation system
    :return: splitted matrix for decomposition of the design matrix
    """
    kwargs.setdefault("splitting_name", "jacobi")
    match (kwargs["splitting_name"]):
        case "jacobi":
            return np.diag(np.diag(M))
        case "gauss_seidel":
            return np.tril(M, -1)
        case "richardson":
            kwargs.setdefault("tau", 1)
            return kwargs["tau"] * np.identity(M.shape[0])
        case "sor":
            # this is already a relaxed procedure (relaxation of the gauss-seidel-procedure)
            kwargs.setdefault("omega", 1.5)
            return (1. / kwargs["omega"]) * np.diag(np.diag(M)) + np.tril(M, -1)
        case _:
            raise ValueError(f"Unknown splitting method: {kwargs['splitting_name']}")


def general_splitting_iteration(M, b, initial_guess, iterations=100, tolerance=1e-10, use_tolerance=True,
                                use_condition=False, condition=None, splitting_algo=None, **kwargs):
    """
    author: Dominik Fischer
    Implements a general splitting algorithm for solving linear equation systems.
    :param M: design matrix of the linear equation system
    :param b: inhomogeneity of the linear equation system
    :param initial_guess: initial guess for the solution vector of the linear equation system.
    :param iterations: number of iterations to perform at most.
    :param tolerance: tolerance for the stopping criterion (could also break the iteration earlier).
    :param use_tolerance: boolean, to determine whether to use the tolerance or not.
    :param use_condition: boolean, to determine whether to use a self-defined condition to stop the iteration.
    :param condition: function, to evaluate the condition to stop the iteration. It must take the current result vector,
            the residual vector and the tolerance as arguments.
    :param splitting_algo: callable, to perform the splitting of the design matrix.
    :return: result of the linear equation system.
    """
    # set the default values for the additional keyword arguments
    # whether to use the provided callable for the splitting or not
    kwargs.setdefault("use_functional_splitting", False)
    # name of the splitting algorithm to use (depending on the choice of the algorithm it could be necessary to provide further keyword arguments)
    kwargs.setdefault("splitting_name", "jacobi")

    # get the actual splitting matrix and perform the decomposition of the design matrix
    if kwargs["use_functional_splitting"] and splitting_algo is not None:
        assert callable(splitting_algo)
        invertible = splitting_algo(M)
    else:
        invertible = apply_splitting(M, **kwargs)
    decomposition = M - invertible

    # need some vectors or at least fields to perform the iterations and store the (temporary) results
    current_iteration = np.array(initial_guess)
    residue = np.full(current_iteration.shape, np.inf)

    # need the inverse of the splitting matrix to perform the iteration step
    diag = matrix_inverse_helper(invertible)

    # distinguish between the different stopping criteria
    if use_condition and condition is not None:
        while condition(current_iteration, residue, tolerance):
            current_iteration, residue = general_splitting_iteration_step(diag, decomposition, b, current_iteration)

    elif iterations is not None:
        for _ in range(iterations):
            current_iteration, residue = general_splitting_iteration_step(diag, decomposition, b, current_iteration)
            if use_tolerance and norm_helper(residue) < tolerance:
                break

    elif use_tolerance:
        while norm_helper(residue) > tolerance:
            current_iteration, residue = general_splitting_iteration_step(diag, decomposition, b, current_iteration)

    return current_iteration


def jacobi_iteration_step(diag, decompose, b, current):
    """
    author: Dominik Fischer
    Note: This function was inspired by the Jacobi-Iteration used for approximately solving differential equations in the bachelor's lecture "Computerphysik" by Marcus Petschlies.
    The implementation was modified to general linear equation systems. (A trivial process as the design matrix is determined in the dgl version by the coefficient to determine the numerical derivative).
    In second step this was transformed to a matrix formalism.

    This implementation of the iteration step, will use numpy to perform elementary matrix operations.
    :param diag: inverse of the matrix containing only the **main** diagonal of the design matrix.
    :param decompose: the decomposed matrix of the design matrix. (What's remaining after subtracting the diagonal matrix from the design matrix)
    :param b: inhomogeneity of the linear equation system.
    :param current: current result vector of the iteration.
    :return: new result vector and the residual vector to it's predecessor.
    """
    next_iteration = diag @ (b - np.matvec(decompose, current))
    residue = next_iteration - current
    return next_iteration, residue


def jacobi_iteration(M, b, initial_guess, iterations=100, tolerance=1e-10, use_tolerance=True, use_condition=False,
                     condition=None):
    """
    author: Dominik Fischer
    Note: This function was inspired by the Jacobi-Iteration used for approximately solving differential equations in the bachelor's lecture "Computerphysik" by Marcus Petschlies.
    The implementation was modified to general linear equation systems. (A trivial process as the design matrix is determined in the dgl version by the coefficient to determine the numerical derivative).
    In second step this was transformed to a matrix formalism.

    Implements the Jacobi-Iteration for solving linear equation systems.
    The implementation is based on the matrix formalism and was not optimized for performance yet.
    So there are operations performed multiple times without changing results in between.

    :param M: design matrix of the linear equation system
    :param b: inhomogeneity of the linear equation system
    :param initial_guess: initial guess for the solution vector of the linear equation system.
    :param iterations: number of iterations to perform at most.
    :param tolerance: tolerance for the stopping criterion (could also break the iteration earlier).
    :param use_tolerance: boolean, to determine whether to use the tolerance or not.
    :param use_condition: boolean, to determine whether to use a self-defined condition to stop the iteration.
    :param condition: function, to evaluate the condition to stop the iteration. It must take the current result vector,
            the residual vector and the tolerance as arguments.
    :return: result of the linear equation system.
    """
    # decompose the design matrix for the Jacobi-Iteration by splitting a matrix with the main diagonal of the design matrix away from the rest.
    diagonal = np.diag(np.diag(M))
    decomposition = M - diagonal
    current_iteration = np.array(initial_guess)
    residue = np.full(current_iteration.shape, np.inf)

    # for the iteration step we need the inverse of the diagonal matrix
    diagonal_helper = matrix_inverse_helper(diagonal)

    # perform the iteration step for different stopping criteria
    if use_condition and condition is not None:
        while condition(current_iteration, residue, tolerance):
            current_iteration, residue = jacobi_iteration_step(diagonal_helper, decomposition, b, current_iteration)

    elif iterations is not None:
        for _ in range(iterations):
            current_iteration, residue = jacobi_iteration_step(diagonal_helper, decomposition, b, current_iteration)
            if (use_tolerance and norm_helper(residue) < tolerance):
                break

    elif use_tolerance:
        while norm_helper(residue) > tolerance:
            current_iteration, residue = jacobi_iteration_step(diagonal_helper, decomposition, b, current_iteration)

    return current_iteration


def gauss_seidel_iteration_step_simple(M, b, current, next):
    """
    author: Dominik Fischer
    Note: This function was inspired by the Gauss-Seidel-Iteration used for approximately solving differential equations in the bachelor's lecture "Computerphysik" by Marcus Petschlies.
    The implementation was modified to general linear equation systems. (A trivial process as the design matrix is determined in the dgl version by the coefficient to determine the numerical derivative).

    This implementation of the iteration step, will compute the new result vector elementwise.
    :param M: design matrix of the linear equation system.
    :param b: inhomogeneity of the linear equation system.
    :param current: current result vector of the iteration.
    :return: new result vector and the residual vector to it's predecessor.
    """
    shape = M.shape
    for k in range(shape[0]):
        current_sum = b[k]
        for i in range(k - 1):
            current_sum -= M[k, i] * next[i]

        for i in range(k + 1, shape[1]):
            current_sum -= M[k, i] * current[i]

        next[k] = current_sum / M[k, k]

    residue = next - current
    return next, residue


def gauss_seidel_iteration_step_matrix(l, u, b, current):
    """
    author: Dominik Fischer
    Note: This function was inspired by the Gauss-Seidel-Iteration used for approximately solving differential equations in the bachelor's lecture "Computerphysik" by Marcus Petschlies.
    The implementation was modified to general linear equation systems. (A trivial process as the design matrix is determined in the dgl version by the coefficient to determine the numerical derivative).
    In second step this was transformed to a matrix formalism.

    This implementation of the iteration step, will use numpy to perform elementary matrix operations.
    :param l: inverse of the matrix containing only the main diagonal and the lower triangular part of the design matrix.
    :param u: the decomposed matrix of the design matrix. (This is the upper triangular part of the design matrix)
    :param b: inhomogeneity of the linear equation system.
    :param current: current result vector of the iteration.
    :return: new result vector and the residual vector to it's predecessor.
    """
    next_iteration = l @ (b - np.matvec(u, current))
    residue = next_iteration - current
    return next_iteration, residue


def gauss_seidel_iteration(M, b, initial_guess, iterations=100, tolerance=1e-10, use_tolerance=True,
                           use_condition=False, condition=None):
    """
    author: Dominik Fischer
    Note: This function was inspired by the Gauss-Seidel-Iteration used for approximately solving differential equations in the bachelor's lecture "Computerphysik" by Marcus Petschlies.
    The implementation was modified to general linear equation systems. (A trivial process as the design matrix is determined in the dgl version by the coefficient to determine the numerical derivative).
    In second step this was transformed to a matrix formalism. (The matrix formalism is currently not used for the solving the equations)

    Implements the Gauss-Seidel-Iteration for solving linear equation systems.
    The Gauss-Seidel-Iteration uses the already calculated values of the next solution vector as well as the from the last iteration step.
    The current implementation uses elementwise computations.

    :param M: design matrix of the linear equation system
    :param b: inhomogeneity of the linear equation system
    :param initial_guess: initial guess for the solution vector of the linear equation system.
    :param iterations: number of iterations to perform at most.
    :param tolerance: tolerance for the stopping criterion (could also break the iteration earlier).
    :param use_tolerance: boolean, to determine whether to use the tolerance or not.
    :param use_condition: boolean, to determine whether to use a self-defined condition to stop the iteration.
    :param condition: function, to evaluate the condition to stop the iteration. It must take the current result vector,
            the residual vector and the tolerance as arguments.
    :return: result of the linear equation system.
    """
    # here no simple matrix arithmetics could be used
    # need some fields to store intermediate results
    current_iteration = np.array(initial_guess)
    next_iteration = np.zeros(current_iteration.shape)
    residue = np.full(current_iteration.shape, np.inf)

    # advanced techniques allow to use a matrix based formalism
    # lower_triangular = np.tril(M)
    # # this implementation for the upper triangular part reduces computations
    # upper_triangular = M - lower_triangular
    # # need the inverse of the lower triangular matrix to perform the iteration step
    # lower_inverse = matrix_inverse_helper(lower_triangular)

    # perform the iteration step for different stopping criteria
    if use_condition and condition is not None:
        while condition(current_iteration, residue, tolerance):
            current_iteration, residue = gauss_seidel_iteration_step_simple(M, b, current_iteration, next_iteration)

    elif iterations is not None:
        for _ in range(iterations):
            current_iteration, residue = gauss_seidel_iteration_step_simple(M, b, current_iteration, next_iteration)
            if use_tolerance and norm_helper(residue) < tolerance:
                break

    elif use_tolerance:
        while norm_helper(residue) > tolerance:
            current_iteration, residue = gauss_seidel_iteration_step_simple(M, b, current_iteration, next_iteration)

    return current_iteration


def sor_iteration(M, b, initial_guess, iterations=100, tolerance=1e-10, use_tolerance=True,
                  use_condition=False, condition=None):
    """

    :param M:
    :param b:
    :param initial_guess:
    :param iterations:
    :param tolerance:
    :param use_tolerance:
    :param use_condition:
    :param condition:
    """
    raise NotImplementedError()


def cg_iteration(M, b, initial_guess, iterations=100, tolerance=1e-10, use_tolerance=True,
                 use_condition=False, condition=None):
    """

    :param M:
    :param b:
    :param initial_guess:
    :param iterations:
    :param tolerance:
    :param use_tolerance:
    :param use_condition:
    :param condition:
    """
    raise NotImplementedError()


def gmres_iteration(M, b, initial_guess, iterations=100, tolerance=1e-10, use_tolerance=True,
                    use_condition=False, condition=None):
    """

    :param M:
    :param b:
    :param initial_guess:
    :param iterations:
    :param tolerance:
    :param use_tolerance:
    :param use_condition:
    :param condition:
    """
    raise NotImplementedError()


def relaxed_jacobi_iteration_step(diag, decompose, b, current, relaxation=1):
    """
    author: Dominik Fischer
    Note: This function was inspired by the Jacobi-Iteration used for approximately solving differential equations in the bachelor's lecture "Computerphysik" by Marcus Petschlies.
    The implementation was modified to general linear equation systems. (A trivial process as the design matrix is determined in the dgl version by the coefficient to determine the numerical derivative).
    In second step this was transformed to a matrix formalism.

    This implementation of the iteration step, will use numpy to perform elementary matrix operations.
    This a variant of the Jacobi-Iteration using relaxation.
    The result is in this case the weighted sum of the last iterations' result vector and the next one.
    Using the matrix formalism the mixin of the last iteration's result vector is performed by using the identity matrix.
    :param diag: inverse of the matrix containing only the **main** diagonal of the design matrix.
    :param decompose: the decomposed matrix of the design matrix. (What's remaining after subtracting the diagonal matrix from the design matrix)
    :param b: inhomogeneity of the linear equation system.
    :param current: current result vector of the iteration.
    :param relaxation: relaxation parameter for the relaxation of the Jacobi-Iteration.
    :return: new result vector and the residual vector to it's predecessor.
    """
    next_iteration = (1 - relaxation) * current + relaxation * diag @ (b - np.matvec(decompose, current))
    residue = next_iteration - current
    return next_iteration, residue


def relaxed_jacobi_iteration(M, b, initial_guess, relaxation=1, iterations=100, tolerance=1e-10, use_tolerance=True,
                             use_condition=False, condition=None):
    """
    author: Dominik Fischer
    Note: This function was inspired by the Jacobi-Iteration used for approximately solving differential equations in the bachelor's lecture "Computerphysik" by Marcus Petschlies.
    The implementation was modified to general linear equation systems. (A trivial process as the design matrix is determined in the dgl version by the coefficient to determine the numerical derivative).
    In second step this was transformed to a matrix formalism.

    Implements the Jacobi-Iteration for solving linear equation systems.
    The implementation is based on the matrix formalism and was not optimized for performance yet.
    So there are operations performed multiple times without changing results in between.

    This a variant of the Jacobi-Iteration using relaxation.
    The result is in this case the weighted sum of the last iterations' result vector and the next one.
    Using the matrix formalism the mixin of the last iteration's result vector is performed by using the identity matrix.

    :param M: design matrix of the linear equation system
    :param b: inhomogeneity of the linear equation system
    :param initial_guess: initial guess for the solution vector of the linear equation system.
    :param relaxation: relaxation parameter for the relaxation of the Jacobi-Iteration.
    :param iterations: number of iterations to perform at most.
    :param tolerance: tolerance for the stopping criterion (could also break the iteration earlier).
    :param use_tolerance: boolean, to determine whether to use the tolerance or not.
    :param use_condition: boolean, to determine whether to use a self-defined condition to stop the iteration.
    :param condition: function, to evaluate the condition to stop the iteration. It must take the current result vector,
            the residual vector and the tolerance as arguments.
    :return: result of the linear equation system.
    """
    # decompose the design matrix for the Jacobi-Iteration by splitting a matrix with the main diagonal of the design
    # matrix away from the rest.
    diagonal = np.diag(np.diag(M))
    decomposition = M - diagonal
    current_iteration = np.array(initial_guess)
    residue = np.full(current_iteration.shape, np.inf)

    # for the iteration step we need the inverse of the diagonal matrix
    diagonal_helper = matrix_inverse_helper(diagonal)

    # perform the iteration step for different stopping criteria
    if use_condition and condition is not None:
        while condition(current_iteration, residue, tolerance):
            next_iteration, residue = jacobi_iteration_step(diagonal_helper, decomposition, b,
                                                            current_iteration)
            current_iteration = (1 - relaxation) * current_iteration + relaxation * next_iteration

    elif iterations is not None:
        for _ in range(iterations):
            next_iteration, residue = jacobi_iteration_step(diagonal_helper, decomposition, b,
                                                            current_iteration)
            current_iteration = (1 - relaxation) * current_iteration + relaxation * next_iteration
            if use_tolerance and norm_helper(residue) < tolerance:
                break

    elif use_tolerance:
        while norm_helper(residue) > tolerance:
            next_iteration, residue = jacobi_iteration_step(diagonal_helper, decomposition, b,
                                                            current_iteration)
            current_iteration = (1 - relaxation) * current_iteration + relaxation * next_iteration

    return current_iteration


def relaxed_gauss_seidel_iteration_step_simple(M, b, current, next_estimate, relaxation=1):
    """
    author: Dominik Fischer
    Note: This function was inspired by the Gauss-Seidel-Iteration used for approximately solving differential equations in the bachelor's lecture "Computerphysik" by Marcus Petschlies.
    The implementation was modified to general linear equation systems. (A trivial process as the design matrix is determined in the dgl version by the coefficient to determine the numerical derivative).

    This implementation of the iteration step, will compute the new result vector elementwise.
    This a variant of the Gauss-Seidel-Iteration using relaxation.
    The result is in this case the weighted sum of the last iterations' result vector and the next one.
    Using the matrix formalism the mixin of the last iteration's result vector is performed by using the identity matrix.
    :param M: design matrix of the linear equation system.
    :param b: inhomogeneity of the linear equation system.
    :param current: current result vector of the iteration.
    :param next_estimate: field to hold the next estimate of the solution vector.
    :param relaxation: relaxation parameter for the relaxation of the Gauss-Seidel-Iteration.
    :return: new result vector and the residual vector to it's predecessor.
    """
    shape = M.shape
    for k in range(shape[0]):
        current_sum = b[k]
        for i in range(k - 1):
            current_sum -= M[k, i] * next_estimate[i]

        for i in range(k + 1, shape[1]):
            current_sum -= M[k, i] * current[i]

        next_estimate[k] = current_sum / M[k, k]
        next_estimate[k] *= relaxation
        next_estimate[k] += (1 - relaxation) * current[k]

    residue = next_estimate - current
    return next_estimate, residue


def relaxed_gauss_seidel_iteration_step_matrix(l, u, b, current, relaxation=1):
    """
    author: Dominik Fischer
    Note: This function was inspired by the Gauss-Seidel-Iteration used for approximately solving differential equations in the bachelor's lecture "Computerphysik" by Marcus Petschlies.
    The implementation was modified to general linear equation systems. (A trivial process as the design matrix is determined in the dgl version by the coefficient to determine the numerical derivative).
    In second step this was transformed to a matrix formalism.

    This implementation of the iteration step, will use numpy to perform elementary matrix operations.
    This a variant of the Gauss-Seidel-Iteration using relaxation.
    The result is in this case the weighted sum of the last iterations' result vector and the next one.
    Using the matrix formalism the mixin of the last iteration's result vector is performed by using the identity matrix.
    :param l: inverse of the matrix containing only the main diagonal and the lower triangular part of the design matrix.
    :param u: the decomposed matrix of the design matrix. (This is the upper triangular part of the design matrix)
    :param b: inhomogeneity of the linear equation system.
    :param current: current result vector of the iteration.
    :param relaxation: relaxation parameter for the relaxation of the Gauss-Seidel-Iteration.
    :return: new result vector and the residual vector to it's predecessor.
    """
    next_iteration = l @ (b - np.matvec(u, current))
    next_iteration *= relaxation
    next_iteration += (1 - relaxation) * current
    residue = next_iteration - current
    return next_iteration, residue


def relaxed_gauss_seidel_iteration(M, b, initial_guess, relaxation=1, iterations=100, tolerance=1e-10,
                                   use_tolerance=True,
                                   use_condition=False, condition=None):
    """
    author: Dominik Fischer
    Note: This function was inspired by the Gauss-Seidel-Iteration used for approximately solving differential equations in the bachelor's lecture "Computerphysik" by Marcus Petschlies.
    The implementation was modified to general linear equation systems. (A trivial process as the design matrix is determined in the dgl version by the coefficient to determine the numerical derivative).
    In second step this was transformed to a matrix formalism. (The matrix formalism is currently not used for the solving the equations)

    Implements the Gauss-Seidel-Iteration for solving linear equation systems.
    The Gauss-Seidel-Iteration uses the already calculated values of the next solution vector as well as the from the last iteration step.
    The current implementation uses elementwise computations.

    This a variant of the Gauss-Seidel-Iteration using relaxation.
    The result is in this case the weighted sum of the last iterations' result vector and the next one.
    Using the matrix formalism the mixin of the last iteration's result vector is performed by using the identity matrix.

    :param M: design matrix of the linear equation system
    :param b: inhomogeneity of the linear equation system
    :param initial_guess: initial guess for the solution vector of the linear equation system.
    :param iterations: number of iterations to perform at most.
    :param tolerance: tolerance for the stopping criterion (could also break the iteration earlier).
    :param use_tolerance: boolean, to determine whether to use the tolerance or not.
    :param use_condition: boolean, to determine whether to use a self-defined condition to stop the iteration.
    :param condition: function, to evaluate the condition to stop the iteration. It must take the current result vector,
            the residual vector and the tolerance as arguments.
    :return: result of the linear equation system.
    """
    # here no simple matrix arithmetics could be used
    # need some fields to store intermediate results
    current_iteration = np.array(initial_guess)
    next_iteration = np.zeros(current_iteration.shape)
    residue = np.full(current_iteration.shape, np.inf)

    # advanced techniques allow to use a matrix based formalism
    # lower_triangular = np.tril(M)
    # # this implementation for the upper triangular part reduces computations
    # upper_triangular = M - lower_triangular
    # # need the inverse of the lower triangular matrix to perform the iteration step
    # lower_inverse = matrix_inverse_helper(lower_triangular)

    # perform the iteration step for different stopping criteria
    if use_condition and condition is not None:
        while condition(current_iteration, residue, tolerance):
            current_iteration, residue = relaxed_gauss_seidel_iteration_step_simple(M, b, current_iteration,
                                                                                    next_iteration, relaxation)

    elif iterations is not None:
        for _ in range(iterations):
            current_iteration, residue = relaxed_gauss_seidel_iteration_step_simple(M, b, current_iteration,
                                                                                    next_iteration, relaxation)
            if use_tolerance and norm_helper(residue) < tolerance:
                break

    elif use_tolerance:
        while norm_helper(residue) > tolerance:
            current_iteration, residue = relaxed_gauss_seidel_iteration_step_simple(M, b, current_iteration,
                                                                                    next_iteration, relaxation)

    return current_iteration


def relaxed_general_splitting_iteration(M, b, initial_guess, relaxation=1, iterations=100, tolerance=1e-10,
                                        use_tolerance=True,
                                        use_condition=False, condition=None, splitting_algo=None, **kwargs):
    """
    author: Dominik Fischer
    Implements a general splitting algorithm for solving linear equation systems.
    The result is in this case the weighted sum of the last iterations' result vector and the next one.
    Using the matrix formalism the mixin of the last iteration's result vector is performed by using the identity matrix.
    :param M: design matrix of the linear equation system
    :param b: inhomogeneity of the linear equation system
    :param initial_guess: initial guess for the solution vector of the linear equation system.
    :param relaxation: relaxation parameter for the relaxation of the Gauss-Seidel-Iteration.
    :param iterations: number of iterations to perform at most.
    :param tolerance: tolerance for the stopping criterion (could also break the iteration earlier).
    :param use_tolerance: boolean, to determine whether to use the tolerance or not.
    :param use_condition: boolean, to determine whether to use a self-defined condition to stop the iteration.
    :param condition: function, to evaluate the condition to stop the iteration. It must take the current result vector,
            the residual vector and the tolerance as arguments.
    :param splitting_algo: callable, to perform the splitting of the design matrix.
    :return: result of the linear equation system.
    """
    # set the default values for the additional keyword arguments
    # whether to use the provided callable for the splitting or not
    kwargs.setdefault("use_functional_splitting", False)
    # name of the splitting algorithm to use (depending on the choice of the algorithm it could be necessary to
    # provide further keyword arguments)
    kwargs.setdefault("splitting_name", "jacobi")

    # get the actual splitting matrix and perform the decomposition of the design matrix
    if kwargs["use_functional_splitting"] and splitting_algo is not None:
        assert callable(splitting_algo)
        invertible = splitting_algo(M)
    else:
        invertible = apply_splitting(M, **kwargs)
    decomposition = M - invertible

    # need some vectors or at least fields to perform the iterations and store the (temporary) results
    current_iteration = np.array(initial_guess)
    residue = np.full(current_iteration.shape, np.inf)

    # need the inverse of the splitting matrix to perform the iteration step
    diag = matrix_inverse_helper(invertible)

    # distinguish between the different stopping criteria
    if use_condition and condition is not None:
        while condition(current_iteration, residue, tolerance):
            next_iteration, residue = general_splitting_iteration_step(diag, decomposition, b, current_iteration)
            current_iteration = (1 - relaxation) * current_iteration + relaxation * next_iteration

    elif iterations is not None:
        for _ in range(iterations):
            next_iteration, residue = general_splitting_iteration_step(diag, decomposition, b, current_iteration)
            current_iteration = (1 - relaxation) * current_iteration + relaxation * next_iteration
            if use_tolerance and norm_helper(residue) < tolerance:
                break

    elif use_tolerance:
        while norm_helper(residue) > tolerance:
            next_iteration, residue = general_splitting_iteration_step(diag, decomposition, b, current_iteration)
            current_iteration = (1 - relaxation) * current_iteration + relaxation * next_iteration

    return current_iteration
