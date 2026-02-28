import logging
import sys
import uuid
from enum import StrEnum
from inspect import signature
from typing import Iterable, Callable
from warnings import warn

import numdifftools as nd
import numpy as np
from numpy.random import default_rng
from scipy.optimize import line_search as linesearch, dual_annealing
from typing_extensions import deprecated

from derivatives.FindiffWrapper import FindiffWrapper, DerivativeWrapper, NumDiffToolsDerivatives, \
    FiniteDifferenceDerivatives
from linalg.matrix import norm_helper, matrix_inverse_second as matrix_inverse
from linalg.search import line_search_helper
from optimize.util import permuted_cost_function, ValueView

print(sys.path)


# enumeration class to avoid typos when writting down the keys for the names of the algorithms!
class DescentAlgorithms(StrEnum):
    GRADIENT_DESCENT = "gradient descent"
    MODIFIED_GRADIENT_DESCENT = "gradient descent adv."
    BFGS_SIMPLE = "bfgs simple"
    BFGS_QUASI = "bfgs quasi"
    BFGS_ADVANCED = "bfgs quasi"
    BFGS = "bfgs"
    NEWTON = "newton"
    QUASI_NEWTON = "quasi-newton"


# enumerate the level names for the step information collection
class StepInformationLevel(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    TRACEONLY = "trace"


# additional handler for Hessian computations if required
class HessianDelegate:
    def __init__(self, optimizer: GradientDescent, handler: Callable):
        self.delegate = handler
        self.optimizer = optimizer

    def __call__(self, func, x, xp, last_hessian, derivatives: DerivativeWrapper, inverse=True, next_grad=None,
                 current_grad=None):
        # not able to submit inverse matrices
        self.optimizer.add_hessian_evaluation()
        return self.delegate(x), False


# handler function to collect the intermediate steps of the algorithms
def no_op_collection_helper(optimizer: GradientDescent, xk0, xk, alpha, gk, pk, scipy_fk0, scipy_alpha,
                            trace_divider=500,
                            *args, **kwargs):
    # no-op function as a default implementation
    # should do nothing to not waste compution time
    pass


def debug_collection_helper(optimizer: GradientDescent, xk0, xk, alpha, gk, pk, scipy_fk0, scipy_alpha,
                            trace_divider=500,
                            *args, **kwargs):
    """

    :param optimizer: optimizer using this function to extract information about the steps done in process.
    :param scipy_alpha: (current) step size value from the scipy implementation of the line search algorithm
    :param pk: search direction of the current iteration
    :param gk: gradient of the current iteration
    :param xk: result of this steps update of the approximation to the optimum
    :param xk0: previous approximation to the optimum
    :param scipy_fk0: value of the function for the previous approximation with scipy line search implementation
    :param args: further arguments directly passed down to the function to be optimised
    :param kwargs: further keyword arguments directly passed down to the function to be optimised.
    """
    # this two are always written on method invokation => need not to save them as a class property!
    remember_state = np.vstack((pk, gk, xk))
    step_cost_state = [optimizer.likelihood_function(xk0, *args, **kwargs), np.nan if scipy_fk0 is None else scipy_fk0, alpha,
                       np.nan if scipy_alpha is None else scipy_alpha]
    if optimizer.previous_result is None:
        optimizer.previous_result = np.array([remember_state])
        optimizer.previous_cost = np.array([step_cost_state])
        optimizer.previous_counter = 1
    elif optimizer.previous_counter % trace_divider == 0:
        optimizer.flush_steps(history=remember_state, cost=step_cost_state)
        optimizer.previous_counter += 1
    else:
        optimizer.previous_result = np.append(optimizer.previous_result, [remember_state], axis=0)
        optimizer.previous_cost = np.append(optimizer.previous_cost, [step_cost_state], axis=0)
        optimizer.previous_counter += 1


def info_collection_helper(optimizer: GradientDescent, xk0, xk, alpha, gk, pk, scipy_fk0, scipy_alpha,
                           trace_divider=500,
                           *args, **kwargs):

    cost = optimizer.likelihood_function(xk, *args, **kwargs)
    # will only collect the visitied positions and the values of the likelihood
    if optimizer.previous_result is None:
        optimizer.previous_result = np.array([xk])
        optimizer.previous_cost = np.array([cost])
        optimizer.previous_counter = 1
    elif optimizer.previous_counter % trace_divider == 0:
        optimizer.flush_steps(history=xk, cost=cost)
        optimizer.previous_counter += 1
    else:
        optimizer.previous_result = np.append(optimizer.previous_result, [xk], axis=0)
        optimizer.previous_cost = np.append(optimizer.previous_cost, [cost], axis=0)
        optimizer.previous_counter += 1


def trace_collection_helper(optimizer: GradientDescent, xk0, xk, alpha, gk, pk, scipy_fk0, scipy_alpha,
                            trace_divider=500,
                            *args, **kwargs):
    # we are only interested in the visited points
    if optimizer.previous_result is None:
        optimizer.previous_result = np.array([xk])
        optimizer.previous_counter = 1
    elif optimizer.previous_counter % trace_divider == 0:
        optimizer.flush_steps(history=xk)
        optimizer.previous_counter += 1
    else:
        optimizer.previous_result = np.append(optimizer.previous_result, [xk], axis=0)
        optimizer.previous_counter += 1


collecting_step_information = True
advanced_debug = False
logger = logging.getLogger(__name__)


# sceleton of a hessian update function
def update_hessian(func, x, xp, last_hessian, derivatives: DerivativeWrapper, inverse=True, next_grad=None,
                   current_grad=None):
    """
    Sceleton of function which updates the hessian matrix or to be more accurate it's inverse.
    :param func: cost function used for optimization.
    :param x: x value to evaluate the new hessian at.
    :param xp: The parameter vector used for the last iteration.
    :param last_hessian: the hessian or it's inverse from the last iteration, which is no to be updated.
    :param derivatives: wrapper class for the derivative calculation.
    :param inverse: boolean, indicating whether the hessian matrix should be updated or its inverse.
    :param next_grad: (optional) gradient of the cost function at x
    :param current_grad: (optional) gradient of the cost function at x - step
    :return: new hessian matrix or it's inverse and a boolean indicating whether the hessian matrix is it's inverse or not.
    """
    return last_hessian, True


def extract_signature_information(update_strategy: Callable):
    """
    extract_signature_information

    @author Dominik Fischer
    @date 2026-02-26

    Helper function to extract signature information.


    :param update_strategy: function to check signature information from
    :return: tuple of extracted information: (positional arguemnts, keyword arguments, number of positional arguments, contribution from numpy arrays)
    """
    sig = signature(update_strategy)
    position_argument = []
    keyword_argument = []
    positional_counter = 0
    has_ndarray = False
    for param in sig.parameters.values():
        if param.kind == param.POSITIONAL_OR_KEYWORD:
            position_argument.append(param.name)
            positional_counter += 1
            if issubclass(param.annotation, np.ndarray):
                has_ndarray = True
        elif param.kind == param.KEYWORD_ONLY:
            keyword_argument.append(param.name)
    return position_argument, keyword_argument, positional_counter, has_ndarray


update_strategy_signature, _, _, _ = extract_signature_information(update_hessian)


def bfgs_update(func, x, xp, last_hessian, derivatives: DerivativeWrapper, inverse=True, next_grad=None,
                current_grad=None,
                **kwargs):
    """
    Implementation of the hessian update stragtegy for BFGS.

    The algorithm updates the hessian or merely it's inverse.
    This implementation is based on the formulas of [Nocedal, p. 24 and pp. 135-162].
    This implementation is done such that it reuse already calculated gradients if possible to reduce the number of gradient evaluations (saves time).

    @author: Dominik Fischer
    @date: no date could be provided

    :param func: cost function used for optimization.
    :param x: x value to evaluate the new hessian at.
    :param xp: The parameter vector used for the last iteration.
    :param last_hessian: the hessian or it's inverse from the last iteration, which is no to be updated.
    :param derivatives: wrapper class for the derivative calculation.
    :param inverse: boolean, indicating whether the hessian matrix should be updated or its inverse.
    :param next_grad: (optional) gradient of the cost function at x
    :param current_grad: (optional) gradient of the cost function at x - step
    :return: new hessian matrix or it's inverse and a boolean indicating whether the hessian matrix is it's inverse or not.
    """
    # list of keyword arguments used in this function:
    # enforce_gradient_calculation: boolean, indicating whether the
    #       gradient calculation should be performed manually or not. default: False
    internal_keywords = ["enforce_gradient_calculation"]
    forward_kwargs = kwargs.copy()
    for key in internal_keywords:
        if key in forward_kwargs:
            del forward_kwargs[key]
    # list of forwarded keyword arguments
    # all unused keyword arguments will be forwared for the evaluation of the function/it's gradient.

    assert x is not None and xp is not None
    assert callable(func)
    # performance and efficiency could be increased by reusing gradients if they were already computed
    if next_grad is None or current_grad is None or kwargs.get("enforce_gradient_calculation", False):
        grad_diff = (derivatives.gradient(func, x, **forward_kwargs) - derivatives.gradient(func, xp, **forward_kwargs))
    else:
        grad_diff = next_grad - current_grad
    delta_x = x - xp

    # define often used quantities for normalisation of the expressions
    norm_factor = grad_diff.T @ delta_x
    norm_fraction = 1. / norm_factor

    # slightly different procedures when approximating the inverse in opposite to approximating the hessian
    # the implemented formulas are taken from [Nocedal]
    if inverse:
        dyadic = np.outer(delta_x, delta_x)
        temp_first = grad_diff.T @ last_hessian @ grad_diff + norm_factor
        temp_first /= (norm_factor ** 2)
        temp_second = last_hessian @ np.outer(grad_diff, delta_x) + np.outer(delta_x, grad_diff) @ last_hessian
        temp_second /= norm_factor
        new_h = last_hessian + temp_first * dyadic - temp_second
        if advanced_debug:
            reference_approximation = (np.identity(len(x)) - norm_fraction * np.outer(delta_x,
                                                                                      grad_diff)) @ last_hessian @ (
                                              np.identity(len(x)) - norm_fraction * np.outer(grad_diff,
                                                                                             delta_x)) + norm_fraction * dyadic
            print("Test for the closeness of the two procedures")
            print(np.allclose(reference_approximation, new_h, rtol=1e-12, atol=1e-12))
            cmp_newton = np.linalg.inv(derivatives.hessian(func, x, **forward_kwargs))
            print(np.allclose(reference_approximation, cmp_newton, rtol=1e-12, atol=1e-12))
            logger.debug("Test for the closeness of the two procedures")
            logger.debug(np.isclose(reference_approximation, new_h, rtol=1e-12, atol=1e-12))

    else:
        dyadic_grad = np.outer(grad_diff, grad_diff)
        hessian_effect = last_hessian @ grad_diff
        temp_first = dyadic_grad * norm_fraction
        temp_second = np.outer(hessian_effect, hessian_effect) / np.dot(delta_x, last_hessian @ delta_x)
        new_h = last_hessian + temp_first - temp_second

    return new_h, inverse


def newton_update(func, x, xp, last_hessian, derivatives: DerivativeWrapper, inverse=True, next_grad=None,
                  current_grad=None,
                  **kwargs):
    """
    newton_update

    @author Dominik Fischer
    @date 2026-01-01

    Update strategy for the hesse matrix for Newton procedure.
    It is less a update strategy than an procedure to compute the (exact) hesse matrix and invert it if necessary.

    :param func: cost function used for optimization.
    :param x: x value to evaluate the new hessian at.
    :param xp: The parameter vector used for the last iteration.
    :param last_hessian: the hessian or it's inverse from the last iteration, which is no to be updated.
    :param derivatives: wrapper class for the derivative calculation.
    :param inverse: boolean, indicating whether the hessian matrix should be updated or its inverse.
    :param next_grad: (optional) gradient of the cost function at x
    :param current_grad: (optional) gradient of the cost function at x - step
    :return: new hessian matrix or it's inverse and a boolean indicating whether the hessian matrix is it's inverse or not.
    """
    # the newton update strategy consists simply in recomputing the hessian and inverse it if necessary
    hessian = derivatives.hessian(func, x)
    if inverse:
        return matrix_inverse(hessian), True
    else:
        return hessian, inverse


def gradient_descent_advanced_update(func, x, xp, last_hessian, derivatives: DerivativeWrapper, inverse=True,
                                     next_grad=None,
                                     current_grad=None, **kwargs):
    """
    gradient_descent_advanced_update

    @author Dominik Fischer
    @date 2026-01-01

    Update strategy for simplified procedures to update the direction matrix of procedures which won't use the heses matrix to compute the search direction.
    Strictly speaking it is not an update strategy, but a full calculation.
    For the approximation the diagonal elements of the hessian are used.
    If these are not vanishing the diagonal matrix is invertible, which also done.

    :param func: cost function used for optimization.
    :param x: x value to evaluate the new hessian at.
    :param xp: The parameter vector used for the last iteration.
    :param last_hessian: the hessian or it's inverse from the last iteration, which is no to be updated.
    :param derivatives: wrapper class for the derivative calculation.
    :param inverse: boolean, indicating whether the hessian matrix should be updated or its inverse.
    :param next_grad: (optional) gradient of the cost function at x
    :param current_grad: (optional) gradient of the cost function at x - step
    :return: new hessian matrix or it's inverse and a boolean indicating whether the hessian matrix is it's inverse or not.
    """
    # for this function it is necessary to make sure that the DerivativeWrapper supplied uses second order derivatives!
    second_order_estimators = np.array(
            [derivatives.derivative(lambda y: permuted_cost_function(func, y, perm), x) for perm in range(x.shape[0])])
    return np.diag(np.reciprocal(second_order_estimators)), True


def get_function(f, vector_like, dimension=None):
    """
    get_function

    @author Dominik Fischer
    @date 2026-01-30

    Transform the function signature to a structure which could be used by the algorithms and the derivative methods.

    :param f: function, which signature is to be transformed
    :param vector_like: boolean, whether the function recieves it's arguments by a vector object
    :param dimension: number of (independent) arguments of this function
    :return: callable of suitable signature to perform operations on it by the implemented minimisers
    """
    assert callable(f)
    # vector arguments could be used directly by the derivatives implementations
    if vector_like:
        return f

    if dimension is None:
        raise ValueError("Dimension of the parameter space must be specified if it is not a vector-like function!")

    # internal helper function to map from a given set of independent arguments to the function to the version accepting just a arguments vector for processing.
    def export_function(x, *args, **kw):
        # TODO: verify the correct operation of this wrapper!
        independent_arguments = []

        # this iteration will slow things down!
        for i in range(dimension):
            independent_arguments.append(x[i])

        return np.vectorize(f)(*independent_arguments, *args, **kw)

    return export_function


class GradientDescent:
    """
    GradientDescent

    Implementation of the gradient descent algorithm for multivariate functions.
    Implements a whole bunch of algorithms.

    Algorithms:
    * newton: (NO IMPLEMENTATION)
    * quasi-Newton: (NO IMPLEMENTATION)
    * gradient descent: gradient descent algorithm as gradient descent method; Implementation is using the negative of the gradient for the direction in which to search for a minimum of the cost function.
    * gradient descent adv.: gradient descent algorithm as a gradient descent method; Implementation is operating quite similar to a quasi-Newton method and modifiny the gradient by a multiplication with the inverse hessian matrix to get the direction in which to search, but here the hessian is only estimated by it's diagonal elements
    * bfgs simple: BFGS algorithm as gradient descent method; implementation is using the identity matrix as a first approximation of the (inverse) hessian.
    * bfgs quasi: BFGS algorithm as gradient descent method; implementation is using only the diagonal elements of the hessian as a first approximation of the (inverse) hessian.
    * bfgs: BFGS algorithm as quasi-newton gradient descent method; implementation is using a here in a first a full hessian calculation like for newtonian method.
    """

    flush_steps = None
    previous_result = None
    previous_cost = None
    collect_step = no_op_collection_helper
    previous_counter = 0

    @classmethod
    def old_init(cls, likelihood_function, dimension, derivative_method="findiff", derivative_class=None, acc_goal=2,
                 deriv_stepsize=0.001, target_acc=1e-12):
        # wrapper to allow the old syntax to be still used, while the new one is already deployed
        return cls.__new__(cls).__init__(likelihood_function, dimension, [], derivative_method, derivative_class,
                                         acc_goal, deriv_stepsize, target_acc)

    def __init__(self, likelihood_function, dimension=0, initial_guess=None, derivative_method="findiff",
                 derivative_class=None, acc_goal=2, deriv_stepsize=0.001, target_acc=1e-12, hesse=None, collect_steps=False, **kwargs):
        """
        @author Dominik Fischer
        @date 2026-01-01

        Intialises the fitter object.
        The fitter will use gradient descent algorithms to compute the minimum value of the likelihood function.
        Most setup os done here, including the initial guess (preferred way), the delegation objects for evaluation of derivatives and hessians.


        :param likelihood_function: callable, likelihood function to be minimised (or in general a function to be minimised)
        :param dimension: number of (independent) arguments of the function to be minimised
        :param initial_guess: intial guess vector to start the minimising iterations (optionally; if not provided it must be provided when calling the minimiser)
        :param derivative_method: name of the method to use to acquire the derivatives (allowed values: 'findiff', 'scipy', 'numdifftools', 'submission')
        :param derivative_class: class to use for performing the derivatives (must not be an initialized object)
        :param acc_goal: parameter for the accuracy goal to choose for the derivative estimators
        :param deriv_stepsize: step size for the lattice used by the numerical derivatives
        :param target_acc: accuracy required for finishing the iteration
        :param kwargs: further keyword base arguments to use.
        """
        # keyword arguments by this function itself:
        # model_names: string, name of the parameters to use for the likelihood function

        # handle initialisation of the cost or likelihood function: If the cost function explicitly uses a parameter
        # array, declare this explicitly. Get required information from the signature or
        # the number of initial arguments of the likelihood function? We want to known which parameters the
        # likelihood function would expect to decide what the dimension of our problem is and whether the function
        # will need to be wrapped for use with the derivative handlers.
        self.parameter_mapping, _, positional_counter, has_ndarray = extract_signature_information(likelihood_function)

        if has_ndarray:
            # could use the provided initial guess to estimate the number of independent arguments required by the
            # likelihood function
            if initial_guess is None or len(initial_guess) == 0:
                assert dimension > 0
                self.dimension = dimension
            else:
                self.dimension = len(initial_guess)
        else:
            self.dimension = positional_counter

        self.likelihood_function = get_function(likelihood_function, has_ndarray, self.dimension)

        # set the handler of the derivatives up (delegate the derivativese to other classes/objects)
        if derivative_class is not None:
            assert issubclass(derivative_class, DerivativeWrapper)
            self.derivative_wrapper = derivative_class(self.dimension, deriv_stepsize, 1, acc_goal, axis=0)
            self.hessian_wrapper = derivative_class(self.dimension, deriv_stepsize, 2, acc_goal, axis=0)
        elif derivative_method is not None:
            match derivative_method:
                case "findiff":
                    self.derivative_wrapper = FindiffWrapper(self.dimension, deriv_stepsize, 1, acc_goal, axis=0)
                    self.hessian_wrapper = FindiffWrapper(self.dimension, deriv_stepsize, 2, acc_goal, axis=0)
                case "scipy":
                    self.derivative_wrapper = DerivativeWrapper(self.dimension, deriv_stepsize, 1, acc_goal, axis=0)
                    self.hessian_wrapper = DerivativeWrapper(self.dimension, deriv_stepsize, 2, acc_goal, axis=0)
                case "numdifftools":
                    self.derivative_wrapper = NumDiffToolsDerivatives(self.dimension, deriv_stepsize, 1, acc_goal,
                                                                      axis=0)
                    self.hessian_wrapper = NumDiffToolsDerivatives(self.dimension, deriv_stepsize, 2, acc_goal, axis=0)
                case "submission":
                    self.derivative_wrapper = FiniteDifferenceDerivatives(self.dimension, deriv_stepsize, 1, acc_goal,
                                                                          axis=0)
                    self.hessian_wrapper = FiniteDifferenceDerivatives(self.dimension, deriv_stepsize, 2, acc_goal,
                                                                       axis=0)
                case _:
                    raise ValueError(f"Unknown derivative method {derivative_method}!")

        # some further objects required for some implementations: delegate the update of the hessian to algorithm
        # specific implementations, which could even reside out of this class
        if hesse is None:
            self._update_hessian = update_hessian
            self._hesse_user_defined = False
        else:
            assert callable(hesse)
            self._update_hessian = HessianDelegate(self, hesse)
            self._hesse_user_defined = True
        # only used to remember, whether the hessian was computed at least once; it's actual value should be the last
        # computed hessian
        self._hessian = None
        self.flush_steps = self.no_op_flush_steps

        # specify internal parameters for the optimisation like target accuracy
        self._eps = target_acc
        self.method_id = None
        self._collect_step_data = collect_steps

        # remember the best minimum found so far
        self.minimum = np.inf
        self.minimum_parameters = None

        # save the state of the parameter estimations for later use
        # should extract these from the likelihood function signature!
        if "model_names" in kwargs:
            model_parameter_names = kwargs["model_names"]
            if isinstance(model_parameter_names, np.ndarray):
                self.parameter_names = model_parameter_names
            elif isinstance(model_parameter_names, Iterable):
                self.parameter_names = np.array(model_parameter_names)
            else:
                raise ValueError(f"Unknown type for model_names: {type(model_parameter_names)}")
        elif len(self.parameter_mapping) > 0 and not has_ndarray:
            self.parameter_names = np.array(self.parameter_mapping)
        else:
            self.parameter_names = np.array([f"x{idx}" for idx in range(self.dimension)])

        # this here should already be handled by the setter of ´parameter_names` property!
        # these instance fields are for remembering the mapping between the index of the parameters in the estimators
        # array and the name of the parameters
        self._pos2var = tuple(self.parameter_names)
        self._var2pos = {k: i for i, k in enumerate(self.parameter_names)}

        # submit the ´initial_guess` to the current state
        if initial_guess is not None and (len(initial_guess) > 0 or (isinstance(initial_guess, np.ndarray) and (
                len(initial_guess.shape) > 0 and initial_guess.shape[0] > 0))):
            self.estimators = np.array(initial_guess)
        else:
            self.estimators = np.full(self.dimension, fill_value=1, dtype=float)
        self.initial_values = self.estimators.copy()
        self.__values = ValueView(self)

        self.generator = default_rng()

    def add_hessian_evaluation(self):
        pass

    # handle the names of the parameters of the likelihood function for correct visualization
    @property
    def parameter_names(self) -> np.ndarray:
        return self._parameter_names

    @parameter_names.setter
    def parameter_names(self, value):
        self._parameter_names = np.array(value)
        # update the internal helper properties to map between the parameter names and the arguments index w.r.t. the estimator
        self._pos2var = tuple(self._parameter_names)
        self._var2pos = {k: i for i, k in enumerate(self._parameter_names)}

    # number of parameters
    @property
    def npar(self) -> int:
        return self.dimension

    # handle the public interface to retrieve the last (available) estimator.
    # this would be the initial guess if no fit was performed.
    @property
    def values(self) -> ValueView:
        return self._values

    @property
    def _values(self) -> ValueView:
        return self.__values

    @_values.setter
    def _values(self, values: Iterable):
        # settings the values of the optimised parameters given as an array
        for idx, param in enumerate(values):
            key = self.parameter_names[idx]
            self.__values[key] = param

    # handle internal properties to independently save the initial guess.
    @property
    def initial_values(self) -> np.ndarray:
        return self._initial_values

    @initial_values.setter
    def initial_values(self, *args):
        # will need to reinit the values of fit parameters mapping
        if isinstance(args[0], np.ndarray):
            self._initial_values = args[0].copy()
        elif isinstance(args[0], Iterable):
            self._initial_values = np.array(args[0])
        else:
            self._initial_values = np.array(args)

    def matrix_inversion(self, matrix):
        """
        matrix_inversion

        @author: Dominik Fischer
        @date: 2026-01-01


        Helper function to calculate the inverse of a matrix.
        This is done by solving a linear system for each column of vector of the inverted matrix.
        This could be speed up by a lu decomposition of the matrix.
        This decomposition could then be used to solve each equation system much faster than using the gaussian elimination method.


        :param matrix: matrix to invert.
        :return: inverse of matrix.
        """
        return matrix_inverse(matrix)

    def first_hessian_fit(self, x, inverse=False, algo="Newton"):
        """
        Helper Function to calculate the very first approximation of the hessian matrix.
        Required for the algorithm implementations which should be seeded by an exact hessian.


        :param x: point in parameter space where the hessian matrix is estimated.
        :param inverse: indicitaing whether the inverse of the hessian matrix should be returned or not.
        :param algo: gradient descent algorithm to use for the optimization problem.
        :return: hessian (or it's inverse) of the cost function at x and the boolean indicating whether the hessian matrix is it's inverse or not.
        """
        # the simplest algorithms are always seeded by the identity matrix.
        if algo in (DescentAlgorithms.BFGS_SIMPLE, DescentAlgorithms.GRADIENT_DESCENT):
            return np.identity(x.shape[0]), inverse

        # more advanced methods could seed with only the diagonal elements of the hessian.
        elif algo in (DescentAlgorithms.BFGS_ADVANCED, DescentAlgorithms.MODIFIED_GRADIENT_DESCENT):
            # want to calculate only the diagonal elements and invert these
            # must use the second-order derivatives here to get approximations for the inverse hesse matrix
            return gradient_descent_advanced_update(self.likelihood_function, x, None, None, self.hessian_wrapper, True,
                                                    None, None)

        # seeding with the full hessian: calculate the hessian and invert if the algorithm requires an inverse hessian
        elif algo in (DescentAlgorithms.BFGS, DescentAlgorithms.NEWTON, DescentAlgorithms.QUASI_NEWTON):
            hessian = self.derivative_wrapper.hessian(self.likelihood_function, x)
            if inverse:
                return self.matrix_inversion(hessian), inverse
            else:
                return hessian, inverse

        else:
            raise ValueError("Unknown algorithm for calculating the first hessian!")

    def gradient_actor(self, x, *args, **kwargs):
        """
        gradient_actor

        @author: Dominik Fischer
        @date: 2026-01-20

        This is a simple helper function to calculate the gradient of the cost function.
        This function was not optimised in any way

        :param x: position at which to evaluate the gradient.
        :param kwargs: further keyword arguments for the mode of operation not further specified here.
        :return: calculated gradient of the cost function in parameter space.
        """
        # further keyword arguments for direct and exclusive use in this method
        # use_derivative: whether to use the derivative calculation or the gradient calculation directly (THIS IS CURRENTLY NOT IN USE)
        # direct_gradient: whether to use the gradient calculation directly or the derivative calculation
        internal_keywords = ["use_derivative", "direct_gradient"]
        forward_kwargs = kwargs.copy()
        for key in internal_keywords:
            if key in forward_kwargs:
                del forward_kwargs[key]
        # further keyword arguments to be forwarded to the derivative calculation
        # TODO: Which arguments are passed on.

        kwargs.setdefault("use_derivative", True)
        kwargs.setdefault("direct_gradient", True)
        if kwargs["direct_gradient"]:
            return self.derivative_wrapper.gradient(self.likelihood_function, x)
        else:
            # will still need to implement the gradient by my self as a fall-back solution!
            # two options:
            #   make sure that the axis argument is accepted by the wrapping classes
            #   implement a suitable method to permute the arguments of the cost function!
            return np.array([self.derivative_wrapper.derivative(self.likelihood_function, x, axis=i) for i in
                             range(self.dimension)])

    def hessian_actor(self, x, last_value, gradient=None, last_gradient=None, algo="newton"):
        """
        hessian_actor

        @author: Dominik Fischer
        @date: 2026-01-01

        Actor for the update and calculation of the hessian matrix and it's inverse.
        This actor is just a wrapper around the actual update and/or calculation of the hessian matrix to control the procedure for the different stages within the optimisation procedure.

        :param x: parameter vector at which the hessian matrix should be calculated.
        :param last_value: parameter at which the hessian matrix was last calculated.
        :param gradient: gradient of the cost function at x if provided (optional parameter)
        :param last_gradient: gradient of the cost function at last_value if provided (optional parameter)
        :param algo: algorithm to use for the optimization problem.
        :return: updated hessian matrix and a boolean indicating whether the hessian matrix is it's inverse or not.
        """
        # calculating the hessian is much more complicated as these are the second-order derivatives across the whole
        # parameter space how to calculate the current hessian will depend on the choice of the algorithm for
        # standard gradient descent this is unused

        # need to specify for return if the inverse is returned or the hessian itself
        if self._hessian is None:
            # this is the first iteration => need to provide an initial hessian or more accurate it's inverse
            # the implementation of this will depend on the choice of the algorithm
            full_hessian, is_inverse = self.first_hessian_fit(x, True, algo)
        else:
            # update the hessian
            # for the form of a implementing see the top of this file!
            derivative_helper = self.hessian_wrapper if algo == DescentAlgorithms.MODIFIED_GRADIENT_DESCENT else self.derivative_wrapper
            full_hessian, is_inverse = self._update_hessian(self.likelihood_function, x, last_value, self._hessian,
                                                            derivative_helper, inverse=True, next_grad=gradient,
                                                            current_grad=last_gradient)

        # it's necessary to store the hessian for further iterations
        self._hessian = full_hessian
        return full_hessian, is_inverse

    def select_hessian_strategy(self, algorithm):
        """
        select_hessian_strategy

        @author: Dominik Fischer
        @date 2026-01-01

        Algorithm to select a algorithm for updating the hessian matrix.
        It just transforms the name of an algorithm into the corresponding function for updating the hessian matrix.

        :param algorithm: string, name of the algorithm to select
        :return: callable of the algorithms update strategy
        """
        # user could define an estimation algorithm; this must not be overriden
        if self._hesse_user_defined:
            return self._update_hessian
        match algorithm:
            case DescentAlgorithms.BFGS_SIMPLE:
                return bfgs_update
            case DescentAlgorithms.BFGS_ADVANCED:
                return bfgs_update
            case DescentAlgorithms.BFGS:
                return bfgs_update
            case DescentAlgorithms.NEWTON:
                return newton_update
            case DescentAlgorithms.QUASI_NEWTON:
                raise NotImplementedError("Quasi-Newton gradient descent is not yet implemented!")
            case DescentAlgorithms.MODIFIED_GRADIENT_DESCENT:
                return gradient_descent_advanced_update
            case _:
                raise ValueError(f"Unknown algorithm {algorithm} for updating the hessian matrix!")

    def calc_direction(self, current_value, previous_optimum, current_direction, current_gradient, next_gradient,
                       **kwargs):
        """
        calc_direction


        @author: Dominik Fischer
        @date: 2026-01-01


        (Internal) Helper function to calculate the direction for the gradient descent algorithm.
        This direction must enclose an angle larger than 90° with the gradient to enforce the descent of function values of the cost function.
        To reduce the calculation times, already calculated gradients can be provided and will be used to simplify the calculation and improve the efficiency.
        The mapping from the computed gradient to the search direction is performed by matrix-vector multiplication.
        The choice for this positive-semidefinite matrix depends on the choice of the optimization algorithm.


        :param current_value: parameter vector for the current iteration of minimization.
        :param previous_optimum: parameter vector for the previous iteration of minimization.
        :param current_direction: search direction used for the last iteration.
        :param current_gradient: gradient used for the last iteration.
        :param next_gradient: gradient to use for the current iteration.
        :param kwargs: further keyword arguments
        :return: new search direction and gradient for the current iteration.
        """
        # internally used keyword arguments
        # algo: string, name of the algorithm to use for the gradient descent algorithm.
        forward_kwargs = kwargs.copy()
        for exclusive_key in ["algo"]:
            if exclusive_key in forward_kwargs:
                del forward_kwargs["algo"]
        # forwarded keyword arguments:
        # anything necessary to evaluate the likelihood function

        # will calculate the direction for the iteration by the matrix-vector product D*grad(L)
        grad = self.derivative_wrapper.gradient(self.likelihood_function, current_value,
                                                **forward_kwargs) if next_gradient is None else next_gradient
        if current_gradient is None:
            current_gradient = self.derivative_wrapper.gradient(self.likelihood_function, previous_optimum,
                                                                **forward_kwargs)

        # require a matrix to determine the step
        if kwargs.get("algo", DescentAlgorithms.GRADIENT_DESCENT) == DescentAlgorithms.GRADIENT_DESCENT:
            direction_matrix = np.identity(self.npar)
        else:
            temp_matrix, inverse = self.hessian_actor(current_value, previous_optimum, grad, current_gradient,
                                                      algo=kwargs.get("algo", DescentAlgorithms.GRADIENT_DESCENT))
            # for quasi-newton procedures the inverse hessian is required; but some implementations may only
            # compute the hessian and not it's inverse!
            if inverse:
                direction_matrix = temp_matrix
            else:
                direction_matrix = self.matrix_inversion(temp_matrix)
        return -1 * direction_matrix @ grad, grad

    def gradient_step_size(self, current_value, direction, gradient, f_value, step_size, **kwargs):
        """
        gradient_step_size

        Helper function to determine the step size for the gradient descent algorithm.
        (Applies to the general class of gradient descent algorithms and not only to *the* gradient descent algorithm)
        But this is not a implementation of a line search. It just wraps around an existing implemention.

        :param current_value: point in parameter space where the gradient descent algorithm is currently at.
        :param direction: direction in which to search for a minimum of the cost function.
        :param gradient: gradient of the cost function at current_value.
        :param f_value: value of the cost function at current_value.
        :param step_size: last step size used for the gradient descent algorithm.
        :param kwargs: further keyword arguments for the line search algorithm (which are directly forwarded to the actual implementation).
        :return: step size, slope of the cost function at the end of the line search and the value of the likelihood function.
        """
        # any keyword argument except for 'last_a' will be forwarded, to declare this explicit with the last step size.
        # these are the keyword arguments needed to evaluate the likelihood function and:
        # algo: str The chosen algorithm to performt the line search (default: backtracking)
        # c1: float The parameter for the armijo condition (default: 1e-4)
        # c2: float The parameter for the wolfe conditions (default: 0.9)
        # amax: float The maximum step size to use (default: 20)
        # maxiter: int The maximum number of iterations to perform until stopping the backtracking procedure without convergence (default: 20)
        # amax: number, defines maximum step size allowed for the iteration
        # maxiter: int, maximum number of iterations allowed for each of the three phases.
        # strong_wolfe: boolean, whether to use the strong wolfe conditions
        # preconditioning: boolean, whether apply the first stage preconditioning of the step size used as the initial value

        if "last_a" in kwargs:
            del kwargs["last_a"]

        step, slope, f_x = line_search_helper(cost_function=self.likelihood_function, x0=current_value, pk=direction,
                                              grad=gradient, f_x0=f_value,
                                              derivative_helper=self.derivative_wrapper, last_a=step_size, **kwargs)

        # handle failure of the line search to get still a new step size
        if step is None:
            print("Step size failure.")
            print(current_value, direction, gradient, f_value, step_size, step, slope, f_x, norm_helper(gradient))
            print(slope, f_x)
            print("\n")
            # select a random step size and retry the line search if it is still failing try exact line search or something else
            new_selection = self.generator.uniform(0.0001, 1)
            step, slope, f_x = line_search_helper(cost_function=self.likelihood_function, x0=current_value,
                                                  pk=direction, grad=gradient, f_x0=f_value,
                                                  derivative_helper=self.derivative_wrapper, last_a=new_selection,
                                                  **kwargs)
            if step is None:
                # FIXME: might be an issue for submission!
                warn(f"The simple line search algorithm failed to converge.")
                new_selection = self.generator.uniform(0.001, 1)

                def export_cost_function(export_step, *args):
                    return self.likelihood_function(current_value + export_step * direction, *args)

                result = dual_annealing(export_cost_function, [(1e-5, 10)], x0=[1])
                print("annealing result:")
                print(result)
                if result.success:
                    f_x = result.fun
                    step = result.x[0]
                    if step <= 1e-5:
                        step = 0.5
                    slope = self.derivative_wrapper.gradient(self.likelihood_function, current_value + step * direction)
                else:
                    warn(f"Annealing search failed by {result.message}")
                    raise ValueError("Line search failed!")

        return step, slope, f_x

    def update_acc_condition(self, last_value, next_value, step_size, grad):
        """
        Check whether the current iteration has reached the desired accuracy.
        :param last_value: parameter vector for the minimum of the last iteration.
        :param next_value: parameter vector for the minimum of the current iteration.
        :param step_size: step size used for the current iteration. (line search)
        :param grad: gradient of the cost function at next_value.
        :return: boolean indicating whether the current iteration has reached the desired accuracy and the next value in parameter space.
        """
        # as a stopping condition it might not be enough to encounter an sufficiently small gradient
        # calculate some goodness measure for the minimum of the cost function
        gof_local_gradient = norm_helper(grad)

        # TODO: this might not be able to handle the difference between local and global minima
        return gof_local_gradient < self._eps, next_value

    # need to collect some information about the visited points in the parameter space for later visulization
    @property
    def collect_step_information(self):
        return self._collect_step_data

    @collect_step_information.setter
    def collect_step_information(self, value):
        self._collect_step_data = value

    @deprecated("Old implementation; please use the new implementation")
    def general_gradient_actor(self, initial_values, args=(), search_args={}, **kwargs):
        """
       general_gradient_actor

       @author: Dominik Fischer
       @date: 2026-01-01

       (Internal) Helper function to calculate the minimum value of the cost function and return these. Iterates
       over the descend steps along the gradient until the desired accuracy is reached. Therefore the gradient is
       calculated at each step and used to determine the search direction. The way the search direction is
       calculated is determined by the algorithm used for the gradient descent algorithm. The general form used here
       could be written as p_k = - D_k * grad(L_k). Where p_k is the search direction vector at the k-th iteration,
       D_k is the k-th iteration matrix, and L_k is the k-th iteration of the cost function.
       In some algorithms like newton, quasi-newton or bfgs algorithms D is the inverse of the hessian matrix or at least a approximation of it.

       The gradient descent algorithms use a diagonal matrix for this operation. In the easiest case the identity
       matrix, such that the search direction is equal to the negative gradient of the cost function. Another
       implementation for gradient descent uses a matrix where the reciprocals of the second-derivatives of the
       cost-functions are on the diagonals.

       :param args: further positional arguments, to be passed to the cost function.
       :param initial_values: initial guess to start the optimization with.
       :param kwargs: further keyword arguments. Keywords not used for configuration of the optimisation procedure WILL be passed on to the provided cost function!
       :return: argument for the minimum value of the coset function and the minimum value of the cost function.
       """
        return self.do_fit(initial_values, args, search_args, **kwargs)


    # TODO: need to inline anything which refers to the arguments of the cost function as this should be set on init of the fitter!
    def do_fit(self, initial_values=None, args=(), search_args={}, **kwargs):
        """
        do_fit

        @author: Dominik Fischer
        @date: 2026-01-01



        (Internal) Helper function to calculate the minimum value of the cost function and return these. Iterates
        over the descend steps along the gradient until the desired accuracy is reached. Therefore the gradient is
        calculated at each step and used to determine the search direction. The way the search direction is
        calculated is determined by the algorithm used for the gradient descent algorithm. The general form used here
        could be written as p_k = - D_k * grad(L_k). Where p_k is the search direction vector at the k-th iteration,
        D_k is the k-th iteration matrix, and L_k is the k-th iteration of the cost function.
        In some algorithms like newton, quasi-newton or bfgs algorithms D is the inverse of the hessian matrix or at least a approximation of it.

        The gradient descent algorithms use a diagonal matrix for this operation. In the easiest case the identity
        matrix, such that the search direction is equal to the negative gradient of the cost function. Another
        implementation for gradient descent uses a matrix where the reciprocals of the second-derivatives of the
        cost-functions are on the diagonals.


        Note:
        ----------
        The general form of the algorithm follows [nonLinOptimierung, Nocedal].
        The allowed keywords for search_args are:
        algo: str The chosen algorithm to performt the line search (default: backtracking)
        c1: float The parameter for the armijo condition (default: 1e-4)
        c2: float The parameter for the wolfe conditions (default: 0.9)
        amax: float The maximum step size to use (default: 20)
        maxiter: int The maximum number of iterations to perform until stopping the backtracking procedure without convergence (default: 20)
        amax: number, defines maximum step size allowed for the iteration
        maxiter: int, maximum number of iterations allowed for each of the three phases.
        strong_wolfe: boolean, whether to use the strong wolfe conditions
        preconditioning: boolean, whether apply the first stage preconditioning of the step size used as the initial value (default: True)


        :param args: further positional arguments, to be passed to the cost function.
        :param initial_values: initial guess to start the optimization with.
        :param kwargs: further keyword arguments. Keywords not used for configuration of the optimisation procedure WILL be passed on to the provided cost function!
        :return: argument for the minimum value of the coset function and the minimum value of the cost function.
        """
        # the available kwargs only for this method are:
        # max_iterations: int, 1000,
        # algo: string, gradient_descent,
        # step_level: string, level of step information collection default: info
        internal_keywords = ["max_iterations", "algo", "step_level"]
        # forwarded keyword arguments are:
        # anything necessary to evaluate the likelihood function
        # forwarding from calc_direction without filtering:
        # algo: string, name of the algorithm to use for the gradient descent algorithm.
        # keyword arguments to evaluate the likelihood function
        forward_kwargs = kwargs.copy()
        for key in internal_keywords:
            if key in forward_kwargs:
                del forward_kwargs[key]

        if self.method_id is None:
            self.method_id = kwargs.get("algo", str(uuid.uuid4()))

        # check the data type of the initial values and convert these to a numpy array/vector if necessary
        if isinstance(initial_values, np.ndarray):
            general_shape = initial_values.shape
        else:
            general_shape = (len(initial_values),)

        # parameter array for the current and the preceding ones arguments for minimal reached function values
        if initial_values is None:
            if self.initial_values is None:
                raise ValueError("No initial values provided!")
            current_iteration = self.initial_values
        elif isinstance(initial_values, np.ndarray):
            current_iteration = initial_values
        else:
            current_iteration = np.array(initial_values)
        last_iteration = None
        current_iteration_step = kwargs.get("last_a", 10)

        # store the state for current search direction and gradient of the cost function
        current_iteration_gradient = np.zeros(general_shape)
        current_iteration_direction = np.zeros(general_shape)

        # store the state for the next gradient if this was already calculated by the line search algortihm used to
        # determine the step size
        # this is not initialised yet, to known whether this is the first iteration or not!
        next_iteration_gradient = None

        # store the state of the stopping conditions
        reached_acc = False

        # might need to set the correct algorithms for the hessian update strategy here
        if kwargs.get("algo", DescentAlgorithms.GRADIENT_DESCENT) != DescentAlgorithms.GRADIENT_DESCENT:
            self._update_hessian = self.select_hessian_strategy(kwargs["algo"])

        # definitions/selection rules for tracking the points in parameter space visited by the algorithm.
        if self.collect_step_information:
            match kwargs.get("step_level", StepInformationLevel.INFO):
                case StepInformationLevel.INFO:
                    self.collect_step = info_collection_helper
                    self.flush_steps = self.info_flush_steps
                case StepInformationLevel.DEBUG:
                    self.collect_step = debug_collection_helper
                    self.flush_steps = self.debug_flush_steps
                case StepInformationLevel.TRACEONLY:
                    self.collect_step = trace_collection_helper
                    self.flush_steps = self.trace_flush_steps
                case _:
                    raise NotImplementedError(f"Unknown step information level {kwargs.get('step_level')}")

            # apply some checks to the signature of the flush function
            assert callable(self.flush_steps), f"flush_steps must be callable, but is {self.flush_steps}"

        # repeat the descending steps until the desired accuracy is reached
        current_iteration_counter = 0
        while not reached_acc and current_iteration_counter <= kwargs.get("max_iterations", 1000):
            current_iteration_counter += 1
            # determine the next search direction and gradient
            current_iteration_direction, current_iteration_gradient = self.calc_direction(current_iteration,
                                                                                          last_iteration,
                                                                                          current_iteration_direction,
                                                                                          current_iteration_gradient,
                                                                                          next_iteration_gradient,
                                                                                          **kwargs)

            # determine the next step size/perform a line search
            try:
                current_iteration_step, next_iteration_gradient, current_minimum = self.gradient_step_size(
                    current_iteration,
                    current_iteration_direction,
                    current_iteration_gradient,
                    self.likelihood_function(current_iteration),
                    current_iteration_step,
                    **search_args, **forward_kwargs)
            except ValueError as e:
                # not used for an actual computation but for a reference value while handling errors in the code.
                # FIXME:
                # this part is just used for debug purposes of the line_search and reference to library
                # implementations for validation. the results are NOT used for the actual implementation!
                scipy_stepsize, n_iterations, n_grads, scipy_minimum, _, scipy_slope = linesearch(
                        f=self.likelihood_function, myfprime=nd.Gradient(self.likelihood_function),
                        xk=current_iteration,
                        pk=current_iteration_direction, gfk=current_iteration_gradient)
                print(n_iterations, n_grads, scipy_minimum, scipy_stepsize)
                print(f"Search failed in iteration {current_iteration} with error")
                raise e
            except AssertionError as e:
                print(current_iteration_counter, current_iteration, current_iteration_direction, current_iteration_gradient,
                      current_iteration_step)
                raise e
            next_value = current_iteration + current_iteration_step * current_iteration_direction
            if current_minimum < self.minimum:
                self.minimum = current_minimum
                self.minimum_parameters = next_value
            last_iteration = current_iteration
            # check the stopping condition for reaching a local minimum, but what about a global minimum to reach?
            reached_acc, current_iteration = self.update_acc_condition(current_iteration, next_value,
                                                                       current_iteration_step,
                                                                       current_iteration_gradient)

            # implementations of some additional metrics for debugging and tracking of the algorithms procedures and progress
            self.collect_step(self, last_iteration, current_iteration, current_iteration_step,
                              current_iteration_gradient,
                              current_iteration_direction, None, None)

        if not reached_acc:
            self.converged = False
        else:
            self.converged = True
        if self.collect_step_information:
            self.flush_steps(history=None, cost=None)
        else:
            self.previous_result = None
            self.previous_cost = None

        self.estimators = current_iteration
        self.f_min = current_minimum
        self.n_iterations = current_iteration_counter
        if current_minimum > self.minimum:
            warn("There was an value of the likelihood function which was larger than the calculated minimum!")

        print(self.estimators)
        return self.values.to_dict(), self.f_min, self.converged

    # these internal helper functions are used to write arrays which tracks the intermediate steps of the algorithm to disk
    # should reduce the used memory by the tracker.
    def debug_flush_steps(self, history, cost=None, trace_divider=500):
        print("Resetting tracker and write it to disk!")
        path_idx = self.previous_counter // trace_divider
        # this part is commented out currently to save a bit of time and efficiency while the algortihms seems to work correctly!
        # np.savetxt(f"step_tracker_{self.method_id}_direction_{path_idx}.txt", self.previous_result[:, 0, :])
        # np.savetxt(f"step_tracker_{self.method_id}_gradient_{path_idx}.txt", self.previous_result[:, 1, :])
        # np.savetxt(f"step_tracker_{self.method_id}_position_{path_idx}.txt", self.previous_result[:, 2, :])
        try:
            # np.savetxt(f"step_tracker_{self.method_id}_cost_{path_idx}.txt", self.previous_cost)
            pass
        except TypeError as e:
            print("Could not save cost history!")
            print(self.previous_cost)
            raise e
        # np.save(f"step_tracker_{self.method_id}_higher_{path_idx}", self.previous_result)
        if history is not None and isinstance(history, np.ndarray):
            self.previous_result = np.array([history])
        if cost is not None:
            self.previous_cost = np.array([cost])

    def info_flush_steps(self, history, cost=None, trace_divider=500):
        print("Resetting tracker and write it to disk!")
        path_idx = self.previous_counter // trace_divider
        # this part is commented out currently to save a bit of time and efficiency while the algortihms seems to work correctly!
        # np.savetxt(f"step_tracker_{self.method_id}_position_{path_idx}.txt", self.previous_result[:, :])
        try:
            # np.savetxt(f"step_tracker_{self.method_id}_cost_{path_idx}.txt", self.previous_cost)
            pass
        except TypeError as e:
            print("Could not save cost history!")
            print(self.previous_cost)
            raise e
        # np.save(f"step_tracker_{self.method_id}_higher_{path_idx}", self.previous_result)
        if history is not None and isinstance(history, np.ndarray):
            self.previous_result = np.array([history])
        if cost is not None:
            self.previous_cost = np.array([cost])

    def trace_flush_steps(self, history, cost=None, trace_divider=500):
        print("Resetting tracker and write it to disk!")
        path_idx = self.previous_counter // trace_divider
        # this part is commented out currently to save a bit of time and efficiency while the algortihms seems to work correctly!
        # np.savetxt(f"step_tracker_{self.method_id}_position_{path_idx}.txt", self.previous_result[:, :])
        # np.save(f"step_tracker_{self.method_id}_higher_{path_idx}", self.previous_result)
        if history is not None and isinstance(history, np.ndarray):
            self.previous_result = np.array([history])

    def no_op_flush_steps(self, history, cost=None, trace_divider=500):
        pass

    def extract_intermediate_results(self):
        """
        extract_intermediate_results


        @author: Dominik Fischer
        @date: 2026-01-20

        Helper function to get the intermediate steps and it's gradients and directional search vectors from the last evaluated optimisation/minimisation procedure.

        :return: numpy ndarray of the intermediate results/steps each one is a tensor like quantity, consisting of the three vectors gradient direction and position in the parameter space.
        """

        # recover intermediate results which were already written to disk?
        return self.previous_result
