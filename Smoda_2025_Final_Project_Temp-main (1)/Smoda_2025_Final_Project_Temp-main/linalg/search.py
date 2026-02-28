import warnings
from typing import Callable

import numdifftools as nd
import numpy as np
import scipy
from numdifftools import Gradient
from scipy.optimize import Bounds, dual_annealing

from derivatives.FindiffWrapper import DerivativeWrapper


def bypass_wolfe_condition(step):
    return True


track_iterator = -10


def adjust_alpha(upper_bound, lower_bound, x0, pk, grad, grad_xl, f_x0, f_xl, f_xu, derivative_delegate, tau, use_wolfe,
                 f, **kwargs):
    """
    Helper function to adjust the step size after finding an initial step size which violates the armijo condition as this condition
    could always be fulfilled by a very small step size. But these very small step sizes wont necessarily lead to convergences of the line search.

    :param upper_bound: upper boundary for the step size
    :param lower_bound: lower boundary for the step size
    :param x0: point to start the line search from
    :param pk: current search direction to use
    :param grad: gradient of the cost function at x0
    :param grad_xl: gradient of the cost function at the currently lower bound of the step size
    :param f_x0: cost function value at x0
    :param f_xl: cost function value at the currently lower bound of the step size
    :param f_xu: cost function value at the currently upper bound of the step size [UNUSED, but could be used to refine the choice of the optimal step size]
    :param derivative_delegate: callable to delegate the calculation of the gradient of the cost function to
    :param tau: parameter to change the step size
    :param use_wolfe: whether to use the wolfe conditions
    :param f: cost function itself (In our case it should be a likelihood function or nll)
    :param kwargs: further keyword arguments
    :return: step size, gradient, value of the cost function at the end of the step
    """
    global track_iterator
    track_iterator = 0
    # now adjust the steps
    # will choose for this task a value with in the interval bounds to prevent getting much to small step sizes
    # the lower bound is the current step size minus a factor tau times the step size to be decreased
    upper = upper_bound
    lower = lower_bound

    # perhaps this here should be updated?
    slope_parameter_0 = np.dot(pk, grad)

    # iterate the adjustments to the stepsize
    for _ in range(kwargs["maxiter"]):
        # setup to the interval limiting the step sizes
        interval = upper - lower

        # new step should be in between the interval
        # could this step be further optimised?
        # could further optimise it, when approximating the cost function at the currently lower bound of the step sizes
        # as a quadratic functions and using analytical formulas to derive the minimum of this approximation.
        gamma = lower_bound + tau * interval

        # now perform some checks on the updated step size
        cost = f(x0 + gamma * pk)
        slope = derivative_delegate(x0 + gamma * pk)
        slope_parameter = np.dot(pk, slope)

        # violation of the armijo condition will require an further update
        if cost > f_x0 + gamma * kwargs["c1"] * slope_parameter_0 or cost > f_xl:
            upper = gamma
            f_xu = cost
        else:
            # armijo condition is fulfilled => check for the wolfe conditions if necessary
            if use_wolfe and False:
                if kwargs.get("strong_wolfe", False):
                    if np.abs(slope_parameter) <= -kwargs["c2"] * slope_parameter_0:
                        # all fine, return the results
                        return gamma, slope, cost
                else:
                    if -slope_parameter <= -kwargs["c2"] * slope_parameter_0:
                        # all refine return the results
                        return gamma, slope, cost

                # if not need to make sure we are still "propagating" in the right direction (of a minimum)
                if slope_parameter * interval >= 0:
                    # might propagating in the wrong direction of a maximum
                    upper = lower
                    f_xu = f_xl
            else:
                return gamma, slope, cost

            lower = gamma
            f_xu = cost
            grad_xl = slope

    warnings.warn(
            f"Could not find a suitable step size in the interval [{lower_bound},{upper_bound}]. The last scalar slope was {slope_parameter}.")
    print(
            f"The current step is {gamma}; interval: {lower}-{upper}; general slope: {slope_parameter} from {slope_parameter_0}; slope={slope} from {grad}; L={cost};{use_wolfe}")
    print(kwargs)
    print(track_iterator)
    print(cost > f_x0 + gamma * kwargs["c1"] * slope_parameter_0)
    print(cost >= f_xl)
    print(cost, f_xl, f_x0)
    return None, upper_bound, cost


def backtracking_line_search(cost_function, x0, pk, grad, f_x0, derivative_delegate, tau=0.5, use_wolfe=True, **kwargs):
    global track_iterator
    slope = None
    # implementation of a backtracking line search algorithm applying the armijo condition
    try:
        local_slope = np.dot(pk, grad)
    except ValueError as e:
        print("The parameters for the line search could not be calculated.")
        print(grad)
        print(pk)
        print(grad.shape, pk.shape)
        raise e

    if local_slope >= 0:
        print(f"The local slope {local_slope} violated the assumption of negative definity.")
        print(pk, grad)
        raise ValueError("The local slope is not negative definite.")

    # define a parameter for current step size to use
    gamma = min(kwargs["last_a"], kwargs["amax"])

    # to do the adjustment of the step size a bit smarter use a lower bound
    lower = 0
    cost_0 = f_x0
    slope_0 = grad
    cost = cost_0
    for i in range(kwargs["maxiter"]):
        cost = cost_function(x0 + gamma * pk)
        slope = derivative_delegate(x0 + gamma * pk)
        slope_parameter = np.dot(pk, slope)

        # first make sure the armijo condition is fulfilled
        if cost > cost_0 + gamma * kwargs["c1"] * local_slope or (cost >= f_x0 and i > 0):
            # adjust by decreasing the step size gamma
            track_iterator = i
            gamma, slope, cost = adjust_alpha(upper_bound=gamma, lower_bound=lower, x0=x0, pk=pk, grad=grad,
                                              grad_xl=grad, f_x0=f_x0, f_xl=f_x0, f_xu=cost,
                                              derivative_delegate=derivative_delegate, tau=tau, use_wolfe=use_wolfe,
                                              f=cost_function, **kwargs)
            break

        # perhaps we should ignore all the other parts first and get upto a point where the armijo condition is violated.
        if use_wolfe:
            # armijo fulfilled => check for the wolfe conditions
            if kwargs.get("strong_wolfe", False):
                if abs(np.dot(pk, slope)) <= -kwargs["c2"] * local_slope:
                    break

            else:
                if -np.dot(pk, slope) <= -kwargs["c2"] * local_slope:
                    break

            # What to do if they are not fulfilled?
            if slope_parameter >= 0:
                gamma, slope, cost = adjust_alpha(lower, gamma, x0, pk, grad, slope, f_x0, cost, f_x0,
                                                  derivative_delegate, tau, use_wolfe, cost_function, **kwargs)
                break

        # need to increase the step size gamma
        gamma = min(gamma / tau, kwargs["amax"])
    else:
        # handle the non-converging case
        warnings.warn("Could not find a suitable step size in the interval.")
        return slope, gamma, cost
    return slope, gamma, cost


def line_search_helper(cost_function: Callable, x0, pk, grad, f_x0, derivative_helper: DerivativeWrapper = None,
                       myfprime=None, bounds=None, use_wolfe=True, tau=0.5, args=(), kw={}, **kwargs):
    """
    @author: Dominik Fischer
    @date: 2026-01-01
    Generous function to perform a line search for a minimum of a cost function.
    It implements numerous algorithms to perform the line search.
    Only the backtracking algorithm is implemented by our selfs.
    For the other algorithms see scipy's documentation, as the scipy module is used for them.
    :param cost_function: function onto which the line search for a minimum is performed
    :param x0: current point in the iteration of the minimizer (parameter space)
    :param pk: direction to search for a minimum
    :param grad: gradient of the cost function at x0 in the parameter space
    :param f_x0: function value at x0 of the cost function
    :param derivative_helper: helper/wrapper class for the derivative calculation
    :param myfprime: analytical expression for the first derivative of the cost function if available
    :param bounds: boundary values for the step size
    :param use_wolfe: boolean, whether to use the wolfe conditions for the backtracking line search
    :param tau: parameter for reduction of the step size using backtracking algorithms
    :param kwargs: further keyword arguments for the line search algorithm
    :param algo: string, name of the line search algorithm to use (for submission only 'backtracking' is to use)
    :param c1: parameter for the armijo condition
    :param c2: parameter for the wolfe conditions
    :param amax: maximum step size to use
    :param maxiter: maximum number of iterations to perform until stopping the backtracking procedure wihtout convergence
    :param last_a: initial step size to use e.g. from a previous iteration
    :param extra_condition: some of the implemented line search algorithms allow for an extra condition to apply.
    :return: step size, slope of the cost function at the end of the line search. These values could only be provided if the line search converges and it was possible to calculate the numerical or analytical derivative. If these conditions are not met, the values are None.
    """
    # the internal used keyword arguments are
    # algo: str The chosen algorithm to performt the line search (default: backtracking)
    # c1: float The parameter for the armijo condition (default: 1e-4)
    # c2: float The parameter for the wolfe conditions (default: 0.9)
    # amax: float The maximum step size to use (default: 20)
    # maxiter: int The maximum number of iterations to perform until stopping the backtracking procedure without convergence (default: 20)
    # last_a: float The initial step size to use e.g. from a previous iteration (default: 10)
    kwargs.setdefault("algo", "backtracking")
    # parameters for the armijo condition
    kwargs.setdefault("c1", 1e-4)
    kwargs.setdefault("c2", 0.9)
    kwargs.setdefault("amax", 20)
    kwargs.setdefault("maxiter", 20)
    kwargs.setdefault("last_a", 10)

    forward_kwargs = kwargs.copy()
    if "algo" in forward_kwargs:
        del forward_kwargs["algo"]
    if "last_a" in forward_kwargs:
        del forward_kwargs["last_a"]
    if "strong_wolfe" in forward_kwargs:
        del forward_kwargs["strong_wolfe"]
    assert 0 < tau < 1

    # define a lambda to perform derivatives of the function if necessary
    if myfprime is not None:
        assert callable(myfprime)
        derivative_handler = myfprime
    else:
        if derivative_helper is None:
            derivative_handler = Gradient(cost_function)
            print("for some reason the derivative handler is missing!")
        else:
            def export_derivative(x, *args, **kwargs):
                return derivative_helper.gradient(cost_function, x)

            derivative_handler = export_derivative

    # FIXME: needs to be removed, such that operation without ever using library implementations is possible
    derivative_handler = nd.Gradient(cost_function)

    cost = f_x0
    alpha = None
    slope = None
    match (kwargs.get("algo", "backtracking")):
        case "scipy":
            # use scipy's line search function'
            alpha, _, _, _, cost, slope = scipy.optimize.line_search(f=cost_function, myfprime=derivative_handler,
                                                                     xk=x0, pk=pk, gfk=grad, old_fval=f_x0, args=args,
                                                                     **forward_kwargs)

        case "annealing":
            # require bounds for the annealing schedule
            if bounds is None:
                bounds = Bounds(0, kwargs["amax"])
            else:
                assert isinstance(bounds, Bounds) or isinstance(bounds, tuple)

            search_function = lambda x: cost_function(x0 + x * pk)

            # make sure there are no further keyword arguments passed to the annealing function, which could perhaps confuse it.
            annealing_kwargs = kwargs.copy()
            del annealing_kwargs["last_a"]
            del annealing_kwargs["amax"]
            del annealing_kwargs["algo"]
            del annealing_kwargs["c1"]
            del annealing_kwargs["c2"]

            # no_local_search=True is necessary as this minimizer is only used for the line search to get the optimum step size
            result = dual_annealing(search_function, bounds, no_local_search=True, x0=kwargs['last_a'], **kwargs)
            if not result.success:
                print(f"Annealing failed with message: {result.message}")
                alpha = None
            else:
                alpha = result.x[0]

            # this method does not return the slope at the endpoint
            slope = None

        case "backtracking":
            slope, alpha, cost = backtracking_line_search(cost_function, x0, pk, grad, f_x0, derivative_handler,
                                                          tau=tau, use_wolfe=use_wolfe, **kwargs)

        case _:
            alpha = None
            slope = None
            raise ValueError(f"Unknown line search algorithm: {kwargs['algo']}")

    # if the slope is not calculated yet but there is a converged step size, calculate the step size
    if slope is None and alpha is not None:
        slope = derivative_handler(x0 + alpha * pk)
    if cost is None and alpha is not None:
        cost = cost_function(x0 + alpha * pk)
    return alpha, slope, cost
