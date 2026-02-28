from abc import abstractmethod
from collections.abc import Callable
import numpy as np
import scipy
from findiff import Diff, Gradient
from linalg.matrix import matrix_system_solve
from scipy.special import binom, factorial
import numdifftools as nd


class DerivativeWrapper:
    # The standard implementations in this class may not work (I'm unsure whether the scipy default implementations could handle multi-dimensional derivatives
    def __init__(self, dimension, initial_step, order, acc, axis, additional_grid_point=20, require_grid=False):
        """
        DerivativeWrapper


        @author: Dominik Fischer
        @data: 2026-01-01

        Helper Class to provide some derivative functionality by my self and integrate it with some of the frameworks.
        This class is not a implementation of this functionality but a wrapper around it (a interface).
        Anyway it provides the capability for some special vector analysis features like gradient and hessian.

        Additional it could provide a derivative of arbitrary order w.r.t. to a axis defined on init.
        It maybe implemented a option to choose the axis to derivate at call of the derivative in the future.

        :param dimension: number of arguments of the function to differentiate
        :param initial_step: array of initial step sizes for each dimension
        :param order: order of the derivative to use
        :param acc: accuracy of the derivative calculation for any of the provided methods
        :param axis: axis to differentiate w.r.t.
        :param additional_grid_point: additional number of points to add to the grid for some subclasses to calculate the derivatives
        """
        # must generate a grid and a differentiation object for each dimension!
        self.dimension = dimension
        if dimension > 1 and not isinstance(initial_step, np.ndarray):
            self._stepsize = np.full(dimension, initial_step)
        else:
            self._stepsize = initial_step
        self.axis = axis

        # order could also be a vector for different orders for the different directions
        self._order = order
        self._acc = acc
        self._points = 2 * np.floor((order + 1) / 2) - 1 + acc
        one_sided_point = self._points // 2 + additional_grid_point

        # boundaries to use for the internal grid to evaluate the function values (scaled in units of a step size)
        self.__intern_lower = -one_sided_point
        self.__intern_upper = one_sided_point + 1

        axis_spacings = np.arange(self.__intern_lower, self.__intern_upper, step=1)
        self.spacings = np.zeros((dimension, axis_spacings.shape[0]))
        print(f"Setting spacings to {self.spacings.shape}.")
        for idx in range(dimension):
            # print(f"Handling axis {idx} with spacings {axis_spacings}.")
            self.spacings[idx, :] = axis_spacings
        # print("Trying to update the underlying grid.")
        self.require_grid = require_grid
        self.grid = None
        self.update_grid(0 if dimension == 1 else np.zeros(dimension))
        # print("Grid updated.")

    def update_grid(self, x):
        """

        @author: Dominik Fischer
        @date: 2026-01-07

        Update the grid which is used to evaluate the function values for a point given point to derivate after.
        :param x: point to evaluate the function values at.
        """
        # the grid generation get in trouble if it is necessary to get onto high dimensional grids
        # only perform this here if it is really necessary
        if self.require_grid:
            # adjust the grid to the step sizes and apply the offset by the vector which should be evaluated
            space_grid = []
            for i in range(self.dimension):
                # need to center the grid around the point to evaluate the derivatives at and scale their distances by the step size
                temp = np.array(x[i] + self.spacings[i, :] * self._stepsize[i])
                space_grid.append(temp)

            # apply the new grid
            self.grid = np.meshgrid(*space_grid)

    def derivative(self, func: Callable, x, *remaining_args, axis=None, **kwargs):
        """
        derivative

        @author: Dominik Fischer
        @date: 2026-01-01


        Evaluate the derivative of a given function at a given point numerically.

        :param func: function to differentiate
        :param x: point at which the derivative should be evaluated.
        :param axis: axis to differentiate w.r.t. (not yet implemented)
        :return: value of the derivative at x.
        """
        # make some the checks, to make sure that the callers requirements match the capabilities of this function.
        assert callable(func)

        # this is example implementation can only handle first order derivatives
        assert self._order == 1

        def effective_function(vector, *args):
            if len(vector.shape) == 0:
                return None

        res = scipy.differentiate.derivative(func, x[self.axis], args=remaining_args, order=self._acc,
                                             initial_step=self._stepsize[self.axis])
        return res['df']

    def gradient(self, func: Callable, x, **kwargs):
        """
        gradient


        @author: Dominik Fischer
        @date: 2026-01-01


        Calculate the gradient of a function at a given point.
        At the current point of time these function must be scalar valued and must not be a vector field.

        :param func: scalar function to differentiate
        :param x: point at which the gradient should be evaluated.
        :return: vector field representing the gradient of the function at x.
        """
        final_result = []
        for i in range(self.dimension):
            # must work around the missing gradient in scipy
            reference = np.arange(0, self.dimension)
            mask = reference != i
            remaining_args = x[mask]
            @np.vectorize
            def effective_function(xi, *args):
                effective_args = x.copy()
                if isinstance(xi, np.ndarray) and len(xi.shape) > 0:
                    print(xi, xi.shape, type(xi), len(xi.shape))
                    result = np.zeros(xi.shape[1])
                    for j in range(result.shape[0]):
                        effective_args[i] = xi[0, j]
                        result[j] = func(*effective_args)
                    return result
                else:
                    effective_args[i] = xi
                    return func(*effective_args)

            final_result.append(
                    scipy.differentiate.derivative(effective_function, x[i], args=remaining_args, order=self._acc,
                                                   initial_step=self._stepsize[i])['df'])

        # return scipy.differentiate.gradient(func, x, order=self._acc, initial_step=self._stepsize[self.axis])
        return np.array(final_result)

    def hessian(self, func: Callable, x, **kwargs):
        """
        hessian


        @author: Dominik Fischer
        @date: 2026-01-01

        Calculate the hessian matrix of a function at a given point.

        :param func: function to differentiate
        :param x: point at which the hessian should be evaluated.
        :return: matrix representing the hessian of the function at x.
        """

        def effective_function(array):
            result = []
            print(array)
            print(array.shape)
            for i in range(array.shape[1]):
                args = array[:, i]
                result.append(func(*args))

            return np.array(result)

        hes = scipy.differentiate.hessian(effective_function, x, order=self._acc,
                                          initial_step=self._stepsize[self.axis])
        return hes['ddf']

    def curl(self, func: Callable, x, **kwargs):
        raise NotImplementedError(
                "This method is not yet implemented! In particular acting on vector fields is not supported yet.")

    def divergence(self, func: Callable, x, **kwargs):
        raise NotImplementedError(
                "This method is not yet implemented! In particular acting on vector fields is not supported yet.")

    def laplace(self, func: Callable, x, **kwargs):
        """
        laplace

        @author: Dominik Fischer
        @date: 2026-01-07

        A very simple implementation of the laplace operator acting on a *scalar* field.
        Please note that vector fields are not supported yet.
        This simple implementations will use other implementations in this class to first calculate the hessian matrix
        and then use the fact that the trace of the hessian is just the action of the laplace operator on the field.
        The implementation maybe easy but it will produce a lot of overhead.

        :param func: field to apply the laplace operator on.
        :param x: point at which the laplace operator should be evaluated.
        :return: result of the laplace operator acting on the field at x.
        """
        return np.trace(self.hessian(func, x))


# wrapper class for findiff to be used to generate a lattice and reuse it multiple times
# this wrapper class uses a foreign framework not written by us and should not be used for submission code.
class FindiffWrapper(DerivativeWrapper):
    def __init__(self, dimension, initial_step, order, acc, axis):
        super().__init__(dimension, initial_step, order, acc, axis, require_grid=True)

        # init the findiff objects use to perform the differentiation w.r.t to a given axis
        # make sure to not use a particular grid for the definitions as such a grid will depend on the particular point
        # to evaluate the derivatives at
        self._diff = Diff(self.axis, self._stepsize, acc=self._acc) ** self._order
        self._gradient = Gradient(h=self._stepsize, acc=self._acc)
        hessian_definition_vector = np.array([Diff(i, self._stepsize, acc=self._acc) for i in range(dimension)])
        self._hessian = hessian_definition_vector @ hessian_definition_vector.T

    def derivative(self, func: Callable, x, *arguments, axis=None, **kwargs):
        # but will need to adjust the step size and therefore the grid anyway
        assert callable(func)
        assert axis is None
        self.update_grid(x)

        def export_function(*args):
            assert args[0].shape == args[1].shape
            effective_args = np.array(args).T
            return func(effective_args, *arguments, **kwargs)

        # get the function values at the grid points; using stencils as we are only interested in the value of the
        # derivative at exactly one point on the grid
        f_values = export_function(*self._grid)
        return self._diff.stencils(f_values.shape).apply(f_values, x)

    def gradient(self, func: Callable, x, *arguments, **kwargs):
        # but will need to adjust the step size and therefore the grid anyway
        assert callable(func)
        self.update_grid(x)

        def export_function(*args):
            assert args[0].shape == args[1].shape
            effective_args = np.array(args).T
            return func(effective_args, *arguments, **kwargs)

        # get the function values at the grid points
        f_values = export_function(*self._grid)
        return self._gradient.stencils(f_values.shape).apply(f_values, x)

    def hessian(self, func: Callable, x, *arguments, order=None, acc=None, **kwargs):
        # make sure that the function is a callable
        assert callable(func)
        self.update_grid(x)

        def export_function(*args):
            assert args[0].shape == args[1].shape
            effective_args = np.array(args).T
            return func(effective_args, *arguments, **kwargs)

        # get the function values at the grid points
        f_values = export_function(*self._grid)
        # TODO: not tested yet!
        return apply_derivatives(self._hessian, f_values, x)


@np.vectorize
def apply_derivatives(deriv, f_values, x):
    return deriv.stencils(f_values.shape).apply(f_values, x)


# this implementation of the Wrapper class is written by my self to be used for exam submission
class FiniteDifferenceDerivatives(DerivativeWrapper):
    def calculate_shifts(self, offsets):
        return np.array([offsets * step for step in self._stepsize])

    def __init__(self, dimension, initial_step, order, acc, axis):
        super().__init__(dimension, initial_step, order, acc, axis)

        # calculate the coefficients for the derivatives and the hessian
        self.derivative_coefficients = advanced_coefficients(order, acc)
        n_coeffs = (len(self.derivative_coefficients) - 1) // 2
        self._offsets = np.arange(-n_coeffs, n_coeffs + 1)
        self._shifts = self.calculate_shifts(self._offsets)
        self.dimension = dimension

        # calculate the coefficients for the hessian
        # these are all first order coefficients but they need to be arranged in a specific way
        # could reuse the same coefficients on any calculation
        # additionally calculate an array of step sizes for any case (This might have the form of an matrix)
        # calculating a step vector for any of the elements of the hessian matrix would be quite expensive
        self.hessian_coordinate_coefficients = advanced_coefficients(1, acc)
        hessian_n_coeffs = (len(self.hessian_coordinate_coefficients) - 1) // 2
        self.hessian_coordinate_coefficients_offsets = np.arange(-hessian_n_coeffs, hessian_n_coeffs + 1)
        # will need to pay attention to the different step sizes depending on the direction of the partial derivative
        self.hessian_coordinate_shifts = self.calculate_shifts(self.hessian_coordinate_coefficients_offsets)

        # also calculate the second order derivatives as this are simpler
        self.hessian_second_order_coefficients = advanced_coefficients(2, acc)
        hessian_second_order_n_coeffs = (len(self.hessian_second_order_coefficients) - 1) // 2
        self.hessian_second_order_coefficients_offsets = np.arange(-hessian_second_order_n_coeffs,
                                                                   hessian_second_order_n_coeffs + 1)
        self.hessian_second_order_shifts = self.calculate_shifts(self.hessian_second_order_coefficients_offsets)

    def hessian(self, func: Callable, x, **kwargs):
        assert callable(func)
        # wont require grid update here

        # for any element of a hessian matrix calculate the numerical value by using the second-order derivatives
        result = np.zeros((self.dimension, self.dimension))
        for i, j in np.indices(self.hessian_idx_grid.shape):
            result[i, j] = self._internal_second_order_mixed_derivative(func, x, (i, j))
        return result

    def gradient(self, func: Callable, x, **kwargs):
        assert callable(func)
        # wont require grid update here

        # please note, that mixed order derivatives could not be handled
        return np.array([self._internal_derivative(func, x, perm) for perm in range(self.dimension)])

    def derivative(self, func: Callable, x, axis=None, **kwargs):
        assert callable(func)
        assert axis is None
        # wont require grid update here

        # please note, that mixed order derivatives could not be handled
        return self._internal_derivative(func, x, self.axis if axis is None else axis)

    def _internal_derivative(self, func, x, axis):
        # perhaps it would be more optimal if using predefined step vector and adding these directly instead of
        # these expensive calculations at any point; but in this case it wont be possible to change the axis without reinit of the whole class
        f_values = np.array([shift_function_call(func, x, cs, axis) for cs in self._shifts])
        weights_sum = np.sum(self.derivative_coefficients * f_values)

        # in the last step we will need to use a effective step size depending on the subtraction scheme which
        # depends on the order of the derivative
        stepping = np.where(self._order % 2 == 0, 1, 2) * self._stepsize[axis]
        return weights_sum / (stepping ** self._order)

    def _internal_second_order_mixed_derivative(self, func, x, axis):
        # axis should here be tuple containing the two axis to use
        first_axis, second_axis = axis
        if first_axis == second_axis:
            # in this case we could use directly second order derivatives and the implementation is exactly like the standard way defined above
            f_values = np.array(
                    [shift_function_call(func, x, cs, axis) for cs in self.hessian_second_order_shifts[first_axis]])
            weights_sum = np.sum(self.hessian_second_order_coefficients * f_values)
            stepping = np.where(self._order % 2 == 0, 1, 2) * self._stepsize[first_axis]
            return weights_sum / (stepping ** 2)

        # last manage the mixed axis case of the derivatives
        # must combine all the offsets of one axis with all the offsets from the second axis
        # as if we would apply the derivative to previously done first-order derivative
        weights_sum = 0
        for first_cs in self.hessian_coordinate_shifts[first_axis]:
            f_values = np.array(
                    [shift_function_call_multiple(func, x, [first_cs, cs], [first_axis, second_axis]) for cs in
                     self.hessian_coordinate_shifts[second_axis]])
            weights_sum += np.sum(self.hessian_coordinate_coefficients * f_values)

        stepping = np.where(self._order % 2 == 0, 1, 2) * self._stepsize[first_axis] * self._stepsize[second_axis]
        return weights_sum / stepping


def shift_function_call(func, x, shift, idx):
    actual_x = x.copy()
    actual_x[idx] += shift
    return func(actual_x)


def shift_function_call_multiple(func, x, shift, idx):
    actual_x = x.copy()
    for param in np.vstack([idx, shift]).T:
        index = param[0]
        shifting = param[1]
        actual_x[index] += shifting

    return func(actual_x)


def basic_coefficients(order):
    indices = np.arange(np.where(order % 2 == 0, order + 1, order + 2))
    offsets = np.where(order % 2 == 0, indices, indices - ((order + 1) // 2))
    s = (order - 1) // 2
    # only for forward ones
    # return (-1) ** (order + indices) * binom(order, indices)
    return np.where(order % 2 == 0, (-1) ** indices * binom(order, indices),
                    (1. / 2.) * np.where(np.abs(offsets) < (order - 1) // 2,
                                         (-1) ** (offsets + s + 1) * 2 * offsets * binom(order, offsets + s) / (
                                                 offsets + s + 1),
                                         np.sign(offsets) * np.where(np.abs(offsets) == s + 1, 1, -2 * s)))


def advanced_coefficients(deriv, acc):
    # for leading order we could use the analytical formula
    if acc <= 2:
        return basic_coefficients(deriv)
    # will use a linear equation to solve numerical for calculating the coefficients
    num_central_coefficients = 2 * np.floor((deriv + 1) / 2) - 1 + acc
    num_side_coefficients = num_central_coefficients // 2
    offsets = np.arange(-num_side_coefficients, num_side_coefficients + 1)
    b = np.zeros(len(offsets))
    b[deriv] = factorial(deriv)
    A = np.zeros((len(offsets), len(offsets)))
    A[0] = np.full(len(offsets), 1)
    for order in range(1, len(offsets)):
        for idx, weight in enumerate(offsets):
            A[order][idx] = weight ** order

    numpy_reference_result = np.linalg.solve(A, b)
    result = matrix_system_solve(A, b)
    print(f"Validated by numpy implementation: {np.allclose(numpy_reference_result, result)}")
    return result


class NumDiffToolsDerivatives(DerivativeWrapper):
    def __init__(self, dimension, initial_step, order, acc, axis, additional_grid_point=20):
        super().__init__(dimension, initial_step, order, acc, axis)
        print("Init numdifftools wrapper")

    def derivative(self, func: Callable, x, axis=None, **kwargs):
        """
        derivative

        @author: Dominik Fischer
        @date: 2026-01-01


        Evaluate the derivative of a given function at a given point numerically.

        :param func: function to differentiate
        :param x: point at which the derivative should be evaluated.
        :param axis: axis to differentiate w.r.t. (not yet implemented)
        :return: value of the derivative at x.
        """
        # make some the checks, to make sure that the callers requirements match the capabilities of this function.
        assert callable(func)

        if self.axis == 0 and self.dimension == 1:
            return nd.Derivative(func, n=self._order)(x)
        else:
            direction = np.zeros(self.dimension)
            direction[self.axis] = 1
            return nd.directionaldiff(func, x, direction, n=self._order)

    def gradient(self, func: Callable, x, **kwargs):
        """
        gradient


        @author: Dominik Fischer
        @date: 2026-01-01


        Calculate the gradient of a function at a given point.
        At the current point of time these function must be scalar valued and must not be a vector field.

        :param func: scalar function to differentiate
        :param x: point at which the gradient should be evaluated.
        :return: vector field representing the gradient of the function at x.
        """
        return nd.Gradient(func)(x)

    def hessian(self, func: Callable, x, **kwargs):
        """
        hessian


        @author: Dominik Fischer
        @date: 2026-01-01

        Calculate the hessian matrix of a function at a given point.

        :param func: function to differentiate
        :param x: point at which the hessian should be evaluated.
        :return: matrix representing the hessian of the function at x.
        """
        return nd.Hessian(func)(x)
