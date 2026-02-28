import numpy as np
from optimize.GradientDescent import GradientDescent, DescentAlgorithms, collecting_step_information


class RawGradientDescent(GradientDescent):
    def __init__(self, likelihood_function, dimension, derivative_method="numdifftools", derivative_class=None,
                 acc_goal=2,
                 deriv_stepsize=0.001, target_acc=1e-12):
        super().__init__(likelihood_function, dimension, [], derivative_method, derivative_class, acc_goal,
                         deriv_stepsize, target_acc)
        self.method_id = "raw_gradient_descent"

    # not necessary to override `first_hessian_fit` as it is not needed here
    def first_hessian_fit(self, x, inverse=False, algo="Newton"):
        return np.identity(x.shape[0]), True

    # Is `gradient_actor´ needed here?

    # is ´hessian_actor` needed here?

    # will need to override the direction searcher to simplify things a bit!
    def calc_direction(self, current_value, last_parameter, current_direction, current_gradient, next_gradient,
                       **kwargs):
        # will calculate the direction for the iteration by the matrix-vector product D*grad(L)
        grad = self.derivative_wrapper.gradient(self.likelihood_function, current_value, **kwargs) if next_gradient is None else next_gradient

        # the matrix specifing the direction to search for is trivial in this case
        direction_matrix = np.identity(self.npar)
        # apply the general procedure for gradient descent operations!
        return -1 * direction_matrix @ grad, grad

    # not necessary to override ´gradient_step_size`as the line search is used by any of the implementations and not only by a particular one

    # not necessary to override úpdate_acc_condition`as it is the same for any of the implementations

    def general_gradient_actor(self, initial_values, **kwargs):
        if "algo" in kwargs:
            del kwargs["algo"]

        return super().general_gradient_actor(initial_values, algo=DescentAlgorithms.GRADIENT_DESCENT, **kwargs)


class GradientDescent(GradientDescent):
    def __init__(self, likelihood_function, dimension, derivative_method="numdifftools", derivative_class=None,
                 acc_goal=2,
                 deriv_stepsize=0.001, target_acc=1e-12):
        super().__init__(likelihood_function, dimension, [], derivative_method, derivative_class, acc_goal,
                         deriv_stepsize, target_acc)
        self.method_id = "gradient_descent (with advancements)"

    # not necessary to override `first_hessian_fit` as it is not needed here
    def first_hessian_fit(self, x, inverse=False, algo="Newton"):
        return np.identity(x.shape[0]), True

    # Is `gradient_actor´ needed here?

    # is ´hessian_actor` needed here?

    # will need to override the direction searcher to simplify things a bit!
    def calc_direction(self, current_value, last_parameter, current_direction, current_gradient, next_gradient,
                       **kwargs):
        # will calculate the direction for the iteration by the matrix-vector product D*grad(L)
        # optimise efficiency by performing only a minimal number of gradient evaluations
        grad = self.derivative_wrapper.gradient(self.likelihood_function, current_value, **kwargs) if next_gradient is None else next_gradient
        if current_gradient is None:
            current_gradient = self.derivative_wrapper.gradient(self.likelihood_function, last_parameter, **kwargs)

        # get some more accurate results with approximating the diagonal elements by the reciprocal second-order derivatives
        # this implementation is computational not the best idea as it evaluates not only the diagonal elements of the hessian but the whole hessian instead of only using the diagonal elements.
        diagonal_derivatives = np.array(
                np.diag(self.derivative_wrapper.hessian(self.likelihood_function, current_value)))
        for idx, val in enumerate(diagonal_derivatives):
            if val > 1e-12:
                diagonal_derivatives[idx] = 1 / val

        direction_matrix = np.diag(diagonal_derivatives)

        # apply the general procedure for gradient descent operations!
        return -1 * direction_matrix @ grad, grad

    # not necessary to override ´gradient_step_size`as the line search is used by any of the implementations and not only by a particular one

    # not necessary to override úpdate_acc_condition`as it is the same for any of the implementations

    def general_gradient_actor(self, initial_values, **kwargs):
        if "algo" in kwargs:
            del kwargs["algo"]

        return super().general_gradient_actor(initial_values, algo=DescentAlgorithms.GRADIENT_DESCENT, **kwargs)
