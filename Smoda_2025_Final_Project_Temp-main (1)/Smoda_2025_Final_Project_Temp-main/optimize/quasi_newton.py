import logging

import numpy as np

from optimize.GradientDescent import GradientDescent, bfgs_update, gradient_descent_advanced_update, DescentAlgorithms

logger = logging.getLogger(__name__)

class Template(GradientDescent):
    def __init__(self, likelihood_function, derivative_method="numdifftools", derivative_class=None, acc_goal=2, ):
        super().__init__(likelihood_function, 1, [], derivative_method, derivative_class, acc_goal)

    # not necessary to override ´matrix_inversion`

    # not necessary to override ´first_hessian_fit`

    # not necessary to override ´gradient_actor`

    # not necessary to override ´hessian_actor`

    # not necessary to override ´select_hessian_strategy`

    # not necessary to override `calc_direction`

    # not necessary to override `gradient_step_size`

    # not necessary to override `update_acc_condition`


# very simple BFGS optimizer. Starting from the identity matrix for the inverse hessian matrix
class BFGSOptimizer(GradientDescent):
    def __init__(self, likelihood_function, dimension, derivative_method="numdifftools", derivative_class=None,
                 acc_goal=2,
                 deriv_stepsize=0.001, target_acc=1e-12):
        super().__init__(likelihood_function, dimension, [], derivative_method, derivative_class, acc_goal,
                         deriv_stepsize, target_acc)
        self.hessian = None
        self.method_id = "BFGSOptimumClass"

    # not necessary to override ´matrix_inversion`

    # not necessary to override ´first_hessian_fit`

    # not necessary to override ´gradient_actor`

    # not necessary to override ´hessian_actor`

    # not necessary to override ´select_hessian_strategy`

    def calc_direction(self, current_value, previous_optimum, current_direction, current_gradient, next_gradient,
                       **kwargs):
        # will calculate the direction for the iteration by the matrix-vector product D*grad(L)
        grad = self.derivative_wrapper.gradient(self.likelihood_function, current_value, **kwargs) if next_gradient is None else next_gradient
        if current_gradient is None:
            current_gradient = self.derivative_wrapper.gradient(self.likelihood_function, previous_optimum, **kwargs)

        # will use a BFGS update strategy for the inverse hessian matrix and not the hessian itself.
        if self.hessian is None:
            logger.debug(f"The init gradient iss: {grad} at {current_value}")
            full_hessian = np.identity(self.dimension)
        else:
            full_hessian, _ = bfgs_update(func=self.likelihood_function, x=current_value, xp=previous_optimum, last_hessian=self.hessian,
                                          derivatives=self.derivative_wrapper, inverse=True, next_grad=grad,
                                          current_grad=current_gradient)
        self.hessian = full_hessian
        return -1 * self.hessian @ grad, grad

    # not necessary to override `gradient_step_size`

    # not necessary to override `update_acc_condition`

    def general_gradient_actor(self, initial_values, args=(), **kwargs):
        if "gradient_descent" in kwargs:
            del kwargs["gradient_descent"]

        # sneaking with the arguments to avoid reimplementing helper functions of the more advanced wrapper class which serves here as a super class
        return super().general_gradient_actor(initial_values, algo=DescentAlgorithms.GRADIENT_DESCENT, **kwargs)


# BFGS optimizer using at the very first step a inverse hessian of the inverted second order derivatives only!
class AdvancedBFGS(BFGSOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hessian = None
        self.method_id = "AdvancedBFGSClass"

    # not necessary to override ´matrix_inversion`

    # not necessary to override ´first_hessian_fit`

    # not necessary to override ´gradient_actor`

    # not necessary to override ´hessian_actor`

    # not necessary to override ´select_hessian_strategy`

    def calc_direction(self, current_value, previous_optimum, current_direction, current_gradient, next_gradient,
                       **kwargs):
        if self.hessian is None:
            self.hessian, is_inverse = gradient_descent_advanced_update(self.likelihood_function, current_value, None,
                                                                        None,
                                                                        self.hessian_wrapper, inverse=True,
                                                                        next_grad=None,
                                                                        current_grad=None)
            if not is_inverse:
                self.hessian = self.matrix_inversion(self.hessian)
            logger.info(self.hessian)
            logger.info(self.gradient_actor(current_value, **kwargs))
            logger.info(-1 * self.hessian @ self.gradient_actor(current_value, **kwargs))
            logger.info(current_value)
            return -1 * self.hessian @ self.derivative_wrapper.gradient(self.likelihood_function, current_value, **kwargs), self.derivative_wrapper.gradient(self.likelihood_function, current_value, **kwargs)

        return super().calc_direction(current_value, previous_optimum, current_direction, current_gradient, next_gradient,
                                      **kwargs)


# BFGS optimizer using at the very first step a full inverse hessian matrix calculation
class BFGS(BFGSOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hessian = None
        self.method_id = "BFGSClass"

    # not necessary to override ´matrix_inversion`

    # not necessary to override ´first_hessian_fit`

    # not necessary to override ´gradient_actor`

    # not necessary to override ´hessian_actor`

    # not necessary to override ´select_hessian_strategy`

    def calc_direction(self, current_value, previous_optimum, current_direction, current_gradient, next_gradient,
                       **kwargs):
        if self.hessian is None:
            self.hessian = self.matrix_inversion(
                self.derivative_wrapper.hessian(self.likelihood_function, current_value))
            grad = self.derivative_wrapper.gradient(self.likelihood_function, current_value, **kwargs)
            return -1 * self.hessian @ grad, grad

        return super().calc_direction(current_value, previous_optimum, current_direction, current_gradient, next_gradient,
                                      **kwargs)
