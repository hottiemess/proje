from optimize.GradientDescent import GradientDescent, newton_update


class NewtonOptimizer(GradientDescent):
    def __init__(self, likelihood_function, dimension, derivative_method="numdifftools", derivative_class=None,
                 acc_goal=2,
                 deriv_stepsize=0.001, target_acc=1e-12):
        super().__init__(likelihood_function, dimension, [], derivative_method, derivative_class, acc_goal,
                         deriv_stepsize, target_acc)
        self.hessian = None

    # not necessary to override ´matrix_inversion`

    # not necessary to override ´first_hessian_fit`

    # not necessary to override ´gradient_actor`

    # not necessary to override ´hessian_actor`

    # not necessary to override ´select_hessian_strategy`

    def calc_direction(self, current_value, previous_optimum, current_direction, current_gradient, next_gradient,
                       **kwargs):
        # will calculate the direction for the iteration by the matrix-vector product D*grad(L)
        # optimse the computation by performing only a minimal number of gradient evaluations
        grad = self.derivative_wrapper.gradient(self.likelihood_function, current_value, **kwargs) if next_gradient is None else next_gradient
        if current_gradient is None:
            current_gradient = self.derivative_wrapper.gradient(self.likelihood_function, previous_optimum, **kwargs)

        if self.hessian is None:
            self.hessian = self.derivative_wrapper.hessian(self.likelihood_function, current_value)
            self.hessian = self.matrix_inversion(self.hessian)

        full_hessian, _ = newton_update(self.likelihood_function, current_value, previous_optimum, self.hessian,
                                        self.derivative_wrapper, inverse=True, next_grad=grad,
                                        current_gradient=current_gradient)
        self.hessian = full_hessian
        return -1 * self.hessian @ grad, grad

    # not necessary to override `gradient_step_size`

    # not necessary to override `update_acc_condition`

    def general_gradient_actor(self, initial_values, args=(), **kwargs):
        if "gradient_descent" in kwargs:
            del kwargs["gradient_descent"]

        # sneaking with the arguments to avoid reimplementing helper functions of the more advanced wrapper class
        # which serves here as a super class
        return super().general_gradient_actor(initial_values, gradient_descent=True, **kwargs)
