import argparse

import numpy as np
import scipy
# from FindiffWrapper import FindiffWrapper, FiniteDifferenceDerivatives, DerivativeWrapper
from FindiffWrapper import NumDiffToolsDerivatives
from matplotlib import pyplot as plt
from scipy.optimize import rosen, rosen_der, rosen_hess
from tqdm import trange


def test_potential(x, y, z):
    return -12 * x * (y ** 3) * z


def rosen_transform(x, *args):
    data = np.array([x])
    return np.append(data, args, axis=0)


def test_rosen(x, *args):
    return rosen(rosen_transform(x, *args))


def second_test_potential_vector(x, **kwargs):
    result = test_potential(x[0], x[1], x[2])
    if len(x[0].shape) > 0 and x[0].shape[0] > 1:
        print(len(x[0]))
    return result


def nd_Demo(x, *args):
    print("step")
    print(x)
    return rosen(x)


class SecondTestPotentialVector:
    def __init__(self, potential):
        self._cost_function = potential

    def __call__(self, x, **kwargs):
        # in opposite to the quite simple implementation above we must here infere the number of parameter submitted to the function
        # inference necessary
        shape = np.shape(x)
        if len(shape) >= 1:
            return self._cost_function(*x)
        else:
            return self._cost_function(x)
        # if len(shape) > 3:
        #     raise ValueError("Too many dimensions for the potential vector!")
        # elif len(shape) > 1:
        #     self._indices = shape[0]
        #     return self._cost_function(*x)
        # elif len(shape) == 1:
        #     self._indices = 1
        #     return self._cost_function(*x)
        # else:
        #     return self._cost_function(x)


class TestPotentialVector:
    def __init__(self, potential):
        self.cost_function = potential

    def __call__(self, x, *args):
        # no further arguments => only the x vector of importance as this will perhaps also include all the additional parameter in vector form
        # at least this is to be assumed here!
        if len(args) == 0:
            return self.cost_function(*x)

        else:
            # further arguments are present => need to transform all this into a suitable form for evaluation by the not
            # that advanced functions.
            temp = x.copy()
            eff_shape = x.shape[1:]
            if len(x.shape) > 0:
                for further in args:
                    # for each further argument presented we need to generate as many values as there are data points requested!
                    try:
                        additional_values = np.full(eff_shape, further)
                        temp = np.append(temp, [additional_values], axis=0)
                    except Exception as e:
                        print(temp.shape)
                        print(further.shape)
                        print(len(x.shape))
                        print(x)
                        raise e

                result = [self.cost_function(*temp)]
            else:
                # no real shape information => just one value to evaluate at
                temp = np.array(temp)
                temp = np.append(temp, args)
                result = self.cost_function(*temp)

            return result

def test_potential_vector(x, *args):
    if len(args) == 0:
        return test_potential(*x)

    else:
        temp = x.copy()
        eff_shape = x.shape[1:]
        if len(x.shape) > 0:
            for further in args:
                try:
                    addional_values = np.full(eff_shape, further)
                    temp = np.append(temp, [addional_values], axis=0)
                except Exception as e:
                    print(temp.shape)
                    print(further.shape)
                    print(len(x.shape))
                    print(x)
                    raise e

            result = [test_potential(*temp)]
        else:
            temp = np.array(temp)
            temp = np.append(temp, args)
            result = test_potential(*temp)

        return result


def analytical_gradient(x, y, z):
    return np.array([-12 * (y ** 3) * z, -12 * 3 * x * (y ** 2) * z, -12 * x * (y ** 3)])


def rosen_gradient(x, *args):
    return rosen_der(rosen_transform(x, *args))


def analytical_derivative(x, y, z, axis=0):
    return analytical_gradient(x, y, z)[axis]


def rosen_derivative(x, *args, axis=0):
    return rosen_gradient(x, *args)[axis]


def analytical_hessian(x, y, z):
    temp = np.array([[0, -12 * 3 * (y ** 2) * z, -12 * y ** 3],
                     [-12 * 3 * (y ** 2) * z, -12 * 6 * x * y * z, -12 * 3 * x * (y ** 2)],
                     [-12 * y ** 3, -12 * 3 * x * (y ** 2), 0]], dtype=np.float64)
    return temp


def rosen_hessian(x, *args):
    return rosen_hess(rosen_transform(x, *args))


def evaluate_numerical_derivative(handler, grid, test_potential, analytical_deriv, analytical_grad, analytical_hess,
                                  axis, *fixed_args, argument_vector=True, show=True, tqdm=False):
    # issue for this tests: We can't assume vectorised behaviour is implemented correctly.
    if not tqdm:
        print("Will ignore the axis argument for now.")

    # begin with testing the derivatives
    try:
        sample_deriv = np.zeros(grid.shape[0])
        ref_deriv = np.zeros(grid.shape[0])
        point_vector = np.zeros(len(fixed_args) + 1)
        for idx, arg in enumerate(fixed_args):
            point_vector[idx + 1] = arg

        for idx, point in enumerate(grid):
            point_vector[0] = point
            sample_deriv[idx] = handler.derivative(test_potential, point_vector)
            ref_deriv[idx] = analytical_deriv(point, *fixed_args)

        if not tqdm:
            print(
                    f"The comparison to the analytical results for derivatives yields {np.allclose(sample_deriv, ref_deriv)}")

        fig_deriv, ax_deriv = plt.subplots()
        ax_deriv.plot(grid, sample_deriv, label="Numerical")
        ax_deriv.plot(grid, ref_deriv, "--", label="Analytical")
        if not tqdm:
            print("The analytical values are:")
        ax_deriv.legend()
        ax_deriv.set_title("Test of the derivative!")
    except Exception as e:
        print(e)
        print("Error occurred for implementation in simple test: ", impl)
        print("But will continue with the next one.")
        raise e

    # test the gradient now
    try:
        sample_grad = np.zeros((len(fixed_args) + 1, grid.shape[0]))
        ref_grad = np.zeros((len(fixed_args) + 1, grid.shape[0]))
        point_vector = np.zeros(len(fixed_args) + 1)
        for idx, arg in enumerate(fixed_args):
            point_vector[idx + 1] = arg

        for idx, point in enumerate(grid):
            point_vector[0] = point
            sample_grad[:, idx] = handler.gradient(test_potential, point_vector)
            ref_grad[:, idx] = analytical_grad(point, *fixed_args)

        if not tqdm:
            print(f"The comparison to the analytical results for gradient yields {np.allclose(sample_grad, ref_grad)}")
        fig_grad, ax_grad = plt.subplots(len(fixed_args) + 1, sharex=True)
        for axes in range(len(fixed_args) + 1):
            ax_grad[axes].plot(grid, sample_grad[axes], label="Numerical")
            ax_grad[axes].plot(grid, ref_grad[axes], '--', label="Analytical")
            ax_grad[axes].legend()
        ax_grad[0].set_title("Test of the gradient!")
    except Exception as e:
        print(e)
        print("Error occurred for implementation in gradient test: ", impl)
        print("But will continue with the next one.")
        raise e

    # test the hessian
    try:
        sample_hessian = np.zeros((len(fixed_args) + 1, len(fixed_args) + 1, grid.shape[0]))
        ref_hessian = np.zeros((len(fixed_args) + 1, len(fixed_args) + 1, grid.shape[0]))
        point_vector = np.zeros(len(fixed_args) + 1)
        for idx, arg in enumerate(fixed_args):
            point_vector[idx + 1] = arg

        for idx, point in enumerate(grid):
            point_vector[0] = point
            sample_hessian[:, :, idx] = handler.hessian(test_potential, point_vector)
            ref_hessian[:, :, idx] = analytical_hess(point, *fixed_args)

        if not tqdm:
            print(
                    f"The comparison to the analytical results for the hessian yields {np.allclose(sample_hessian, ref_hessian)}")
        fig_hess, ax_hess = plt.subplots(len(fixed_args) + 1, len(fixed_args) + 1, sharex=True, sharey=True)
        for axes_1 in range(len(fixed_args) + 1):
            for axes_2 in range(len(fixed_args) + 1):
                ax_hess[axes_1, axes_2].plot(grid, sample_hessian[axes_1, axes_2], label="Numerical")
                ax_hess[axes_1, axes_2].plot(grid, ref_hessian[axes_1, axes_2], '--', label="Analytical")
                ax_hess[axes_1, axes_2].legend()
        ax_hess[0, 0].set_title("Test of the hessian!")
    except Exception as e:
        print(e)
        print("Error occurred for implementation in hessian test: ", impl)
        print("But will continue with the next one.")
        raise e

    if show:
        plt.show()
    else:
        plt.close('all')


def test_args(x, *args):
    print("test function called with:")
    print(x)
    print("and the following arguments:")
    for arg in args:
        print(arg)
    print("finished")



if __name__ == '__main__':
    # will be done only for a fixed axis, will only evaluate for different values along the x-axis
    parser = argparse.ArgumentParser(description="Test the gradient descent optimisation procedures.")
    parser.add_argument('--benchmark', action='store_true', )
    parser.add_argument('--show', action='store_true',
                        help='Flag, whether to show the plots or not. (default: %(default)s)')
    parser.add_argument('--bench-iterations', type=int, default=50,
                        help='Number of iterations for the benchmark. (default: %(default)s)')
    parser.add_argument('--sample-axis', type=int, default=0,
                        help='Axis along which to sample the function. (default: %(default)s')
    parser.add_argument('--dev-order', type=int, default=1,
                        help='Order of the numerical differentiation. (default: %(default)s)')
    parser.add_argument('--accuracy', type=int, default=2,
                        help='Accuracy of the numerical differentiation. (default: %(default)s)')
    parser.add_argument('--step-size', type=float, default=0.0001,
                        help='Step size for the numerical differentiation. (default: %(default)s)')
    parser.add_argument('--demo-dimension', type=int, default=3,
                        help='Dimension of the demo problem. (default: %(default)s)')
    parsed_arguments = parser.parse_args()
    sample_axis = parsed_arguments.sample_axis
    step_size = parsed_arguments.step_size
    demo_dimension = parsed_arguments.demo_dimension
    dev_order = parsed_arguments.dev_order
    accuracy = parsed_arguments.accuracy
    print(parsed_arguments.bench_iterations)
    # test_potential_vector = TestPotentialVector(test_potential_vector)
    # second_test_potential_vector = SecondTestPotentialVector(second_test_potential_vector)
    sample_implementations = [NumDiffToolsDerivatives(demo_dimension, step_size, dev_order, accuracy, sample_axis)]
    # sample_implementations = [FindiffWrapper(demo_dimension, step_size, dev_order, accuracy, sample_axis), FiniteDifferenceDerivatives(demo_dimension, step_size, dev_order, accuracy, sample_axis)]
    points_to_evaluate = np.linspace(-50, 50, 10000)

    # print("nd gradient")
    # print(nd.Gradient(nd_Demo)(np.array([1,2,3])))
    # print("scipy hessian")
    # print(hessian(nd_Demo, np.array([1,2,3])))
    # print(rosen_hess(np.array([1,2,3])))

    print("fast check of the list args processing options")
    demo_matrix = np.vstack([(number * 10 + np.arange(5)) for number in range(10)])
    print(demo_matrix.shape)
    print(demo_matrix.T.shape)
    print(demo_matrix)
    test_args(*demo_matrix)
    # raise RuntimeError()



    scipy_hessian = scipy.differentiate.hessian(test_potential_vector, [3, 3, 3])['ddf']
    ana_hessian = analytical_hessian(3, 3, 3)
    numdiff_handler = NumDiffToolsDerivatives(demo_dimension, step_size, dev_order, accuracy, sample_axis)
    # print(np.allclose(numdiff_handler.hessian(second_test_potential_vector, [3, 3, 3]), ana_hessian, rtol=1e-05, atol=5e-08))
    # print(scipy.differentiate.hessian(test_potential_vector, [3, 3, 3]))
    # print(numdiff_handler.hessian(second_test_potential_vector, [3, 3, 3]))
    # print(analytical_hessian(3,3,3))
    # print(scipy.differentiate.derivative(test_potential_vector, 3, args=[3, 3]))
    # print(numdiff_handler.gradient(second_test_potential_vector, [3, 3, 3]))
    # print(numdiff_handler.derivative(second_test_potential_vector, [3, 3, 3]))
    advanced_test_potential = SecondTestPotentialVector(test_potential)
    simple_test_potential = TestPotentialVector(test_potential)
    rosen_test_function = SecondTestPotentialVector(test_rosen)
    for impl in sample_implementations:
        if parsed_arguments.benchmark:
            for _ in trange(20):
                evaluate_numerical_derivative(impl, points_to_evaluate, advanced_test_potential,
                                              analytical_derivative,
                                              analytical_gradient, analytical_hessian, 1, 3, 3, show=False, tqdm=True)
            try:
                for _ in trange(20):
                    evaluate_numerical_derivative(impl, points_to_evaluate, simple_test_potential,
                                                  analytical_derivative,
                                                  analytical_gradient, analytical_hessian, 1, 3, 3, show=False,
                                                  tqdm=True)
            except Exception as e:
                print(e)
                print("Error occurred for implementation: ", impl)
                print("But will continue with the next one.")

        else:
            try:
                evaluate_numerical_derivative(impl, points_to_evaluate, advanced_test_potential, analytical_derivative,
                                              analytical_gradient, analytical_hessian, 1, 3, 3, show=False, tqdm=False)
                evaluate_numerical_derivative(impl, points_to_evaluate, rosen_test_function, rosen_derivative,
                                              rosen_gradient, rosen_hessian, 1, 8, 5, show=False, tqdm=False)
            except Exception as e:
                print(e)
                print("Error occurred for implementation: ", impl)
                print("But will continue with the next one.")

    # now test the different implementations
