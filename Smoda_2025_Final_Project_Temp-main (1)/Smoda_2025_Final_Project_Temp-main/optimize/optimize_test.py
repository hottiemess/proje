import argparse
import os
import sys
from glob import glob
from inspect import signature
from logging import getLogger
from time import time
from warnings import warn

import iminuit
import numdifftools as nd
import numpy as np
from numpy.random import default_rng
from scipy.optimize import rosen

if os.environ.get("CI", "None") == "Github":
    sys.path.append(os.environ["PYTHON_ADD_PATH"])
    print(sys.path)

from linalg.matrix import matrix_inverse_second as matrix_inverse
from optimize.GradientDescent import GradientDescent, DescentAlgorithms
from optimize.SimpleGradient import RawGradientDescent, GradientDescent as AdvancedGradientDescent
from optimize.newton import NewtonOptimizer
from optimize.quasi_newton import BFGSOptimizer, AdvancedBFGS, BFGS

generator = default_rng(42)
logger = getLogger(f"testing.{__name__}")
nll_logger = logger.getChild("nll")
do_cleanup = True

# it seems not be possible enforce a higher accuracy
optimise_accuracy = 1e-5


class TestFunction:
    def __init__(self, data=None, points=1000000, demo_parameters=[[12, 3, 0.1, 5, 6, 20], [0.1, 1, 1, 1, 2, 2]],
                 regenerate=False):
        pass

    def __call__(self, x: np.ndarray, *args, **kwargs):
        return rosen(x)


class TestNLL(TestFunction):
    def __init__(self, data=None, points=1000000, demo_parameters=[[12, 3, 0.1, 5, 6, 20], [0.1, 1, 1, 1, 2, 2]],
                 regenerate=False):
        super().__init__()
        if data is None:
            assert points is not None
            assert demo_parameters is not None
            means = np.array(demo_parameters[0])
            self.cov = np.diag(demo_parameters[1]) if len(np.shape(demo_parameters[1])) == 1 else np.array(
                    demo_parameters[1])
            if regenerate:
                # FIXME: Why this simplification?
                self.cov = np.identity(len(means))
                self.data = generator.multivariate_normal(mean=means, cov=self.cov, size=points)
                np.savetxt("cov_2.txt", self.data)
            else:
                self.data = np.loadtxt("cov.txt")
            ref_cov = np.cov(self.data, rowvar=False)
            print("Covariances just for reference:")
            print(ref_cov)
            print(self.cov)
            print(np.diag(demo_parameters[1]))
        else:
            # data os explicitly provided. Thus, no data generation or saving data procedure is attempted
            self.data = data
        # for receiving accurate results when evaluating the likelihood function it is necessary to use the actual
        # covariance matrix of the sampled data and not the design matrix for random samples!
        self.cov = np.cov(self.data.T)

        self.inv_cov = matrix_inverse(self.cov)
        self.norm = (2 * np.pi) ** 3 * np.sqrt(np.linalg.det(self.cov))

    def __call__(self, x: np.ndarray, *args, **kwargs):
        # this implementation will have an issue!
        # normalisation of the pdf is irrelevant for the optimisation process so it will be dropped!
        return np.sum([get_one_nll_pdf(x, current_data, self.inv_cov) for current_data in self.data])

    @property
    def mean(self):
        return np.mean(self.data, axis=0)

    @property
    def std(self):
        return np.std(self.data, axis=0)

    @property
    def covariance(self):
        return self.cov


# in this no random data is necessary
class TestRosenbrockNLL(TestFunction):
    def __init__(self, data=None, points=1000000, demo_parameters=[[12, 3, 0.1, 5, 6, 20], [0.1, 1, 1, 1, 2, 2]],
                 regenerate=False):
        super().__init__()
        self.a = 1
        self.b = 100

    def __call__(self, x: np.ndarray, *args, **kwargs):
        assert isinstance(x, np.ndarray)
        first_parameter_set = x[1:]
        second_parameter_set = x[:-1]
        squared_parameter_set = second_parameter_set ** 2
        intermediate_result = self.b * (first_parameter_set - squared_parameter_set) ** 2 + (
                    self.a - second_parameter_set) ** 2
        return np.sum(intermediate_result)



def get_one_nll_pdf(x, vector, inv_cov):
    difference = x - vector
    return difference.T @ inv_cov @ difference


def negative_log_likeklihood_gaussian(theta1, theta2, theta3, theta4, theta5, theta6: np.ndarray):
    # Implement - log L_G for 6 parameters and the given optimal parameters here (and remove the pass)
    pass


def test_sample(x: np.ndarray):
    return -x[0] * x[1] ** 3 * x[2]


if __name__ == "__main__":
    # declare an argument parser to optimise the test procedures
    parser = argparse.ArgumentParser(description="Test the gradient descent optimisation procedures.")
    parser.add_argument('--generate-data', action='store_true',
                        help='Flag whether to regenerate the test data set for likelihood optimisation (default: %(default)s)')
    parser.add_argument('--no-steepest', action='store_false',
                        help='Flag whether to use the steepest descent algorithm (default: %(default)s)',
                        dest='use_steepest')
    parser.add_argument('--no-bfgs', action='store_false',
                        help='Flag whether to use the bfgs algorithm (default: %(default)s)',
                        dest='use_bfgs')
    parser.add_argument('--no-newton', action='store_false',
                        help='Flag whether to use the newton algorithm (default: %(default)s)',
                        dest='use_newton')
    parser.add_argument('--no-super-test', action='store_false',
                        help='Flag whether to use the gradient descent algorithm (default: %(default)s)',
                        dest='use_super_test')
    parser.add_argument('--data-points', action='store', type=int, default=100,
                        help='Number of data points to generate for likelihood optimisation (default: %(default)s)')
    parser.add_argument('--demo-parameters', action='store', nargs='+', type=float, default=[[12, 3, 0.1, 5, 6, 20],
                                                                                             [0.1, 1, 1, 1, 2, 2]],
                        help='Demo parameters for likelihood optimisation (default: %(default)s)')
    parser.add_argument('--guess', action='store', nargs='+', type=float, default=[1, 1, 1, 1, 1, 1],
                        help='Initial guess for the optimisation (default: %(default)s)')
    parser.add_argument('--seed', action='store', type=int, default=42,
                        help='Seed for random number generator (default: %(default)s)')
    parser.add_argument('--derivative-method', action='store', type=str, default="numdifftools",
                        help='Method to calculate the derivative (default: %(default)s)')
    parser.add_argument('--test-sig-detection', action='store_true',
                        help='Flag whether to test the signature detection (default: %(default)s)')
    parser.add_argument('--extract-path', action='store_true',
                        help='Flag, whether to extract the path information or not! (default: %(default)s)')
    parser.add_argument('--test-rosenbrock', action='store_true',
                        help="Flag whether to test the rosenbrock function. (default: %(default)s)")
    parser.add_argument('--rosenbrock-dimension', action='store', type=int, default=2,
                        help="Number of dimensions to choose for the rosenbrock function (default: %(default)s)")

    parser.set_defaults(
            use_steepest=True, use_bfgs=True, use_newton=True, use_super_test=True
    )
    arguments = parser.parse_args()

    print((DescentAlgorithms.BFGS == "bfgs"))

    test_name = "GAUSSIAN"

    # do a little bit of clean up
    print("try to perform some cleanup")
    if do_cleanup:
        files = glob("step_tracker_*.txt")
        for file in files:
            os.remove(file)
        files = glob("step_tracker_*.npy")
        for file in files:
            os.remove(file)

    print(arguments.demo_parameters)

    # want to initialise an example data class for instance
    demo_demo = [[-1.9, -1.2, 0.42, 0.9, 1.4, 2.3], [
        [0.8, 0.2, 0.1, 0.0, 0.1, 0.2],
        [0.2, 0.9, 0.0, 0.1, 0.3, 0.1],
        [0.1, 0.0, 0.7, 0.2, 0.1, 0.0],
        [0.0, 0.1, 0.2, 0.6, 0.0, 0.1],
        [0.1, 0.3, 0.1, 0.0, 0.9, 0.2],
        [0.2, 0.1, 0.0, 0.1, 0.2, 0.8]
    ]]

    full_start = time()
    validation_data = TestNLL(points=arguments.data_points, regenerate=arguments.generate_data)
    # TestNLL(points=100, regenerate=True, demo_parameters=demo_demo)
    initial_guess = arguments.guess
    print(validation_data(np.array([12, 3, 0.1, 5, 6, 20])))
    # will try to do some reference options first
    print("A first gradient evaluation:")
    print(nd.Gradient(validation_data)(np.array(initial_guess)))
    np.savetxt("gradient_test.txt", nd.Gradient(validation_data)(np.array(initial_guess)))
    print(nd.Hessian(validation_data)(np.array(initial_guess)))
    np.savetxt("hessian_test.txt", nd.Hessian(validation_data)(np.array(initial_guess)))
    print(nd.Gradient(test_sample)([1, 1, 1]))

    # perhaps we should retrieve the output of our likelihood for some data points?
    print("test the implementation by using iminuit as a reference:")
    print(validation_data(np.array(initial_guess)))
    grid_axis = np.linspace(-5, 5, 10)
    grid = np.array([[value, 1, 1, 1, 1, 1] for value in grid_axis])
    nll_values = np.array([validation_data(vector) for vector in grid])
    np.savetxt("nll_values.txt", nll_values)
    np.savetxt("grid.txt", grid)
    m = iminuit.Minuit(validation_data, initial_guess, grad=nd.Gradient(validation_data),
                       hessian=nd.Hessian(validation_data))
    # m = iminuit.Minuit(validation_data, [1,1,1], errordef=1)
    m.migrad()
    m.hesse()
    print(repr(m))
    print(m.values)
    print(m.errors)
    print(m.covariance)
    print(m.ngrad)
    print(m.nhessian)
    print(type(m.values))

    if arguments.test_sig_detection:
        print("Try to handle simple impl.")
        nll_sig = signature(validation_data)
        for parameter in nll_sig.parameters.values():
            print("next parameter:")
            print(parameter)
            print(parameter.name)
            print(parameter.kind)
            print(parameter.default)
            print(parameter.annotation)
        print("Try to handle default impl.")
        # from submission_framework.framework import negative_log_likeklihood_gaussian
        default_sig = signature(negative_log_likeklihood_gaussian)
        for parameter in default_sig.parameters.values():
            print("next parameter:")
            print(parameter)
            print(parameter.name)
            print(parameter.kind)
            print(parameter.default)
            print(parameter.annotation)
            print(issubclass(parameter.annotation, np.ndarray))

    expected_estimation = np.array(arguments.demo_parameters[0])
    analytical_estimation = validation_data.mean
    print(expected_estimation)
    print(validation_data.mean)
    print(validation_data.std)
    print(validation_data.covariance)

    if arguments.use_steepest:
        # first check the original gradient descent method
        print(f"[{test_name}] Fitting with original gradient descent method:")
        fitter = RawGradientDescent(validation_data, 6, target_acc=optimise_accuracy)
        # try it out currently using the scipy implementations
        start_time = time()
        result = fitter.general_gradient_actor(np.array(initial_guess), max_iterations=1000,
                                               search_args={"algo": "backtracking", "maxiter": 20, "use_wolfe": False,
                                                            "strong_wolfe": False})
        stop_time = time()
        print(result)
        print(fitter.n_iterations)

        print(np.isclose(fitter.estimators, expected_estimation))
        print(np.isclose(fitter.estimators, analytical_estimation))
        print(np.abs(fitter.estimators - expected_estimation))
        if arguments.extract_path:
            print(fitter.extract_intermediate_results()[:, 2, :])
        print(f"It takes about {stop_time - start_time}")

        # second check the gradient descent method
        print(f"[{test_name}] Fitting with gradient descent method:")
        fitter = AdvancedGradientDescent(validation_data, 6, target_acc=optimise_accuracy)
        start_time = time()
        result = fitter.general_gradient_actor(np.array(initial_guess),
                                               search_args={"algo": "backtracking", "maxiter": 20, "use_wolfe": False,
                                                            "strong_wolfe": False})
        stop_time = time()
        print(result)
        print(fitter.n_iterations)
        print(np.isclose(fitter.estimators, expected_estimation))
        print(np.isclose(fitter.estimators, analytical_estimation))
        print(np.abs(fitter.estimators - expected_estimation))
        if arguments.extract_path:
            print(fitter.extract_intermediate_results()[:, 2, :])
        print(f"It takes about {stop_time - start_time}")
    if arguments.use_bfgs:
        # third check the bfgs method
        print(f"[{test_name}] Fitting with bfgs method:")
        fitter = BFGSOptimizer(validation_data, 6, target_acc=optimise_accuracy)
        start_time = time()
        result = fitter.general_gradient_actor(np.array(initial_guess),
                                               search_args={"algo": "backtracking", "maxiter": 20, "use_wolfe": False,
                                                            "strong_wolfe": False})
        stop_time = time()
        print(result)
        print(fitter.n_iterations)
        print(np.isclose(fitter.estimators, expected_estimation))
        print(np.isclose(fitter.estimators, analytical_estimation))
        print(np.abs(fitter.estimators - expected_estimation))
        if arguments.extract_path:
            print(fitter.extract_intermediate_results()[:, 2, :])
        print(f"It takes about {stop_time - start_time}")

        # fourth check the advanced bfgs method
        print(f"[{test_name}] Fitting with advanced bfgs method:")
        fitter = AdvancedBFGS(validation_data, 6, target_acc=optimise_accuracy)
        start_time = time()
        result = fitter.general_gradient_actor(np.array(initial_guess),
                                               search_args={"algo": "backtracking", "maxiter": 20, "use_wolfe": False,
                                                            "strong_wolfe": False})
        stop_time = time()
        print(result)
        print(fitter.n_iterations)
        print(np.isclose(fitter.estimators, expected_estimation))
        print(np.isclose(fitter.estimators, analytical_estimation))
        print(np.abs(fitter.estimators - expected_estimation))
        if arguments.extract_path:
            print(fitter.extract_intermediate_results()[:, 2, :])
        print(f"It takes about {stop_time - start_time}")

        # fifth check the bfgs method with advanced initial guess of the inverse hessian
        print(f"[{test_name}] Fitting with bfgs method with advanced initial guess of the inverse hessian:")
        fitter = BFGS(validation_data, 6, target_acc=optimise_accuracy)
        start_time = time()
        result = fitter.general_gradient_actor(np.array(initial_guess),
                                               search_args={"algo": "backtracking", "maxiter": 20, "use_wolfe": False,
                                                            "strong_wolfe": False})
        stop_time = time()
        print(result)
        print(fitter.n_iterations)
        print(np.isclose(fitter.estimators, expected_estimation))
        print(np.isclose(fitter.estimators, analytical_estimation))
        print(np.abs(fitter.estimators - expected_estimation))
        if arguments.extract_path:
            print(fitter.extract_intermediate_results()[:, 2, :])
        print(f"It takes about {stop_time - start_time}")

    # sixth check the newton method
    if arguments.use_newton:
        print(f"[{test_name}] Fitting with newton method:")
        fitter = NewtonOptimizer(validation_data, 6, target_acc=optimise_accuracy)
        start_time = time()
        result = fitter.general_gradient_actor(np.array(initial_guess),
                                               search_args={"algo": "backtracking", "maxiter": 20, "use_wolfe": False,
                                                            "strong_wolfe": False})
        stop_time = time()
        print(result)
        print(fitter.n_iterations)
        print(np.isclose(fitter.estimators, expected_estimation))
        print(np.isclose(fitter.estimators, analytical_estimation))
        print(np.abs(fitter.estimators - expected_estimation))
        if arguments.extract_path:
            print(fitter.extract_intermediate_results()[:, 2, :])
        print(f"It takes about {stop_time - start_time}")
    if arguments.use_super_test:
        if arguments.use_steepest:
            # seventh check the original gradient descent method
            print(f"[{test_name}] Fitting with original gradient descent method (second try):")
            fitter = GradientDescent(validation_data, 6, [], "numdifftools", target_acc=optimise_accuracy)
            start_time = time()
            result = fitter.general_gradient_actor(np.array(initial_guess), algo=DescentAlgorithms.GRADIENT_DESCENT,
                                                   search_args={"algo": "backtracking", "maxiter": 20,
                                                                "use_wolfe": False, "strong_wolfe": False})
            stop_time = time()
            print(result)
            print(fitter.n_iterations)
            print(np.isclose(fitter.estimators, expected_estimation))
            print(np.isclose(fitter.estimators, analytical_estimation))
            print(np.abs(fitter.estimators - expected_estimation))
            if arguments.extract_path:
                print(fitter.extract_intermediate_results()[:, 2, :])
            print(f"It takes about {stop_time - start_time}")

            # eight check the gradient descent method
            print(f"[{test_name}] Fitting with gradient descent method (second try):")
            fitter = GradientDescent(validation_data, 6, [], "numdifftools", target_acc=optimise_accuracy)
            start_time = time()
            result = fitter.general_gradient_actor(np.array(initial_guess),
                                                   algo=DescentAlgorithms.MODIFIED_GRADIENT_DESCENT,
                                                   search_args={"algo": "backtracking", "maxiter": 20,
                                                                "use_wolfe": False, "strong_wolfe": False})
            stop_time = time()
            print(result)
            print(fitter.n_iterations)
            print(np.isclose(fitter.estimators, expected_estimation))
            print(np.isclose(fitter.estimators, analytical_estimation))
            print(np.abs(fitter.estimators - expected_estimation))
            if arguments.extract_path:
                print(fitter.extract_intermediate_results()[:, 2, :])
            print(f"It takes about {stop_time - start_time}")
        if arguments.use_bfgs:
            # ninth check the bfgs method
            print(f"[{test_name}] Fitting with bfgs method (second try):")
            fitter = GradientDescent(validation_data, 6, [], "numdifftools", target_acc=optimise_accuracy)
            start_time = time()
            result = fitter.general_gradient_actor(np.array(initial_guess), algo=DescentAlgorithms.BFGS_SIMPLE,
                                                   search_args={"algo": "backtracking", "maxiter": 20,
                                                                "use_wolfe": False, "strong_wolfe": False})
            stop_time = time()
            print(result)
            print(fitter.n_iterations)
            print(np.isclose(fitter.estimators, expected_estimation))
            print(np.isclose(fitter.estimators, analytical_estimation))
            print(np.abs(fitter.estimators - expected_estimation))
            if arguments.extract_path:
                print(fitter.extract_intermediate_results()[:, 2, :])
            print(f"It takes about {stop_time - start_time}")

            # tenth check the advanced bfgs method
            print(f"[{test_name}] Fitting with advanced bfgs method (second try):")
            fitter = GradientDescent(validation_data, 6, [], derivative_method="numdifftools",
                                     target_acc=optimise_accuracy)
            start_time = time()
            result = fitter.general_gradient_actor(np.array(initial_guess), algo=DescentAlgorithms.BFGS_ADVANCED,
                                                   search_args={"algo": "backtracking", "maxiter": 20,
                                                                "use_wolfe": False, "strong_wolfe": False})
            stop_time = time()
            print(result)
            print(np.isclose(fitter.estimators, np.array(arguments.demo_parameters[0])))
            print(fitter.n_iterations)
            print(np.isclose(fitter.estimators, expected_estimation))
            print(np.isclose(fitter.estimators, analytical_estimation))
            print(np.abs(fitter.estimators - expected_estimation))
            if arguments.extract_path:
                print(fitter.extract_intermediate_results()[:, 2, :])
            print(f"It takes about {stop_time - start_time}")

            # eleventh check the bfgs method with advanced initial guess of the inverse hessian
            print(
                f"[{test_name}] Fitting with bfgs method with advanced initial guess of the inverse hessian (second try):")
            fitter = GradientDescent(validation_data, 6, [], derivative_method="numdifftools",
                                     target_acc=optimise_accuracy)
            start_time = time()
            result = fitter.general_gradient_actor(np.array(initial_guess), algo=DescentAlgorithms.BFGS,
                                                   search_args={"algo": "backtracking", "maxiter": 20,
                                                                "use_wolfe": False, "strong_wolfe": False})
            stop_time = time()
            print(result)
            print(fitter.n_iterations)
            print(np.isclose(fitter.estimators, expected_estimation))
            print(np.isclose(fitter.estimators, analytical_estimation))
            print(np.abs(fitter.estimators - expected_estimation))
            if arguments.extract_path:
                print(fitter.extract_intermediate_results()[:, 2, :])
            print(f"It takes about {stop_time - start_time}")

        # twelth check the newton method
        if arguments.use_newton:
            print(f"[{test_name}] Fitting with newton method (second try):")
            fitter = GradientDescent(validation_data, 6, [], derivative_method="numdifftools",
                                     target_acc=optimise_accuracy)
            start_time = time()
            result = fitter.general_gradient_actor(np.array(initial_guess), algo=DescentAlgorithms.NEWTON,
                                                   search_args={"algo": "backtracking", "maxiter": 20,
                                                                "use_wolfe": False, "strong_wolfe": False})
            stop_time = time()
            print(result)
            print(fitter.n_iterations)
            print(np.isclose(fitter.estimators, expected_estimation))
            print(np.isclose(fitter.estimators, analytical_estimation))
            print(np.abs(fitter.estimators - expected_estimation))
            if arguments.extract_path:
                print(fitter.extract_intermediate_results()[:, 2, :])
            print(f"It takes about {stop_time - start_time}")

    if arguments.test_rosenbrock:
        rosenbrock = TestRosenbrockNLL()
        rosenbrock_guess = np.full(arguments.rosenbrock_dimension, 0.5)
        print(f"rosenbrock start value is {rosenbrock_guess}")
        rosenbrock_expected_estimation = np.full(arguments.rosenbrock_dimension, 1)
        rosenbrock_analytical_estimation = np.full(arguments.rosenbrock_dimension, 1)
        if arguments.rosenbrock_dimension >= 4 and arguments.rosenbrock_dimension <= 7:
            rosenbrock_expected_estimation[0] = -1
        elif arguments.rosenbrock_dimension > 7:
            warn(
                "For the rosenbrock function with dimension larger 7 the analytical optimas are not computed. So returned information about convergence maybe incorrect.")

        print("[ROSENBROCK] Fitting with bfgs method:")
        fitter = BFGSOptimizer(rosenbrock, arguments.rosenbrock_dimension, target_acc=optimise_accuracy)
        start_time = time()
        result = fitter.general_gradient_actor(rosenbrock_guess,
                                               search_args={"algo": "backtracking", "maxiter": 20, "use_wolfe": True,
                                                            "strong_wolfe": True})
        stop_time = time()
        print(result)
        print(fitter.n_iterations)
        print(np.isclose(fitter.estimators, rosenbrock_expected_estimation))
        print(np.isclose(fitter.estimators, rosenbrock_analytical_estimation))
        print(np.abs(fitter.estimators - rosenbrock_expected_estimation))
        if arguments.extract_path:
            print(fitter.extract_intermediate_results()[:, 2, :])
        print(f"It takes about {stop_time - start_time}")

        # fourth check the advanced bfgs method
        print("[ROSENBROCK] Fitting with advanced bfgs method:")
        fitter = AdvancedBFGS(rosenbrock, arguments.rosenbrock_dimension, target_acc=optimise_accuracy)
        start_time = time()
        result = fitter.general_gradient_actor(rosenbrock_guess,
                                               search_args={"algo": "backtracking", "maxiter": 20, "use_wolfe": True,
                                                            "strong_wolfe": True})
        stop_time = time()
        print(result)
        print(fitter.n_iterations)
        print(np.isclose(fitter.estimators, rosenbrock_expected_estimation))
        print(np.isclose(fitter.estimators, rosenbrock_analytical_estimation))
        print(np.abs(fitter.estimators - rosenbrock_expected_estimation))
        if arguments.extract_path:
            print(fitter.extract_intermediate_results()[:, 2, :])
        print(f"It takes about {stop_time - start_time}")

        # fifth check the bfgs method with advanced initial guess of the inverse hessian
        print("[ROSENBROCK] Fitting with bfgs method with advanced initial guess of the inverse hessian:")
        fitter = BFGS(rosenbrock, arguments.rosenbrock_dimension, target_acc=optimise_accuracy)
        start_time = time()
        # maybe the current line search implementation is still a problem
        result = fitter.general_gradient_actor(rosenbrock_guess,
                                               search_args={"algo": "scipy", "maxiter": 20, "use_wolfe": True,
                                                            "strong_wolfe": True})
        stop_time = time()
        print(result)
        print(fitter.n_iterations)
        print(np.isclose(fitter.estimators, rosenbrock_expected_estimation))
        print(np.isclose(fitter.estimators, rosenbrock_analytical_estimation))
        print(np.abs(fitter.estimators - rosenbrock_expected_estimation))
        if arguments.extract_path:
            print(fitter.extract_intermediate_results()[:, 2, :])
        print(f"It takes about {stop_time - start_time}s and {fitter.n_iterations} iterations.")


    # TODO: still missing an implementation to visualize the steps on the to the found minimum!
    full_stop = time()
    full_diff = full_stop - full_start
    print(f"The full test tooks {full_diff}")
