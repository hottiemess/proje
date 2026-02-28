import arviz as az
import iminuit
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pymc as pm
from scipy import stats


class FittingFramework():

    def __init__(self, negative_log_likelihood: callable, inital_guess: npt.ArrayLike) -> None:
        """
        This function is called when you create a fit object with 
        
        fit = FittingFramework(...)
        
        The points are replaced with the parameters except "self", which is a reference to the current
        object (will be needed later). So if you have a negative log likelihood "nll" defined somewhere and 
        an array "init_guesses" with initial guesses for the fit, then you initiate the object with 
        
        fit = FittingFramework(nll, init_guesses)
        
        If you want to know more about classes, read about Object Orientated Programming in Python.

        Args:
            negative_log_likelihood (func): A function which returns the negative log likelihood. This functions should have n input parameters for n parameters to fit
            inital_guess (Array/List): List or array of initial guesses
        """

        # Here we save the provided negative_log_likelihood in the object, so later and in functions inside this class
        # you can access this function again through `self.negative_log_likelihood(...)` when working in the class
        # When working outside the class (after doing fit = FittingFramwork(...)) you can access it with 
        # fit.negative_log_likelihood
        self.negative_log_likelihood = negative_log_likelihood
        # The same with the initial guess
        self.initial_guess = inital_guess

        # Create the iminuit object with the provided function and initial guess
        self.m = iminuit.Minuit(self.negative_log_likelihood, *self.initial_guess)

        # We are working with -2logL, so you can use the default error definition of iMinuit
        # If you would just do -logL, then you need to add the following code
        # self.m.errordef = iminuit.Minuit.LIKELIHOOD

    def minimize(self) -> None:
        """
        This function minimizes the negativelog likelihood and should be called after creating a fit object with 
        
        fit.minimize()
        
        In the provided framework this just calles the iMinuit migrad() function and saves the best parameters
        inside a variable calles best_params.
        
        If you work on task 1 or 2 you should implement your algorithm in this function
        """

        # Accessing the minuit object again (which was initiated in __init__) to call migrad()
        # Replace this with your minimization algorithm if you do task 1 or 2        
        self.m.migrad()

        # Saving the best parameters. Replace this with the best parameters you found in your own algorithm
        # Also, this allows to access the best values later throut self.best_params inside the class and with 
        # fit.best_params outside the class if you initiated an object called fit
        self.best_params = self.m.values

    def calc_hesse(self) -> None:
        """
        This function calculated the hesse errors. Should be called after initiating a fit object and after calling minimize on this fit object
        
        If you work on task 3, implement your algorithm here
        """

        # Replace this with your algorithm and save your calculated hesse errors 
        # Also, you can save the covariance matrix you calcualted here
        self.m.hesse()
        self.hesse_errors = self.m.errors

    def likelihood_profile(self, param1: str, param2: str, sigma: float = 0.39) -> None:
        """
        This function calculates the likelihood profile for two given parameters, e.g. theta1 and theta2 and some given interval. You can read more about the parameters in the iMinuit documentation 

        If you work on task 4, implement your algorithm here

        Args:
            param1 (string): first parameter of the profile (e.g. "theta1")
            param2 (string): second parameter of the profile
            sigma (float): interval for the profile
        """
        # Calculate the likelihood profile with your algorithm here and save it, if you work on task 4
        self.profile = self.m.mncontour(param1, param2, cl=sigma)

    def draw_likelihood_profile(self, param1: str, param2: str, sigma: float = 0.39) -> None:
        """
        Same behaviour as `likelihood_profile` but now draws the calculated profile

        If you work on task 4, implement your vizualisation here. You can of course call `likelihoo_profile` from here
        
        If you want to show/save the plots you also need to call `create_plots` (it is implemented this way so you can combine different plots. This is a very simple implementation, you can extend this to a more complicated plotting routine/to your liking)

        Args:
            param1 (string): first parameter of the profile (e.g. "theta1")
            param2 (string): second parameter of the profile
            sigma (float): interval for the profile
        """

        # Replace this with your visualization if you do task 4
        self.m.draw_mncontour(param1, param2, cl=sigma)

    def draw_error_ellipsis(self, param1: str, param2: str) -> None:
        """
        This function draws the error ellipsis. You can work in here for your visualization if you work on task 3
        
        If you want to show/save the plots you also need to call `create_plots` (it is implemented this way so you can combine different plots. This is a very simple implementation, you can extend this to a more complicated plotting routine/to your liking)

        Args:
            param1 (str): first parameter of the ellipsis
            param2 (str): second parameter of the ellipsis
        """

        # Implement the visualization of the error ellipsis with your hesse errors/covariance matrix here, if you work on task 3
        self.m.draw_contour(param1, param2)

    def instantiate_plot(self):
        # this is only a outer wrapper to determine the plot to be drawn!
        return plt.subplots(figsize=(10, 10))

    def create_plots(self, output_path: str = None) -> None:
        """Create or save a plot

        Args:
            output_path (str, optional): Path where to save the plot. If not given, then just show. Defaults to None.
        """
        # forward the plot creation to the a implementation in the proper subclass
        fig, ax = self.instantiate_plot()

        # default content: this was not changed
        if output_path is None:
            plt.show()
        else:
            plt.savefig(output_path)
        plt.close()

    def mcmc(self, sample_size: int, burn_in_draws: int, keep_burn_in: bool = False, number_of_chains: int = 6) -> None:
        """This functions does a Markow Chain Monte Carlo with pymc. Replace the below code with your implementation of your mcmc algorithm if you work on task 5

        Args:
            sample_size (int): Number of draws (without burn in). This number is draws per chain, so for N chains you have N x sample_size draws
            burn_in_draws (int): how many burn in draws per chain
            keep_burn_in (bool, optional): Keep or discard the burn in draws. Defaults to False.
            number_of_chains (int, optional): How many chains to run. Defaults to 6.
        """
        with pm.Model() as pot1m:
            # These are the parameters of the negative log likelihood. This is hardcoded here
            # Would be nicer to make this parametric. Not necessary but if you are motivated this is not 
            # a really hard task to do and might give you more flexibility
            theta1 = pm.Uniform("theta1", lower=-4, upper=4)
            theta2 = pm.Uniform("theta2", lower=-4, upper=4)
            theta3 = pm.Uniform("theta3", lower=-4, upper=4)
            theta4 = pm.Uniform("theta4", lower=-4, upper=4)
            theta5 = pm.Uniform("theta5", lower=-4, upper=4)
            theta6 = pm.Uniform("theta6", lower=-4, upper=4)

            # Add the likelihood. For some reason, pymc wants the positive log likelihood to work properly
            # To be consistent with the rest of the fitting framwork you can just implement your MCMC for the negative log-likelihood
            # You can of course implement your MCMC in a way that it works with the negative log likelihood
            log_likelihood = lambda x1, x2, x3, x4, x5, x6: -1 * self.negative_log_likelihood(x1, x2, x3, x4, x5, x6)
            pm.Potential("log-likelihood", log_likelihood(theta1, theta2, theta3, theta4, theta5, theta6))

            # Do the sampling    
            # Metropolis is one example for an MCMC algorithm
            trace = pm.sample(sample_size, tune=burn_in_draws, step=pm.Metropolis(), chains=number_of_chains,
                              discard_tuned_samples=not keep_burn_in)

            # Get the sample
            sampled = trace.posterior

            # Combine the draws from the 6 chains
            stacked = az.extract(sampled)

            # The parameters for the bayesian interpretation
            self.mcmc_mean = sampled.mean()
            self.mcmc_lower_quantile = sampled.quantile(0.175)
            self.mcmc_upper_quantile = sampled.quantile(1 - 0.175)

            # Mode for theta1, similar for the others
            self.mcmc_modes = []
            self.mcmc_modes.append(stats.mode(stacked.theta1).mode)

            # For bayesian you also need to extract the densities from the sample...

            # For the frequentist interpretation, you would now extract the sampled points (3D points)
            # Then you can calcualte the likelihood on these points (or you already saved that during the run) and find the minimum
            # For the profile you look for points that fulfill the profile criterion. Do this with some margin, since you sampled a finite number of discrete points. This gives you bins with some points and now you minimize among these points by finding the minimum with all the other parameters
            # Then save your results


gaussian_data = None


def basic_advanced_gaussian(x: np.ndarray):
    raise NotImplementedError("This just wrapper method to make it executable. A real implementation must be provided.")


advanced_gaussian_nll = basic_advanced_gaussian


def negative_log_likelihood_gaussian(theta1, theta2, theta3, theta4, theta5, theta6):
    # Implement - log L_G for 6 parameters and the given optimal parameters here (and remove the pass)
    vector = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
    advanced_gaussian_nll(vector)


def advanced_negative_log_likelihood_rosenbrock(theta: np.ndarray) -> float:
    first_parameter_set = theta[1:]
    second_parameter_set = theta[:-1]
    squared_parameter_set = second_parameter_set ** 2
    intermediate_result = 100 * (first_parameter_set - squared_parameter_set) ** 2 + (1 - second_parameter_set) ** 2
    return np.sum(intermediate_result)


def negative_log_likelihood_rosenbrock(theta1, theta2, theta3, theta4, theta5, theta6):
    # Implement - log L_R for 6 parameters and the given optimal parameters here (and remove the pass)
    vector = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
    return advanced_negative_log_likelihood_rosenbrock(vector)


def perform_framework_application(fit: FittingFramework, name="rosenbrock"):
    print(f"Handling now the implementation/validation: {name}")
    fit.minimize()
    fit.calc_hesse()
    fit.create_plots()
    fit.likelihood_profile("theta1", "theta2", sigma=0.39)
    fit.draw_likelihood_profile("theta1", "theta2", sigma=0.39)
    fit.draw_error_ellipsis("theta1", "theta2")
    fit.mcmc(sample_size=1000, burn_in_draws=1000, keep_burn_in=True, number_of_chains=6)
    fit.create_plots(output_path=f"plots/{name}_mcmc.png")


def main(FrameworkClass=FittingFramework):
    global advanced_gaussian_nll
    assert issubclass(FrameworkClass, FittingFramework)
    # The given model predictions and covariance matrix
    model_predictions = np.array([-1.9, -1.2, 0.42, 0.9, 1.4, 2.3])
    cov_matrix = np.array([
        [0.8, 0.2, 0.1, 0.0, 0.1, 0.2],
        [0.2, 0.9, 0.0, 0.1, 0.3, 0.1],
        [0.1, 0.0, 0.7, 0.2, 0.1, 0.0],
        [0.0, 0.1, 0.2, 0.6, 0.0, 0.1],
        [0.1, 0.3, 0.1, 0.0, 0.9, 0.2],
        [0.2, 0.1, 0.0, 0.1, 0.2, 0.8]
    ])

    # prepare the gaussian model with about 1000 samples
    from optimize.optimize_test import TestNLL

    gaussian_data = TestNLL(points=1000, regenerate=True, demo_parameters=[model_predictions, cov_matrix])
    advanced_gaussian_nll = gaussian_data

    # Call the fitting framework here and run it (and your algorithms)

    # Will do all the main code from here even the primary definition is in the other file for better readability
    # handle the gaussian function here
    gaussian_guess_vector = np.array([0, 0, 0, 0, 0, 0])
    validation_gaussian_fit = FittingFramework(negative_log_likelihood_gaussian, gaussian_guess_vector)

    implemented_gaussian_fit = FrameworkClass(negative_log_likelihood_gaussian, gaussian_guess_vector)
    perform_framework_application(validation_gaussian_fit, name="gaussian_validation")
    perform_framework_application(implemented_gaussian_fit, name="gaussian_implementation")

    # handle the rosenbrock function here
    validation_rosen_fit = FittingFramework(negative_log_likelihood_rosenbrock, gaussian_guess_vector)
    implemented_rosen_fit = FrameworkClass(negative_log_likelihood_rosenbrock, gaussian_guess_vector)
    perform_framework_application(validation_rosen_fit, name="rosenbrock_validation")
    perform_framework_application(implemented_rosen_fit, name="rosenbrock_implementation")


# If you call this script with python, then your code starts here
if __name__ == "__main__":
    # The given model predictions and covariance matrix
    model_predictions = np.array([-1.9, -1.2, 0.42, 0.9, 1.4, 2.3])
    cov_matrix = np.array([
        [0.8, 0.2, 0.1, 0.0, 0.1, 0.2],
        [0.2, 0.9, 0.0, 0.1, 0.3, 0.1],
        [0.1, 0.0, 0.7, 0.2, 0.1, 0.0],
        [0.0, 0.1, 0.2, 0.6, 0.0, 0.1],
        [0.1, 0.3, 0.1, 0.0, 0.9, 0.2],
        [0.2, 0.1, 0.0, 0.1, 0.2, 0.8]
    ])

    # prepare the gaussian model with about 1000 samples
    from optimize.optimize_test import TestNLL

    gaussian_data = TestNLL(points=1000, regenerate=True, demo_parameters=[model_predictions, cov_matrix])
    advanced_gaussian_nll = gaussian_data

    # Call the fitting framework here and run it (and your algorithms)

    # Will do all the main code from here even the primary definition is in the other file for better readability
    # handle the gaussian function here
    gaussian_guess_vector = np.array([0, 0, 0, 0, 0, 0])
    validation_gaussian_fit = FittingFramework(negative_log_likelihood_gaussian, gaussian_guess_vector)
    from framework_implementation import MyFittingFramework

    implemented_gaussian_fit = MyFittingFramework(negative_log_likelihood_gaussian, gaussian_guess_vector)
    perform_framework_application(validation_gaussian_fit, name="gaussian_validation")
    perform_framework_application(implemented_gaussian_fit, name="gaussian_implementation")

    # handle the rosenbrock function here
    validation_rosen_fit = FittingFramework(negative_log_likelihood_rosenbrock, gaussian_guess_vector)
    implemented_rosen_fit = MyFittingFramework(negative_log_likelihood_rosenbrock, gaussian_guess_vector)
    perform_framework_application(validation_rosen_fit, name="rosenbrock_validation")
    perform_framework_application(implemented_rosen_fit, name="rosenbrock_implementation")
