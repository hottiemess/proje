import numpy as np
from matplotlib import pyplot as plt

from framework import FittingFramework, main
from optimize.GradientDescent import GradientDescent, DescentAlgorithms


class MyFittingFramework(FittingFramework):
    # this is a subclass of the fitting framework dedicated to performing the implementation tasks!

    # the kind of provided function will not trivially work with my implementation => we will need to map it!
    # luckily I already have some experience with mapping this two kinds of expected arguments to each other!

    def __init__(self, cost_function, initial_guess):
        super().__init__(cost_function, initial_guess)
        # mapping in this class is no longer necessary as it was already done in my delegation class.
        # self.effective_cost_function = np.vectorize(self.negative_log_likelihood)

        # now: as a last step detect the dimension of the problem and initialise the self-defined framework used for
        # the particular fitting operation!
        # FIXME: derivative_method needs to be changed to a self-defined implementation!
        self.user_minimizer = GradientDescent(self.mapped_cost_function, 6, [], "numdifftools")

    def minimize(self):
        # to make sure that the calls to the other methods of the framework won't will calculate the minimum using migrad once
        super().minimize()

        # perform the minimisation process
        result, f_min, converged = self.user_minimizer.general_gradient_actor(self.initial_guess,
                                                                              algo=DescentAlgorithms.BFGS)

        # TODO: iminuit will set here a ValueView for the parameter estimations! => should adpat the return and extraction of our implementation
        self.best_fit = self.user_minimizer.values()

        # perhaps it would be nice to also load this results in the minuit class to use our estimations for the error estimations!
        self.m._fmin = f_min

    n_plot_points = 1000

    # TODO: it is still necessary to implement our visulatisation here!
    def instantiate_plot(self, additional_point=5, plot_args=None, **kwargs):
        """
        instantiate_plot

        @author Dominik Fischer
        @date 2026-02-22

        Helper function to instantiate/create the plot which should visualize the steps of the algorithm during the minimization process.
        :param additional_point: number of additional points to be added to the plot.
        :param plot_args: arguments to be passed to the plot function.
        :return: figure and axis object of the plot.
        """
        # this is only a outer wrapper to determine the plot to be drawn!
        fig, ax = plt.subplots(figsize=(10, 10), **plot_args)
        # extract the trace of points the algorithm has visited
        data_trace = self.user_minimizer.extract_intermediate_results()
        intermediate_approximations = data_trace[:, 2]
        # will only account for the first two dimensions when drawing
        intermediate_x_values = intermediate_approximations[:, 0]
        intermediate_y_values = intermediate_approximations[:, 1]
        ax.plot(intermediate_x_values, intermediate_y_values, label="Minimization Trace")
        x_data = np.linspace(np.min(intermediate_x_values) - additional_point,
                             np.max(intermediate_x_values) + additional_point, self.n_plot_points)
        y_data = np.linspace(np.min(intermediate_y_values) - additional_point,
                             np.max(intermediate_y_values) + additional_point, self.n_plot_points)
        data_blocks = [np.vstack([np.full(self.n_plot_points, x) for x in x_data]).T,
                       np.vstack([np.full(self.n_plot_points, y) for y in y_data])]
        data_blocks.extend([np.full((self.n_plot_points, self.n_plot_points), self.best_fit[i]) for i in
                            range(len(self.best_fit - 2))])
        nll_arguments = np.array(data_blocks)
        nll_values = self.negative_log_likelihood(nll_arguments)

        # will need a heat map to visualize the nll around the found minimum
        ax.contour(x_data, y_data, nll_values)
        return fig, ax


# If you call this script with python, then your code starts here
if __name__ == "__main__":
    main(MyFittingFramework)
