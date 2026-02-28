# This is a sample Python script.
import argparse
import os
import sys

import numpy as np
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from findiff import coefficients
from numpy.random import default_rng

from derivatives.FindiffWrapper import basic_coefficients, DerivativeWrapper
from linalg.matrix import matrix_inverse_helper, matrix_inverse_second

generator = default_rng()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # setup an argparse instance to perform some task selections
    parser = argparse.ArgumentParser(description='Perform some tasks.')
    parser.add_argument('--apply_demo', action='store_true', help='apply a demo (default: %(default)s)')
    parser.add_argument('--generate_grid', action='store_true', help='generate a grid (default: %(default)s)')
    parser.add_argument('--check_inversion', action='store_false',
                        help='verify the matrix inversion algorithms by comparison to numpy results (default: %(default)s)')
    parser.add_argument('--check_decomposition', action='store_true',
                        help='verify the matrix decomposition algorithms by comparison to scipy results (default: %(default)s)')
    parser.add_argument('--generate_grid_coefficients', action='store_true',
                        help='generate the coefficients for the grid (Derivatives) (default: %(default)s)')
    parser.add_argument('--check_coefficients', action='store_false',
                        help='check the coefficients for the grid (Derivatives (default: %(default)s)')
    arguments = parser.parse_args()


    print_hi('PyCharm')
    print("first order")
    if arguments.generate_grid_coefficients:
        print(basic_coefficients(1))
        print(coefficients(deriv=1, acc=2))
        print(coefficients(deriv=1, acc=4))
        print(coefficients(deriv=1, acc=6))
        print(coefficients(deriv=1, acc=8))
        print(coefficients(deriv=1, acc=16))
        print("second order")
        print(basic_coefficients(2))
        print(coefficients(deriv=2, acc=2))
        print(coefficients(deriv=2, acc=4))
        print(coefficients(deriv=2, acc=6))
        print(coefficients(deriv=2, acc=8))
        print(coefficients(deriv=2, acc=16))

    if arguments.check_coefficients:
        for deriv in range(1, 10):
            coeffs_base = coefficients(deriv, 2)['center']
            coeffs = coeffs_base['coefficients']
            off = coeffs_base['offsets']
            print(deriv)
            print(off, "|", coeffs, "|", basic_coefficients(deriv))
            print('\n')

    # necessary to perform a test of the matrix decomposition system
    demonstration = generator.uniform(-10, 10, (4, 4))
    print(demonstration)
    if arguments.check_decomposition:
        from scipy.linalg import lu, lu_factor
        from linalg.matrix import lu_decomposition_helper

        lu_res, perm = lu_factor(demonstration)
        print("The factorization is:")
        print(lu_res)
        print(f"The permutation vector is: {perm}")

        print("scipy decomposition:")
        l_1, u_1 = lu(demonstration, permute_l=True)
        print(l_1)
        print(u_1)
        print(np.abs(l_1 @ u_1 - demonstration))
        print(np.allclose(l_1 @ u_1, demonstration))
        print("custom decomposition 1:")
        l_2, u_2 = lu_decomposition_helper(demonstration, False)
        print(l_2)
        print(u_2)
        print(np.abs(l_2 @ u_2 - demonstration))
        print(np.allclose(l_2 @ u_2, demonstration))
        print("custom decomposition 2:")
        l_3, u_3 = lu_decomposition_helper(demonstration, True)
        p3_ref, l3_ref, u3_ref = lu(demonstration)
        print(l_3)
        print(l3_ref)
        print(u_3)
        print(u3_ref)
        print(np.abs(l_3 @ u_3 - demonstration))
        print(np.allclose(l_3 @ u_3, demonstration))
        print(np.abs(l3_ref @ u3_ref - demonstration))
        print(np.allclose(l3_ref @ u3_ref, np.linalg.inv(p3_ref) @ demonstration))

    if arguments.apply_demo:
        demo = []
        demo.append(np.arange(10))
        demo.append(np.arange(10))
        # demo.append(np.arange(10))
        xx, yy = np.meshgrid(*demo)
        print(xx)
        print(yy)
        # from matplotlib import pyplot as plt
        # plt.plot(xx, yy, marker='o', color='k', linestyle='none')
        # plt.show()
        demo.append(np.arange(10))
        demo.append(np.arange(10))
        demo.append(np.arange(10))
        demo.append(np.arange(10))
        print(np.meshgrid(*demo))

    if arguments.check_inversion:
        print("numpy demonstrate inverse")
        numpy_inv_demonstration = np.linalg.inv(demonstration)
        print(numpy_inv_demonstration)
        scipy_inv_demonstration = matrix_inverse_helper(demonstration)
        print(scipy_inv_demonstration)
        own_inv_demonstration = matrix_inverse_second(demonstration)
        print(own_inv_demonstration)
        print("Check scipy base implementation:")
        print(np.allclose(numpy_inv_demonstration, scipy_inv_demonstration))
        print("Check own implementation:")
        print(np.allclose(numpy_inv_demonstration, own_inv_demonstration))

    DerivativeWrapper(2, 0.001, 2, 2, 0, additional_grid_point=20)

    print(sys.path)
    print(os.environ['PYTHON_ADD_PATH'])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
