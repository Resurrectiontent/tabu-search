import numpy as np


def calc_pseudogradient_2p(point1, point2, val1, val2):
    # define a small step size for finite difference approximation
    h = 1e-6

    # calculate the difference vector between the two points
    diff_vect = point2 - point1

    # calculate the norm of the difference vector
    diff_norm = np.linalg.norm(diff_vect)

    # calculate the approximate gradient using finite difference
    gradient_approx = (val2 - val1) / diff_norm

    # calculate the pseudogradient using the approximation
    pseudogradient = gradient_approx * diff_vect / diff_norm

    return pseudogradient


def calc_pseudogradient_3p(point_arr, val_arr):
    # define a small step size for finite difference approximation
    h = 1e-6
    # calculate the number of dimensions
    dim = point_arr.shape[0]
    if dim == 1:
        # use forward difference for 1D case
        diff_vect = np.array([h])
        diff_norm = h
        gradient_approx = (val_arr[1] - val_arr[0]) / diff_norm
    else:
        # use central difference for higher dimensions
        middle_idx = int(dim / 2)
        point_0 = point_arr[:, middle_idx - 1]
        point_1 = point_arr[:, middle_idx]
        point_2 = point_arr[:, middle_idx + 1]
        val_0 = val_arr[middle_idx - 1]
        val_1 = val_arr[middle_idx]
        val_2 = val_arr[middle_idx + 1]
        diff_vect = point_2 - point_0
        diff_norm = np.linalg.norm(diff_vect)
        gradient_approx = (val_2 - val_0) / (2 * diff_norm)
    pseudogradient = gradient_approx * diff_vect / diff_norm
    return pseudogradient
