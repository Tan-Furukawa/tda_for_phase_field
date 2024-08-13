# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Tuple, Iterable, Callable, NewType, overload
from phase_field_2d_ternary.matrix_plot_tools import Ternary
from functools import partial
from pipetools import pipe

# odd_sum(10)  # -> 25
# odd_sum = pipe | range | partial(filter, lambda x: x % 2) | sum
# odd_sum = pipe | range | where(lambda x: x % 2) | sumtxt


def npMap(func: Callable[..., object], iter: Iterable) -> NDArray:
    return np.array(list(map(func, iter)))


def npFilter(func: Callable[..., bool], iter: Iterable) -> NDArray:
    return np.array(list(filter(func, iter)))


RandomSamplingResult = NewType("RandomSamplingResult", NDArray[np.float64])


def random_sampling_from_matrices(
    mat: NDArray[np.float64] | list[NDArray[np.float64]], num: int
) -> RandomSamplingResult:
    """
    Randomly samples points from the matrix or list of matrices and performs linear interpolation to get the values at those points.

    Parameters:
    mat (Union[NDArray[np.float64], List[NDArray[np.float64]]]): Nx * Ny matrix or list of Nx * Ny matrices with float elements.
    num (int): Number of random samples.

    Returns:
    NDArray[np.float64]: Nx * Ny * (len(mat)+2) array with all float elements. The shape will be (num, 2+len(mat)),
                         where each row contains [x, y, mat1_value, mat2_value, ...].

    Example:
    >>> mat1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> mat2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    >>> random_sampling([mat1, mat2], 2)
    array([[0.5       , 0.5       , 2.5       , 6.5       ],
           [1.3       , 1.7       , 3.52      , 7.52      ]])
    """
    if isinstance(mat, list):
        matrices = mat
        Nx, Ny = matrices[0].shape
    else:
        matrices = [mat]
        Nx, Ny = mat.shape

    samples = np.zeros((num, 2 + len(matrices)), dtype=np.float64)

    for i in range(num):
        x, y = random_point(Nx, Ny)
        values = [interpolate(mat, x, y) for mat in matrices]
        samples[i] = [x, y] + values

    return RandomSamplingResult(samples)


def random_point(Nx: int, Ny: int) -> Tuple[float, float]:
    """
    Generates a random point within the given Nx by Ny range.

    Parameters:
    Nx (int): The number of rows in the matrix.
    Ny (int): The number of columns in the matrix.

    Returns:
    Tuple[float, float]: A tuple containing the x and y coordinates.
    """
    x = np.random.uniform(0, Nx - 1)
    y = np.random.uniform(0, Ny - 1)
    return x, y


def interpolate(mat: NDArray[np.float64], x: float, y: float) -> float:
    """
    Performs bilinear interpolation for the given matrix at the point (x, y).

    Parameters:
    mat (NDArray[np.float64]): The input matrix.
    x (float): The x coordinate.
    y (float): The y coordinate.

    Returns:
    float: The interpolated value at the point (x, y).
    """
    Nx, Ny = mat.shape

    x0, x1 = int(np.floor(x)), int(np.ceil(x))
    y0, y1 = int(np.floor(y)), int(np.ceil(y))

    if x0 == x1:
        x1 += 1 if x1 < Nx - 1 else -1
    if y0 == y1:
        y1 += 1 if y1 < Ny - 1 else -1

    Q11 = mat[x0, y0]
    Q12 = mat[x0, y1]
    Q21 = mat[x1, y0]
    Q22 = mat[x1, y1]

    return (
        Q11 * (x1 - x) * (y1 - y)
        + Q21 * (x - x0) * (y1 - y)
        + Q12 * (x1 - x) * (y - y0)
        + Q22 * (x - x0) * (y - y0)
    )


@overload
def select_specific_phase(
    res: RandomSamplingResult, phase: int
) -> RandomSamplingResult: ...
@overload
def select_specific_phase(
    res: RandomSamplingResult, phase: list[int]
) -> RandomSamplingResult: ...


def select_specific_phase(
    res: RandomSamplingResult, phase: int | list[int]
) -> RandomSamplingResult:
    if res.shape[1] == 4:

        def get_main_phase(a: NDArray) -> int:
            return int(np.argmax(np.array([a[2], a[3], 1 - a[2] - a[3]])))
    elif res.shape[1] == 3:

        def get_main_phase(a: NDArray) -> int:
            return int(np.argmax(np.array([a[2], 1 - a[2]])))
    else:
        TypeError("res.shape[1] must be 3 or 4")
    if isinstance(phase, int):
        return RandomSamplingResult(npFilter(lambda x: get_main_phase(x) == phase, res))
    elif isinstance(phase, list):
        return RandomSamplingResult(npFilter(lambda x: get_main_phase(x) in phase, res))
    else:
        raise TypeError("phase must be int or list[int]")

if __name__ == "__main__":

    con1 = np.load("result/output_2024-08-05-12-19-32/con1_60.npy")
    con2 = np.load("result/output_2024-08-05-12-19-32/con2_60.npy")
    res = random_sampling_from_matrices([con1, con2], 10000)
    res = select_specific_phase(res, 1)
    x = npMap(lambda x: x[0], res)
    y = npMap(lambda x: x[1], res)

    Ternary.imshow3(con1, con2)
    plt.show()
    plt.scatter(y, x, s=2)
    plt.show()

# %%
