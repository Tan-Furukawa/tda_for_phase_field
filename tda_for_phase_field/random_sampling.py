# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Tuple, Iterable, Callable, NewType, overload
from phase_field_2d_ternary.matrix_plot_tools import Ternary
# from functools import partial
# from pipetools import pipe

# odd_sum(10)  # -> 25
# odd_sum = pipe | range | partial(filter, lambda x: x % 2) | sum
# odd_sum = pipe | range | where(lambda x: x % 2) | sumtxt


def npMap(func: Callable[..., object], iter: Iterable) -> NDArray:
    return np.array(list(map(func, iter)))


def npFilter(func: Callable[..., bool], iter: Iterable) -> NDArray:
    return np.array(list(filter(func, iter)))


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


def random_point(Nx: int, Ny: int, seed: int = 123) -> Tuple[float, float]:
    """
    Generates a random point within the given Nx by Ny range.

    Parameters:
        Nx (int): The number of rows in the matrix.
        Ny (int): The number of columns in the matrix.

    Returns:
        Tuple[float, float]: A tuple containing the x and y coordinates.
    """

    np.random.seed(seed)
    x = np.random.uniform(0, Nx - 1)
    y = np.random.uniform(0, Ny - 1)
    return x, y


class SamplingFromMatrix:
    def __init__(
        self, matrix_list: list[NDArray], num: int | None = None, seed: int = 123
    ) -> None:
        if len(matrix_list) == 0:
            raise ValueError("length of matrix_list should larger than 0")
        if np.all([len(mat.shape) != 2 for mat in matrix_list]):
            raise ValueError("the dimension of matrix should 2")
        if not np.all([matrix_list[0].shape == mat.shape for mat in matrix_list]):
            raise ValueError("all matrix should have same shape")

        self.matrix_list = matrix_list
        self.random_sampling_result: NDArray | None = None
        self.seed = seed
        # if mat.shape[0] != mat.shape[1]:
        #     raise ValueError("mat should N * N matrix")
        if num is not None:
            self.random_sampling_result = self.random_sampling_from_matrices(num)

    def random_sampling_from_matrices(self, num: int) -> NDArray:
        """
        Randomly samples points from the matrix or list of matrices and performs linear interpolation to get the values at those points.

        Parameters:
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
        samples = np.zeros((num, 2 + len(self.matrix_list)), dtype=np.float64)

        Nx, Ny = self.matrix_list[0].shape
        for i in range(num):
            x, y = random_point(Nx, Ny, self.seed + i)
            values = [interpolate(mat, x, y) for mat in self.matrix_list]
            samples[i] = [x, y] + values

        self.random_sampling_result = samples
        return samples


class SelectPhaseFromSamplingMatrix(SamplingFromMatrix):
    def __init__(
        self, matrix_list: list[NDArray], num: int | None = None, seed: int = 123
    ):
        super().__init__(matrix_list, num, seed)

    def select_specific_phase_as_xy(self, phase: int) -> tuple[NDArray, NDArray]:
        res = self.select_specific_phase(phase)
        x = npMap(lambda x: x[0], res)
        y = npMap(lambda x: x[1], res)
        return x, y

    def select_specific_phase(self, phase: int) -> NDArray:
        """filter the matrix whose type is RandomSamplingResult and extract specific phase.

        Args:
            phase (int | list[int]): phase id or list of them. phase id is 0, 1 in binary and 0, 1, 2 in ternary.

        Raises:
            TypeError: input matrix shape error

        Returns:
            RandomSamplingResult: list[[x, y, value of mat1, value of mat2,...]]
        """
        if self.random_sampling_result is None:
            raise ValueError(
                "use random_sampling_from_matrices before select_specific_phase"
            )

        # if Ternary
        # print(self.random_sampling_result.shape[1])
        if self.random_sampling_result.shape[1] == 4:
            if phase not in [0, 1, 2]:
                raise ValueError("phase value should 0 or 1 or 2")

            def get_main_phase(a: NDArray) -> int:
                return int(np.argmax(np.array([a[2], a[3], 1 - a[2] - a[3]])))

        # if Binary
        elif self.random_sampling_result.shape[1] == 3:
            if phase not in [0, 1]:
                raise ValueError("phase value should 0 or 1")

            def get_main_phase(a: NDArray) -> int:
                return int(np.argmax(np.array([a[2], 1 - a[2]])))
        else:
            raise ValueError("mat.shape[1] must be 3 or 4")
        return npFilter(
            lambda x: get_main_phase(x) == phase, self.random_sampling_result
        )


# if __name__ == "__main__":


#     con1 = np.load("../test/test_data/output_2024-08-05-12-19-32/con1_60.npy")
#     con2 = np.load("../test/test_data/output_2024-08-05-12-19-32/con2_60.npy")
#     # SamplingFromMatrixは[con1] か[con1, con2]の形であるべきである。
#     with pytest.raises(ValueError) as e:
#         sample = SelectPhaseFromSamplingMatrix([con1, con2, con1], 1000)
#         sample.select_specific_phase(1)
#     assert str(e.value) == "mat.shape[1] must be 3 or 4"

#     sample = SelectPhaseFromSamplingMatrix([con1, con2], 1000)
#     x0, y0 = sample.select_specific_phase_as_xy(0)
#     x1, y1 = sample.select_specific_phase_as_xy(1)
#     x2, y2 = sample.select_specific_phase_as_xy(2)
#     assert np.all((0 < x0) & (x0 < 128))
#     assert np.all((0 < y0) & (y0 < 128))
#     assert len(x0) + len(x1) + len(x2) == 1000

#     # Ternary.imshow3(con1, con2)
#     # plt.show()
#     # plt.scatter(y, x, s=2)
#     # plt.show()

# %%
