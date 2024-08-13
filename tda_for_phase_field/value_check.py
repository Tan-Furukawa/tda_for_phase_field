# %%
from numpy.typing import NDArray
import numpy as np


class ValueCheck:
    @staticmethod
    def judge_ndarray_shape(
        arr: NDArray, index_len_list: list[tuple[int, int]]
    ) -> bool:
        """check shape of ndarray

        Args:
            arr (NDArray): input array
            index_len_list (list[tuple[int, int]]): [(index of array.shape, required length of the array at index of array.shape)]

        Example:
        >>> ValueCheck.judge_ndarray_shape(np.array([[1, 2, 3], [1, 2, 3]]), [(1, 2)])
        """
        shape = list(arr.shape)
        return all(
            [shape[index_len[0]] == index_len[1] for index_len in index_len_list]
        )

    @staticmethod
    def check_ndarray_shape(
        arr: NDArray, index_len_list: list[tuple[int, int]]
    ) -> None:
        """check shape of ndarray

        Args:
            arr (NDArray): input array
            index_len_list (list[tuple[int, int]]): [(index of array.shape, required length of the array at index of array.shape)]

        Example:
        >>> ValueCheck.check_ndarray_shape(np.array([[1, 2, 3], [1, 2, 3]]), [(1, 2)])

        """
        shape = list(arr.shape)
        for index_len in index_len_list:
            if not (shape[index_len[0]] == index_len[1]):
                raise ValueError(
                    f"Invalid shape of array: the array.shape[{index_len[0]}] should {index_len[1]} but {shape[index_len[0]]}"
                )


# ValueCheck.check_ndarray_shape(np.array([[1, 2, 3], [1, 2, 3]]), [(1, 4)])
# ValueCheck.judge_ndarray_shape(np.array([[1, 2, 3], [1, 2, 3]]), [(1, 3), (0, 2)])
