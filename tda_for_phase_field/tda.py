# %%
import ripser
from persim import PersistenceImager
import persim
import numpy as np
import matplotlib.pyplot as plt
from tda_for_phase_field.value_check import ValueCheck
from numpy.typing import NDArray
from typing import overload, NewType, TypeAlias, Any
# from tda_for_phase_field.random_sampling import random_sampling_from_matrices


class PersistentDiagram:
    """
    A class to compute and visualize persistent diagrams and persistent images for topological data analysis.

    This class handles the computation of persistent homology for 0-dimensional and 1-dimensional features
    from a set of input data points. It generates persistent diagrams and transforms them into persistent images
    for further analysis or visualization.

    Attributes:
        birth_range (tuple[float, float]): The range of birth times to be considered for the persistent images.
        pers_range (tuple[float, float]): The range of persistence times to be considered for the persistent images.
        pixcel_size (float): The size of each pixel in the persistent image.
        xy (NDArray): A 2D array combining the input x and y coordinates.
        hom0_diagram (list[list[float]]): The persistent diagram for 0-dimensional homology features.
        hom0_image_info (NDArray): The persistent image for 0-dimensional homology features.
        hom1_diagram (list[list[float]]): The persistent diagram for 1-dimensional homology features.
        hom1_image_info (NDArray): The persistent image for 1-dimensional homology features.
        persistent_image_size_for_small_dataset (tuple[int, int]): The size of the output persistent images for small datasets.
        data_length_threshold (int): The minimum number of data points required to compute persistent homology.

    Methods:
        __init__(x, y, data_length_threshold=100, birth_range=(0, 30), pers_range=(0, 30), pixcel_size=0.8):
            Initializes the PersistentDiagram object with the provided parameters.
        get_persistent_image_info(plot=True):
            Computes and returns the persistent images for both 0-dimensional and 1-dimensional features.
        make_hom0_diagram():
            Computes the persistent diagram for 0-dimensional features using the Ripser library.
        calc_hom0_persistent_image_info():
            Transforms the 0-dimensional persistent diagram into a persistent image and returns it.
        make_hom1_diagram():
            Computes the persistent diagram for 1-dimensional features using the Ripser library.
        calc_hom1_persistent_image_info():
            Transforms the 1-dimensional persistent diagram into a persistent image and returns it.
        plot_hom0_persistent_image():
            Plots the persistent image for 0-dimensional features.
        plot_hom1_persistent_image():
            Plots the persistent image for 1-dimensional features.
    """

    def __init__(
        self,
        x: NDArray,
        y: NDArray,
        data_length_threshold: int = 100,
        birth_range: tuple[float, float] = (0, 30),
        pers_range: tuple[float, float] = (0, 30),
        pixcel_size: float = 0.8,
    ) -> None:
        """
        Initializes the PersistentDiagram object.

        Args:
            x (NDArray): The x-coordinates of the input data points.
            y (NDArray): The y-coordinates of the input data points.
            data_length_threshold (int, optional): The minimum number of data points required to compute persistent homology.
                Defaults to 100.
            birth_range (tuple[float, float], optional): The range of birth times to consider for the persistent images.
                Defaults to (0, 30).
            pers_range (tuple[float, float], optional): The range of persistence times to consider for the persistent images.
                Defaults to (0, 30).
            pixcel_size (float, optional): The size of each pixel in the persistent image. Defaults to 0.8.

        Raises:
            ValueError: If the lengths of `x` and `y` are not the same.
            ValueError: If the dimensions of `x` or `y` are not 1D.
        """
        if not len(x) == len(y):
            raise ValueError("x and y must have same length")
        if not np.ndim(x) == 1:
            raise ValueError("the dimension of x must be 1")
        if not np.ndim(y) == 1:
            raise ValueError("the dimension of y must be 1")

        self.birth_range = birth_range
        self.pers_range = pers_range
        self.pixcel_size = pixcel_size

        self.xy = np.column_stack([x, y])
        self.hom0_diagram: list[list[float]] = [[]]
        self.hom0_image_info: NDArray = np.array([[]])
        self.hom1_diagram: list[list[float]] = []
        self.hom1_image_info: NDArray = np.array([])

        self.persistent_image_size_for_small_dataset = (10, 10)
        self.data_length_threshold = data_length_threshold

    def get_persistent_image_info(self, plot: bool = True) -> tuple[NDArray, NDArray]:
        """
        Computes and returns the persistent images for both 0-dimensional and 1-dimensional features.

        Args:
            plot (bool, optional): If True, the persistent images will be plotted. Defaults to True.

        Returns:
            tuple[NDArray, NDArray]: The persistent images for 0-dimensional and 1-dimensional features.
        """
        self.make_hom0_diagram()
        hom0 = self.calc_hom0_persistent_image_info()

        self.make_hom1_diagram()
        hom1 = self.calc_hom1_persistent_image_info()
        if plot:
            self.plot_hom0_persistent_image()
            plt.show()
            self.plot_hom1_persistent_image()
            plt.show()
        return hom0, hom1

    def make_hom0_diagram(self) -> list[list[float]]:
        """
        Computes the persistent diagram for 0-dimensional features using the Ripser library.

        Returns:
            list[list[float]]: The 0-dimensional persistent diagram.
        """
        rips = ripser.Rips(maxdim=1, coeff=2)
        if len(self.xy) > self.data_length_threshold:
            d = rips.fit_transform(self.xy)
            d0 = d[0]
            d0 = d0[~np.isinf(d0).any(axis=1)]
            self.hom0_diagram = d0
            return d0
        else:
            self.hom0_diagram = []
            return []

    def calc_hom0_persistent_image_info(self) -> NDArray:
        """
        Transforms the 0-dimensional persistent diagram into a persistent image.

        Returns:
            NDArray: The persistent image for 0-dimensional features.
        """
        if len(self.hom0_diagram) == 0:
            zeros = np.zeros((self.persistent_image_size_for_small_dataset[0],))
            self.hom0_image_info = zeros
            return zeros
        else:
            pimgr = PersistenceImager(pixel_size=self.pixcel_size)
            pimgr.fit(self.hom0_diagram)
            pimgr.birth_range = self.birth_range
            pimgr.pers_range = self.pers_range
            res = pimgr.transform(self.hom0_diagram)
            self.hom0_image_info = res[0, :]
            return res[0, :]

    def make_hom1_diagram(self) -> list[list[float]]:
        """
        Computes the persistent diagram for 1-dimensional features using the Ripser library.

        Returns:
            list[list[float]]: The 1-dimensional persistent diagram.
        """
        rips = ripser.Rips(maxdim=1, coeff=2)
        if len(self.xy) > self.data_length_threshold:
            self.hom1_diagram = rips.fit_transform(self.xy)[1]
            return self.hom1_diagram
        else:
            self.hom1_diagram = [[]]
            return [[]]

    def calc_hom1_persistent_image_info(self) -> NDArray:
        """
        Transforms the 1-dimensional persistent diagram into a persistent image.

        Returns:
            NDArray: The persistent image for 1-dimensional features.
        """
        if len(self.hom1_diagram) == 1:
            zeros = np.zeros(
                (
                    self.persistent_image_size_for_small_dataset[0],
                    self.persistent_image_size_for_small_dataset[1],
                )
            )
            self.hom1_image_info = zeros
            return zeros
        else:
            pimgr = PersistenceImager(pixel_size=self.pixcel_size)
            pimgr.fit(self.hom1_diagram)
            pimgr.birth_range = self.birth_range
            pimgr.pers_range = self.pers_range
            res = pimgr.transform(self.hom1_diagram)
            self.hom1_image_info = res
            return res

    def plot_hom0_persistent_image(self) -> None:
        """
        Plots the persistent image for 0-dimensional features.
        """
        plt.imshow(self.hom0_image_info.reshape(-1, 1), origin="lower")

    def plot_hom1_persistent_image(self) -> None:
        """
        Plots the persistent image for 1-dimensional features.
        """
        plt.imshow(self.hom1_image_info.transpose(), origin="lower")
