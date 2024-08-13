#%%
import ripser
from persim import PersistenceImager
import persim
import numpy as np
import matplotlib.pyplot as plt
from tda_for_phase_field.value_check import ValueCheck
from numpy.typing import NDArray
from typing import overload, NewType, TypeAlias

TdaDiagram = NewType("TdaDiagram", NDArray)
PersistentImage = NewType("PersistentImage", NDArray)

# class TDA():
#     def __init__


def plot_persistent_diagram(data: NDArray) -> None:
    """plot persistent diagram

    Args:
        data (NDArray): _description_
    """
    ValueCheck.check_ndarray_shape(data, [(1, 2)])
    dgm = ripser.ripser(data)["dgms"]
    persim.plot_diagrams(dgm, show=True)


@overload
def make_tda_diagram(datas: NDArray, dim0_hole: bool = True) -> TdaDiagram: ...
@overload
def make_tda_diagram(
    datas: list[NDArray], dim0_hole: bool = True
) -> list[TdaDiagram]: ...
def make_tda_diagram(
    datas: NDArray | list[NDArray], dim0_hole: bool = False
) -> TdaDiagram | list[TdaDiagram]:
    """calculate tda information

    Args:
        datas (NDArray | list[NDArray]): array of numpy.ndarray or their list.
        dim0_hole (bool, optional): plot dim0 information. Defaults to True.

    Returns:
        TdaDiagram | list[TdaDiagram]: _description_
    """
    rips = ripser.Rips(maxdim=1, coeff=2)
    if isinstance(datas, list):
        if dim0_hole:
            diagrams_h1 = []
            for data in datas:
                d = rips.fit_transform(data)
                d0 = d[0]
                d0 = d0[~np.isinf(d0).any(axis=1)]
                diagrams_h1.append(d0)
            print(diagrams_h1)
            # diagrams_h1 = [rips.fit_transform(data)[0] for data in datas]
        else:
            diagrams_h1 = [rips.fit_transform(data)[1] for data in datas]
        return diagrams_h1
    else:
        d = rips.fit_transform(datas)
        if dim0_hole:
            return d[0]
        else:
            return d[1]


@overload
def get_persistent_image_info(diagrams: list[TdaDiagram]) -> list[PersistentImage]: ...
@overload
def get_persistent_image_info(diagrams: TdaDiagram) -> PersistentImage: ...
def get_persistent_image_info(
    diagrams: TdaDiagram | list[TdaDiagram],
) -> PersistentImage | list[PersistentImage]:
    """calculate the necessary information to plot persistent image

    Args:
        diagrams (TdaDiagram | list[TdaDiagram]): _description_

    Returns:
        PersistentImage | list[PersistentImage]: _description_
    """

    if isinstance(diagrams, list):
        all([ValueCheck.check_ndarray_shape(data, [(1, 2)]) for data in diagrams])
    else:
        ValueCheck.check_ndarray_shape(diagrams, [(1, 2)])

    pimgr = PersistenceImager(pixel_size=0.8)
    pimgr.fit(diagrams)
    imgs = pimgr.transform(diagrams)
    print(f"PI Resolution = {pimgr.resolution}")
    return imgs


def plot_persistence_image(imgs: PersistentImage | list[PersistentImage]) -> None:
    """plot persistence image (PI)

    Args:
        imgs (PersistentImage | list[PersistentImage]): _description_
    """
    if isinstance(imgs, list):
        pimgr = PersistenceImager(pixel_size=0.8)
        for img in imgs:
            plt.figure(figsize=(12, 12))
            ax = plt.subplot(111)
            pimgr.plot_image(img, ax)
            plt.title("PI of $H_1$ for noise")
    else:
        pimgr = PersistenceImager(pixel_size=0.8)
        plt.figure(figsize=(12, 12))
        ax = plt.subplot(111)
        pimgr.plot_image(imgs, ax)
        plt.title("PI of $H_1$ for noise")


# a = make_tda_diagram(datas)
# imgs = get_persistent_image_info(a)
# plot_persistence_image(imgs[0])
# plot_persistence_image(imgs[-1])

# %%
