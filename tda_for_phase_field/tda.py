# %%
import ripser
from persim import PersistenceImager
import persim
import numpy as np
import matplotlib.pyplot as plt
from tda_for_phase_field.value_check import ValueCheck
from numpy.typing import NDArray
from typing import overload, NewType, TypeAlias, Any

TdaDiagram = NewType("TdaDiagram", NDArray)
PersistentImage = NewType("PersistentImage", NDArray)


def plot_persistent_diagram(data: NDArray) -> None:
    """plot persistent diagram

    Args:
        data (NDArray): _description_
    """
    ValueCheck.check_ndarray_shape(data, [(1, 2)])
    dgm = ripser.ripser(data)["dgms"]
    persim.plot_diagrams(dgm, show=True)


def plot_persistent_image(
    datas: NDArray | list[NDArray], **kwargs: Any
) -> PersistentImage | list[PersistentImage]:
    tda_diagram = make_tda_diagram(datas)
    imgs = get_persistent_image_info(tda_diagram, **kwargs)
    plot_persistence_image_from_tda_diagram(imgs)
    return imgs


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
def get_persistent_image_info(
    diagrams: list[TdaDiagram], **kwargs: Any
) -> list[PersistentImage]: ...
@overload
def get_persistent_image_info(
    diagrams: TdaDiagram, **kwargs: Any
) -> PersistentImage: ...
def get_persistent_image_info(
    diagrams: TdaDiagram | list[TdaDiagram], **kwargs: Any
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

    pimgr = PersistenceImager(**kwargs)
    pimgr.fit(diagrams)

    if "birth_range" in kwargs.keys():
        pimgr.birth_range = kwargs["birth_range"]

    if "pers_range" in kwargs.keys():
        pimgr.pers_range = kwargs["pers_range"]
    imgs = pimgr.transform(diagrams)
    print(f"PI Resolution = {pimgr.resolution}")
    return imgs


def plot_persistence_image_from_tda_diagram(
    imgs: PersistentImage | list[PersistentImage],
) -> None:
    """plot persistence image (PI)

    Args:
        imgs (PersistentImage | list[PersistentImage]): _description_
    """
    if isinstance(imgs, list):
        pimgr = PersistenceImager()
        for img in imgs:
            plt.figure(figsize=(12, 12))
            ax = plt.subplot(111)
            pimgr.plot_image(img, ax)
            plt.title("PI of $H_1$ for noise")
    else:
        pimgr = PersistenceImager()
        plt.figure(figsize=(12, 12))
        ax = plt.subplot(111)
        pimgr.plot_image(imgs, ax)
        plt.title("PI of $H_1$ for noise")


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from phase_field_2d_ternary.matrix_plot_tools import Ternary
    from tda_for_phase_field.random_sampling import (
        random_sampling_from_matrices,
        select_specific_phase,
        npMap,
    )

    con1 = np.load("../test/test_data/output_2024-08-05-12-19-32/con1_60.npy")
    con2 = np.load("../test/test_data/output_2024-08-05-12-19-32/con2_60.npy")
    np.random.seed(124)
    r = random_sampling_from_matrices([con1, con2], 1000)
    rr = select_specific_phase(r, 1)
    x = npMap(lambda x: x[0], rr)
    y = npMap(lambda x: x[1], rr)
    d = npMap(lambda x: [x[0], x[1]], rr)

    # plot_persistent_diagram(d)
    # plt.show()

    w = 20
    imgs = plot_persistent_image(
        d, birth_range=(0, w), pers_range=(0, w), pixel_size=0.5
    )
    # %%
    # from ripser import Rips

    # rips = Rips(maxdim=1, coeff=2)
    # k = rips.fit_transform(d)[1]

    # pimgr = PersistenceImager(pixel_size=0.1, birth_range=(0, 1))
    # pimgr.fit(k)
    # pimgr.birth_range = (0, 20)
    # pimgr.pers_range = (0, 20)

    # imgs = pimgr.transform(k)
    # ax = plt.subplot(111)
    # pimgr.plot_image(imgs, ax)
    # plt.title("PI of $H_1$ for noise")
