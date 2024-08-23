# %%
import numpy as np
from tda_for_phase_field import PersistentDiagram
# from tda_for_phase_field.random_sampling import random_sampling_from_matrices


from tda_for_phase_field import SelectPhaseFromSamplingMatrix
import os

data_file_path = os.path.join(os.path.dirname(__file__), "test_data")


def test_hom0_diagram() -> None:  # 関数テストのためのメソッド
    con1 = np.load(f"{data_file_path}/output_2024-08-05-12-19-32/con1_60.npy")
    con2 = np.load(f"{data_file_path}/output_2024-08-05-12-19-32/con2_60.npy")
    # test1----------------------------------------------
    phase = SelectPhaseFromSamplingMatrix([con1, con2], 1000)
    x0, y0 = phase.select_specific_phase_as_xy(0)

    pd = PersistentDiagram(x0, y0)
    if __name__ == "__main__":
        hom0, _ = pd.get_persistent_image_info(plot=True)
    else:
        hom0, _ = pd.get_persistent_image_info(plot=False)
    assert np.ndim(hom0) == 1
    assert not np.all(hom0 == 0)


def test_hom1_diagram() -> None:  # 関数テストのためのメソッド
    con1 = np.load(f"{data_file_path}/output_2024-08-05-12-19-32/con1_60.npy")
    con2 = np.load(f"{data_file_path}/output_2024-08-05-12-19-32/con2_60.npy")
    # test1----------------------------------------------
    phase = SelectPhaseFromSamplingMatrix([con1, con2], 1000)
    x0, y0 = phase.select_specific_phase_as_xy(0)

    pd = PersistentDiagram(x0, y0)
    if __name__ == "__main__":
        _, hom1 = pd.get_persistent_image_info(plot=True)
    else:
        _, hom1 = pd.get_persistent_image_info(plot=False)
    assert not np.all(hom1 == 0)
    assert np.ndim(hom1) == 2
    # if __name__ == "__main__":
    #     pd.plot_hom1_persistent_image()
    #     plt.show()


def test_hom0_diagram_when_without_certain_phase() -> None:
    con1 = np.load(f"{data_file_path}/output_2024-08-05-12-19-32/con1_60.npy")
    con3 = np.zeros_like(con1)
    # test1----------------------------------------------
    phase = SelectPhaseFromSamplingMatrix([con1, con3], 1000)
    x0, y0 = phase.select_specific_phase_as_xy(0)
    x1, y1 = phase.select_specific_phase_as_xy(1)  # x1:[[]],y1:[[]]

    pd = PersistentDiagram(x1, y1)

    if __name__ == "__main__":
        hom0, _ = pd.get_persistent_image_info(plot=True)
    else:
        hom0, _ = pd.get_persistent_image_info(plot=False)
    assert np.all(hom0 == 0)
    assert np.ndim(hom0) == 1


def test_hom1_diagram_when_without_certain_phase() -> None:
    con1 = np.load(f"{data_file_path}/output_2024-08-05-12-19-32/con1_60.npy")
    con3 = np.zeros_like(con1)
    # test1----------------------------------------------
    phase = SelectPhaseFromSamplingMatrix([con1, con3], 1000)
    x0, y0 = phase.select_specific_phase_as_xy(0)
    x1, y1 = phase.select_specific_phase_as_xy(1)  # x1:[[]],y1:[[]]
    pd = PersistentDiagram(x0, y0)

    if __name__ == "__main__":
        _, hom1_0 = pd.get_persistent_image_info(plot=True)
    else:
        _, hom1_0 = pd.get_persistent_image_info(plot=False)
    assert not np.all(hom1_0 == 0)

    pd = PersistentDiagram(x1, y1)

    if __name__ == "__main__":
        _, hom1_2 = pd.get_persistent_image_info(plot=True)
    else:
        _, hom1_2 = pd.get_persistent_image_info(plot=False)
    assert np.all(hom1_2 == 0)


if __name__ == "__main__":
    test_hom0_diagram()
    test_hom1_diagram()
    test_hom0_diagram_when_without_certain_phase()
    test_hom1_diagram_when_without_certain_phase()
# %%
