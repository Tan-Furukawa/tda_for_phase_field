# %%
import numpy as np
from tda_for_phase_field import SelectPhaseFromSamplingMatrix

# from phase_field_2d_ternary.matrix_plot_tools import Ternary
# from tda_for_phase_field.tda import (
#     make_tda_diagram,
#     get_persistent_image_info,
# )
# from tda_for_phase_field.random_sampling import (
#     random_sampling_from_matrices,
#     select_specific_phase,
#     npMap,
# )
# import matplotlib.pyplot as plt
# import numpy as np
import os
import pytest
import matplotlib.pyplot as plt

data_file_path = os.path.join(os.path.dirname(__file__), "test_data")


def test_select_specific_phase() -> None:
    con1 = np.load(f"{data_file_path}/output_2024-08-05-12-19-32/con1_60.npy")
    con2 = np.load(f"{data_file_path}/output_2024-08-05-12-19-32/con2_60.npy")
    with pytest.raises(ValueError) as e:
        sample = SelectPhaseFromSamplingMatrix([con1, con2, con1], 1000)
        sample.select_specific_phase(1)
    assert str(e.value) == "mat.shape[1] must be 3 or 4"

    sample = SelectPhaseFromSamplingMatrix([con1, con2], 1000)
    x0, y0 = sample.select_specific_phase_as_xy(0)
    x1, y1 = sample.select_specific_phase_as_xy(1)
    x2, y2 = sample.select_specific_phase_as_xy(2)
    # k = sample.select_specific_phase(0)
    print(sample.random_sampling_result)
    # print(k)

    if __name__ == "__main__":
        plt.scatter(x0, y0)
        plt.scatter(x1, y1)
        plt.scatter(x2, y2)
        plt.show()
    assert np.all((0 < x0) & (x0 < 128))
    assert np.all((0 < y0) & (y0 < 128))
    assert len(x0) + len(x1) + len(x2) == 1000


if __name__ == "__main__":
    test_select_specific_phase()

# %%
