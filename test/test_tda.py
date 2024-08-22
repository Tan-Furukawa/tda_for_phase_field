# %%
# import numpy as np
# import matplotlib.pyplot as plt
from phase_field_2d_ternary.matrix_plot_tools import Ternary
from tda_for_phase_field.tda import (
    make_tda_diagram,
    get_persistent_image_info,
)
from tda_for_phase_field.random_sampling import (
    random_sampling_from_matrices,
    select_specific_phase,
    npMap,
)
import matplotlib.pyplot as plt
import numpy as np
import os


data_file_path = os.path.join(os.path.dirname(__file__), "test_data")


def test_plot_tda() -> None:
    """とりあえずうまく動くことを確認するコード"""
    # assert True
    con1 = np.load(f"{data_file_path}/output_2024-08-05-12-19-32/con1_60.npy")
    con2 = np.load(f"{data_file_path}/output_2024-08-05-12-19-32/con2_60.npy")
    # con2 = np.array([[0.1, 0.2, 0.3], [0.4, 0.4, 0.5], [0.0, 0.1, 0.3]])
    np.random.seed(123)
    rcon = random_sampling_from_matrices([con1, con2], 1000)
    mat = np.array(rcon)  # mat: [x, y, val_con1, val_con2]
    assert mat.shape == (1000, 4)
    assert np.all(mat[:, 0:2] > 0)  # x, y should >0
    assert np.all(mat[:, 0:2] < len(mat))  # x, y should <len(res)
    assert np.all(
        (0 <= mat[:, 3:4]) & (mat[:, 3:4] < 1)
    )  # val_con1, val_con2 is between 0 to 1
    rcon1 = select_specific_phase(rcon, 1)
    r = make_tda_diagram(npMap(lambda x: [x[0], x[1]], rcon1))
    rr = get_persistent_image_info(r, birth_range=(0, 20), pers_range=(0, 20))
    # if __name__ == "__main__":
    #     print("--------------------------------------------------")
    #     print("persistent diagram of phase 1 (=phase B) of output_2024-08-05-12-19-32")
    #     plt.imshow(rr)
    #     plt.colorbar()
    #     plt.show()
    rr_ = get_persistent_image_info(
        [r, r], birth_range=(0, 20), pers_range=(0, 20)
    )  # リストを渡すとリストを返す。
    assert np.all(rr_[0] == rr)  # rrとrr_[0]は全く同一であるはず。


def test_tda_with_two_phase_system() -> None:
    """select specific phaseで相が見つけられなかったときのテスト"""
    con1 = np.load(f"{data_file_path}/output_2024-08-05-12-19-32/con1_60.npy")
    con2 = np.zeros_like(con1)  # 成分Bが全く含まれていないケース
    # if __name__ == "__main__":
    #     print("--------------------------------------------------")
    #     print("No phase B case")
    #     Ternary.imshow3(con1, con2)
    #     plt.show()
    np.random.seed(123)
    rcon = random_sampling_from_matrices([con1, con2], 1000)
    rcon1 = select_specific_phase(rcon, 1)
    assert len(rcon1) == 0  # there is no phase B
    r = make_tda_diagram(npMap(lambda x: [x[0], x[1]], rcon1))
    assert len(r) == 0  # there is no phase B

    # get_persistent_image_info にlistを渡さ「ない」とき----------------
    rr = get_persistent_image_info(r, birth_range=(0, 20), pers_range=(0, 20))
    assert np.all(rr == 0)  # persistent_image_infoは20*20の0ベクトルであるはず
    assert rr.shape == (20, 20)
    # if __name__ == "__main__":
    #     print("--------------------------------------------------")
    #     print("Persistent diagram")
    #     plt.imshow(rr)
    #     plt.colorbar()
    #     plt.show()

    # get_persistent_image_info にlistを渡すとき----------------
    rr_ = get_persistent_image_info([r, r], birth_range=(0, 20), pers_range=(0, 20))
    assert np.all(rr_[0] == 0) & np.all(rr_[1] == 0)

    # get_persistent_image_info にlistを渡すとき (2)----------------
    rcon2 = select_specific_phase(rcon, 2)
    r2 = make_tda_diagram(npMap(lambda x: [x[0], x[1]], rcon2))
    rr2_ = get_persistent_image_info([r, r2], birth_range=(0, 20), pers_range=(0, 20))
    assert np.all(rr2_[0] == 0)
    # print(pp[1])


if __name__ == "__main__":
    # test_plot_tda()
    test_tda_with_two_phase_system()
# %%
