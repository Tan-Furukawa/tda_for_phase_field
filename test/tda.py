#%%
import numpy as np
import matplotlib.pyplot as plt
from phase_field_2d_ternary.matrix_plot_tools import Ternary
from tda_for_phase_field.tda import (
    make_tda_diagram,
    get_persistent_image_info,
    plot_persistence_image,
)
from tda_for_phase_field.random_sampling import (
    random_sampling_from_matrices,
    select_specific_phase,
    npMap,
)

con1 = np.load("test_data/output_2024-08-05-12-19-32/con1_60.npy")
con2 = np.load("test_data/output_2024-08-05-12-19-32/con2_60.npy")
res = random_sampling_from_matrices([con1, con2], 1000)
res = select_specific_phase(res, 1)
x = npMap(lambda x: x[0], res)
y = npMap(lambda x: x[1], res)

Ternary.imshow3(con1, con2)
plt.show()

a = make_tda_diagram(npMap(lambda x: [x[0], x[1]],res), False)
plt.scatter(npMap(lambda x: x[0], a), npMap(lambda x: x[1], a))
plt.show()
print("-----------")
print(res.shape)
print(a.shape)
print("-----------")
imgs = get_persistent_image_info(a)
plot_persistence_image(imgs)

# %%
