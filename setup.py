from setuptools import setup, find_packages

setup(
    name="tda_for_phase_field",
    version="0.0.3",
    description="tda analysis for 2d phase field simulation of non-elastic lamellae",
    author="Furukawa Tan",
    install_requires=["matplotlib", "numpy", "persim", "ripser"],
    packages=find_packages(),
    license="MIT",
)
