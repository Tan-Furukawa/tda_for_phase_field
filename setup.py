from setuptools import setup, find_packages

setup(
    name="tda_for_phase_field",
    version="0.1.0",
    description="tda analysis for 2d phase field simulation of non-elastic lamellae",
    author="Furukawa Tan",
    install_requires=[
        "matplotlib",
        "numpy",
        "persim",
        "ripser",
        "phase_field_2d_ternary",
    ],
    dependency_links=[
        "git+https://github.com/Tan-Furukawa/phase-field-2d-ternay.git@master#egg=phase_field_2d_ternary",
    ],
    packages=find_packages(),
    license="MIT",
)
