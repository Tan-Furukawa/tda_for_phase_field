from setuptools import setup, find_packages

setup(
    name='tda_for_phase_field',  # パッケージ名（pip listで表示される）
    version="0.0.3",  # バージョン
    description="tda analysis for 2d phase field simulation of non-elastic lamellae",  # 説明
    author='Furukawa Tan',  # 作者名
    packages=find_packages(),  # 使うモジュール一覧を指定する
    license='MIT'  # ライセンス
)