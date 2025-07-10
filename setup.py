from setuptools import setup, find_packages

setup(
    name='tsp_benchmark',
    version='1.0.0',
    author="Amon-koike",
    packages=find_packages(),
    install_requires=["requests"],
    include_package_data=True,
)