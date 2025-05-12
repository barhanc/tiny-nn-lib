from setuptools import setup, find_packages

setup(
    name="tinynn",
    version="0.1",
    packages=find_packages(),
    license="MIT",
    description="Tiny neural network library written from scratch using only Numpy.",
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    python_requires=">=3.10",
)
