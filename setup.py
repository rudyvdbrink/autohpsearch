from setuptools import setup, find_packages

setup(
    name="autohpsearch",
    version="0.1.0",
    description="A package for hyperparameter tuning of models for cross-sectional data.",
    author="Rudy van den Brink",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines() if line.strip()
    ],
    python_requires=">=3.6",
)