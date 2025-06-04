from setuptools import setup, find_packages

setup(
    name="autohpsearch",
    version="0.2.0",
    description="A package for hyperparameter tuning of models for cross-sectional data.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rudyvdbrink/autohpsearch",
    author="Rudy van den Brink",
    packages=find_packages(where="autohpsearch"),
    package_dir={"": "autohpsearch"},
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines() if line.strip()
    ],
    python_requires=">=3.6",
)