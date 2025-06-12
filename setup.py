from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="autohpsearch",
    version="0.6.0",
    author="rudyvdbrink",
    author_email="info@brinkdatascience.com",
    description="A package for hyperparameter tuning of models for cross-sectional data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rudyvdbrink/autohpsearch",
    packages=find_packages(
        include=["autohpsearch", "autohpsearch.*"]
    ),
    python_requires=">=3.7",
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    keywords="hyperparameter search autotuning machine-learning",
)