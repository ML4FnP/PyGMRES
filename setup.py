import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="PyGMRES",
    version="0.1.1",
    author="Johannes Blaschke",
    author_email="jpblaschke@lbl.gov",
    description="A python implementation of the GMRES algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ML4FnP/PyGMRES",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
      'numpy',
    ],
    extra_requires={
        "compile": ["numba"]
    }
)
