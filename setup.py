import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="PyGMRES",
    version="0.1.0",
    author="Johannes Blaschke",
    author_email="johannes@blaschke.science",
    description="A python implementation of the GMRES algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
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
