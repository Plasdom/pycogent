import setuptools

setuptools.setup(
    name="pycogent",
    packages=setuptools.find_packages(),
    install_requires=["h5py", "matplotlib", "numpy", "scipy", "xarray"],
    author="Dominic Power",
    author_email="power8@llnl.gov",
    url="https://github.com/plasdom/pycogent",
    description="Some COGENT analysis tools",
    long_description=open("README.md").read(),
)
