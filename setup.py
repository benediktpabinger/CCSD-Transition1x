from setuptools import find_packages, setup


setup(
    name="transition1x",
    version="1.0.0",
    packages=find_packages(),
    install_requires=['h5py'],
    extras_require={
        "example": ["ase","tqdm"],
    },
)
