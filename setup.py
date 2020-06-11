from setuptools import setup, find_packages

install_requires = [
    "tf2rl>=0.1.12",
    "tensorflow-cpu>=2.2.0",
    "gym",
    "numpy",
    "jupyter",
    "pandas",
    "pickle5",
    "cloudpickle==1.3.0",
    "matplotlib",
    "argparse",
    "pylint",
]

setup(
    name="hwr",
    version="0.0.1",
    author="Katharina Hermann, Ferenc Török",
    author_email="katharina.hermann@tum.de, ferike.trk@gmail.com",
    packages=find_packages("."),
    licence="LICENCE.txt",
    url="https://github.com/tum-adlr-ss20-06/project",
    install_requires=install_requires,
)