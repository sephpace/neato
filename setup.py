
import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='neato',
    version='1.0.0',
    description='An implementation of NEAT (NeuroEvolution of Augmenting Topologies) by Kenneth O. Stanley',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/sephpace/neato',
    author='Seph Pace',
    author_email='sephpace@gmail.com',
    classifiers=[
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3.7',
    ],
    packages=['neato'],
    include_package_data=True,
    install_requires=['numpy'],
)
