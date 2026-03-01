
from setuptools import setup, find_packages # Thêm find_packages ở đây
import os

## extract info from requirements.txt for 'install_requires'
def parse_requirements(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), 'r') as f:
        lines = f.readlines()
    return [line for line in lines if len(line) > 0 and not line.startswith('#')]

setup(
    name='dl_template',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    author='Akiya Nguyen',
    author_email='nhphuoc2416@apcs.fitus.edu.vn',
)