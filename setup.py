# using setup.py we can be able to build an entire machine learning application as a package and even deploy it to PyPI from where anyone can install and use it.

from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if '-e .' in requirements:    # this is used to remove the '-e .' from the requirements.txt file which is present when we install the package in editable mode
            requirements.remove('-e .')  # 
    return requirements


setup(
name='Mlproject',
version='0.0.1',
author='Keval',
author_email='thunnderrr10@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
 )