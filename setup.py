from setuptools import setup, find_packages
from typing import List
HYPHEN_DOT_E = '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open (file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]
        if HYPHEN_DOT_E in requirements:
            requirements.remove(HYPHEN_DOT_E)
    return requirements

setup(
    name = 'sensor_fault_detection_project',
    version= '0.0.2',
    author= 'Daizy',
    author_email= 'daizy@gmail.com',
    description= "sensor fault detection",
    install_requires= get_requirements('requirements.txt'),
    packages=find_packages()

)



