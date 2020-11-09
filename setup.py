from setuptools import setup, find_packages


setup(name='ttpp',
      version='0.1.0',
      authors='Oleksandr Shchur, Nicholas Gao, Marin Bilos',
      description='Fast temporal point processes defined as transformations (triangular TPPs)',
      packages=find_packages('.'),
      zip_safe=False)
