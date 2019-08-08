#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='overcooked_ai',
      version='0.0.1',
      description='Cooperative multi-agent environment based on Overcooked',
      author='Micah Carroll',
      author_email='micah.d.carroll@berkeley.edu',
      packages=find_packages(),
      install_requires=[
        'numpy',
        'tqdm',
        'ipykernel==5.1.0',
        'ipython==7.4.0',
        'ipython-genutils==0.2.0',
        'ipywidgets==7.4.2',
        'jupyter',
        'gym'
      ]
    )