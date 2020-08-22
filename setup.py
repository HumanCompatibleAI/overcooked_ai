#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='overcooked_ai',
      version='1.0.0',
      description='Cooperative multi-agent environment based on Overcooked',
      author='Micah Carroll',
      author_email='mdc@berkeley.edu',
      packages=find_packages('src'),
      package_dir={"": "src"},
      package_data={
        'overcooked_ai_py' : ['data/layouts/*.layout', 'data/planners/*.py', 'data/human_data/*.pickle']
      },
      install_requires=[
        'numpy',
        'tqdm',
        'gym',
        'ipython',
        'pygame'
      ]
    )