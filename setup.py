#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r", encoding='UTF8') as fh:
    long_description = fh.read()

setup(name='overcooked_ai',
      version='1.1.0',
      description='Cooperative multi-agent environment based on Overcooked',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Micah Carroll',
      author_email='mdc@berkeley.edu',
      url='https://github.com/HumanCompatibleAI/overcooked_ai',
      download_url='https://github.com/HumanCompatibleAI/overcooked_ai/archive/refs/tags/1.1.0.tar.gz',
      packages=find_packages('src'),
      keywords=['Overcooked', 'AI', 'Reinforcement Learning'],
      package_dir={"": "src"},
      package_data={
        'overcooked_ai_py' : [
          'data/layouts/*.layout', 'data/planners/*.py', 'data/human_data/*.pickle',
          'data/graphics/*.png', 'data/graphics/*.json', 'data/fonts/*.ttf',
        ],
      },
      install_requires=[
        'dill',
        'numpy',
        'scipy',
        'tqdm',
        'gym',
        'ipython',
        'pygame',
        'ipywidgets'
      ]
    )
