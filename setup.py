#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.md", "r", encoding="UTF8") as fh:
    long_description = fh.read()

setup(
    name="overcooked_ai",
    version="1.1.0",
    description="Cooperative multi-agent environment based on Overcooked",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Micah Carroll",
    author_email="mdc@berkeley.edu",
    url="https://github.com/HumanCompatibleAI/overcooked_ai",
    download_url="https://github.com/HumanCompatibleAI/overcooked_ai/archive/refs/tags/1.1.0.tar.gz",
    packages=find_packages("src"),
    keywords=["Overcooked", "AI", "Reinforcement Learning"],
    package_dir={"": "src"},
    package_data={
        "overcooked_ai_py": [
            "data/layouts/*.layout",
            "data/planners/*.py",
            "data/human_data/*.pickle",
            "data/graphics/*.png",
            "data/graphics/*.json",
            "data/fonts/*.ttf",
        ],
        "human_aware_rl": [
            "static/**/*.pickle",
            "static/**/*.csv",
            "ppo/trained_example/*.pkl",
            "ppo/trained_example/*.json",
            "ppo/trained_example/*/.is_checkpoint",
            "ppo/trained_example/*/.tune_metadata",
            "ppo/trained_example/*/checkpoint-500",
        ],
    },
    install_requires=[
        "dill>=0.4.1",
        "numpy>=2.4.4",
        "scipy>=1.17.1",
        "tqdm>=4.67.3",
        "gymnasium>=1.2.2",
        "ipython>=9.12.0",
        "pygame>=2.6.1",
        "ipywidgets>=8.1.8",
        "opencv-python>=4.13.0.92",
        "flask>=3.1.3",
        "flask-socketio>=5.6.1",
    ],
    # removed overlapping dependencies
    extras_require={
        "harl": [
            "wandb>=0.25.1",
            "GitPython>=3.1.46",
            "memory_profiler>=0.61.0",
            "sacred>=0.8.7",
            "pymongo>=4.16.0",
            "matplotlib>=3.10.8",
            "requests>=2.33.1",
            "seaborn>=0.13.2",
            "ray[rllib]>=2.54.1",
            "protobuf>=7.34.1",
            "tensorflow>=2.21.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "overcooked-demo-up = overcooked_demo:start_server",
            "overcooked-demo-move = overcooked_demo:move_agent",
        ]
    },
)
