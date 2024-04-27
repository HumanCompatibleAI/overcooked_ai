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
        "dill",
        "numpy",
        "scipy",
        "tqdm",
        "gym",
        "pettingzoo",
        "ipython",
        "pygame",
        "ipywidgets",
        "opencv-python",
    ],
    # removed overlapping dependencies
    extras_require={
        "harl": [
            "wandb",
            "GitPython",
            "memory_profiler",
            "sacred",
            "pymongo",
            "matplotlib",
            "requests",
            "seaborn==0.9.0",
            "ray[rllib]==2.0.0",
            "protobuf",
            "tensorflow==2.10",
        ]
    },
    entry_points={
        "console_scripts": [
            "overcooked-demo-up = overcooked_demo:start_server",
            "overcooked-demo-move = overcooked_demo:move_agent",
        ]
    },
)
