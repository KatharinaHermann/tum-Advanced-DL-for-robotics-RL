# Advanced Deep Learning for Robotics Project

This is a project of the course Advanced Deep Learning in Robotics at the Technical University of Munich. Our chosen topic is to develop a method to train a Neural Motion Planner (NMP) agent with Reinforcement Learning (RL)  in an environment with randomly placed obstacles. The aim of the method is to reduce training time as much as possible meanwhile using a very simple and sparse reward function. For this we use Hindsight Experience Replay (HER), but we extend it by not just assigning a new goal, but also a new workspace to the unsuccessful trajectories.

## Installation

Requirements:
* `python=3.7`
* `numpy=1.18.1`
* `os=0.1.4`
* `importlib_resources=1.5.0`
* `pandas=1.0.3`
* `tensorflow-gpu=2.2.0` / `tensorflow-cpu=2.2.0`
* `gym=0.15.3`
* `pickle5=0.0.9`
* `matplotlib=3.2.1`
* `pylint=2.5.2`
* `jupyter=1.0.0`
* `tf2rl=0.1.12`
* `argparse=1.4.0`

The environment can be installed with the following commands:
* With GPU:
    * create environment and install all packages: `conda env create --name <name> -f environment_gpu.yml`
    * activate environment: `conda activate <name>`
* With CPU:
    * create environment and install all packages: `conda env create --name <name> -f environment_cpu.yml`
    * activate environment: `conda activate <name>`
* Installing the custon gym and the 'hwr' package in development mode:
    * `pip install -e gym-pointrobo/`
    * `python setup.py develop`
