# Active Inferants 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

Code for [Active inferants: The basis for an active inference framework for ant colony behavior](https://www.researchgate.net/publication/348003553_Active_inferants_The_basis_for_an_active_inference_framework_for_ant_colony_behavior) by Daniel Friedman, Alexander Tschantz, Maxwell Ramstead, Karl Friston, Axel Constant.

## Overview

This repository contains Python code that simulates ant colony behavior using an active inference framework. The main components are:

- `ants.py`: Defines the `Ant`, `Env`, and `MDP` classes, along with utility functions for plotting and saving results.
- `config.py`: Contains configuration parameters for the simulation.
- `figure_1.py`, `figure_2.py`, `figure_3.py`: Scripts to reproduce the figures from the paper.

## Usage

1. Install the required dependencies (numpy, matplotlib, imageio).
2. Run the desired figure script, e.g., `python figure_1.py`.
3. The generated images and data will be saved in the `imgs` directory.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Initial code by Alexander Tschantz for the 2021 paper. 
Subsequent rewriting and modifications by Daniel Friedman in 2024.
