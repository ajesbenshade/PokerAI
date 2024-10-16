# Poker Simulation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![GitHub Issues](https://img.shields.io/github/issues/yourusername/poker_simulation)
![GitHub Stars](https://img.shields.io/github/stars/yourusername/poker_simulation?style=social)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Overview

**Poker Simulation** is a Python-based project that simulates Texas Hold'em poker games. It leverages reinforcement learning to train agents, evaluate strategies, and analyze gameplay dynamics. The simulation environment is built using the `gym` library, and the agents utilize neural networks implemented with `PyTorch` to make strategic decisions.

This project aims to provide a comprehensive framework for studying poker strategies, testing AI agents, and conducting large-scale simulations to understand game dynamics better.

## Features

- **Modular Architecture:** Organized into distinct modules for classes, utilities, simulation, and environment management.
- **Reinforcement Learning Agents:** Agents trained using neural networks to make informed betting decisions.
- **Comprehensive Evaluation:** Tools to evaluate agent performance, select top-performing players, and analyze average pot winnings.
- **Extensible Framework:** Easily add new strategies, agents, or modify game rules.
- **Logging and Reporting:** Detailed logs and reports to track simulation progress and outcomes.
- **Unit Testing:** Ensures code reliability and correctness through extensive unit tests.

## Installation

### Prerequisites

- **Python 3.8 or higher**: Ensure you have Python installed. You can download it from [here](https://www.python.org/downloads/).
- **Git**: To clone the repository. Download from [here](https://git-scm.com/downloads).

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/poker_simulation.git
   cd poker_simulation
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**

   **On Unix/Linux/MacOS:**

   ```bash
   source venv/bin/activate
   ```

   **On Windows:**

   ```bash
   venv\Scripts\activate
   ```

4. **Install Dependencies**

   Install the required dependencies from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not present, you can manually install the necessary packages:

   ```bash
   pip install gym torch numpy pandas treys tqdm
   ```

## Usage

### Running Simulations

To run a simulation of poker games, execute the `main.py` script. You can specify parameters such as the number of generations, games per generation, and more.

```bash
python main.py --simulate
```

Example with custom parameters:

```bash
python main.py --simulate --generations 50 --games 1000 --chunk_size 100
```

### Training Agents

To train poker agents using reinforcement learning, utilize the training scripts provided in the `simulation/` directory.

```bash
python simulation/train_agents.py
```

### Evaluating Agents

Evaluate the performance of trained agents using the evaluation tools.

```bash
python utils/evaluation.py
```

> **Note**: Replace with actual script paths and commands based on your implementation.

## Project Structure

```markdown
poker_simulation/
│
├── classes/
│   ├── __init__.py
│   ├── player.py
│   ├── card.py
│   └── game_state.py
│
├── environment/
│   ├── __init__.py
│   └── poker_env.py
│
├── simulation/
│   ├── __init__.py
│   ├── simulate.py
│   └── train_agents.py
│
├── utils/
│   ├── __init__.py
│   ├── helper_functions.py
│   ├── training.py
│   └── evaluation.py
│
├── tests/
│   ├── __init__.py
│   ├── test_game_state.py
│   └── test_evaluation.py
│
├── logs/
│   └── simulation.log
│
├── main.py
├── requirements.txt
├── README.md
└── LICENSE
```

### Description

- **`classes/`**: Contains core classes such as `Player`, `Card`, and `GameState` that define the fundamental components of the poker game.
- **`environment/`**: Defines the simulation environment using `gym`, encapsulating the game logic and state management.
- **`simulation/`**: Houses scripts related to running simulations, training agents, and generating game data.
- **`utils/`**: Utility modules for helper functions, training utilities, and evaluation tools.
- **`tests/`**: Unit tests to ensure code reliability and correctness.
- **`logs/`**: Directory for storing log files generated during simulations.
- **`main.py`**: The entry point for running simulations and managing the workflow.
- **`requirements.txt`**: Lists all Python dependencies required for the project.
- **`README.md`**: Documentation of the project.
- **`LICENSE`**: Licensing information.

## Running Tests

Unit tests are provided to validate the functionality of various components.

To run all tests, execute:

```bash
python -m unittest discover tests
```

Alternatively, use `pytest` if preferred:

### Install pytest

```bash
pip install pytest
```

### Run Tests

```bash
pytest
```

## Contributing

Contributions are welcome! Follow these steps to contribute:

### Fork the Repository

### Create a New Branch

```bash
git checkout -b feature/YourFeatureName
```

### Make Your Changes

### Commit Your Changes

```bash
git commit -m "Add some feature"
```

### Push to the Branch

```bash
git push origin feature/YourFeatureName
```

### Open a Pull Request

Please ensure your code adheres to the project's coding standards and passes all tests.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute it as per the license terms.

## Contact

**Aaron Esbenshade**  
Email: [ajesbenshade@outlook.com](mailto:ajesbenshade@outlook.com)  
GitHub: [ajesbenshade](https://github.com/ajesbenshade)  
LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/aaron-esbenshade-5227a81a/)

Feel free to reach out for questions, suggestions, or collaboration opportunities.

## Acknowledgments

- **Gym** for providing the reinforcement learning environment framework.
- **PyTorch** for the deep learning library.
- **Treys** for poker hand evaluation.
- **TQDM** for progress bars in simulations.
- **OpenAI** for their contributions to AI and machine learning.
