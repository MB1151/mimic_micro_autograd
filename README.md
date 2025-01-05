# Mimic Andrej Karpathy's Micrograd

This repository implements a smaller version of autograd based on Andrej Karpathy's micrograd implementation. 

Please navigate to his original [video](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2) to understand the concepts better and circle back to this repository.

This repository contains two parallel sets of implementations for the same model:

- Step by Step implementation in the Jupiter notebooks where each line of code is explained in detail and executed.
- Modularized implementation of the same model in python scripts which is used to train a a very small toy dataset.

## Table Of Contents

- [Useful Resources](#useful-resources)
- [Repository Structure](#repository-structure)
- [Usage](#usage)

## Useful Resources

- <u>[Video](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2)</u> giving a walk through of micrograd implementation &rarr; *By Andrej Karpathy*
- <u>[Documentation](https://graphviz.readthedocs.io/en/stable/manual.html)</u> of the graphviz library used to visualize expressions &rarr; *By AT&T Labs*
- <u>[Google Doc](https://docs.google.com/document/d/1d4NJrhJxVw9sPNhDdS2xi7xfaVmpicQ3pZ3msplT9IY/edit?usp=sharing)</u> explains variable capturing in Python &rarr; *By Maneesh Babu Adhikari using Gemini*

## Repository Structure

This repository contains two parallel sets of implementations for the same model:

- Step by Step implementation in the Jupiter notebooks where each line of code is explained in detail and executed.
- Modularized implementation of the same model in python scripts which is used to train a toy model on toy dataset.

### `building_autograd_step_by_step/`

This directory contains step by step implementation, execution, and implementation of autograd.

- [`step_1_understanding_gradients.ipynb`](building_autograd_step_by_step/step_1_understanding_gradients.ipynb)
    * Shows how to calculate gradients wrt variables for any mathematical expression.
- [`step_2_visualizing_graphs.ipynb`](building_autograd_step_by_step/step_2_visualizing_graphs.ipynb)
    * Explores how to visualize mathematical expressions.
- [`step_3_object_expressions.ipynb`](building_autograd_step_by_step/step_3_object_expressions.ipynb)
    * Explores how to wrap variables in objects and overload operators to build expressions on top of these objects.
    * Shows how to visualize the mathematical expressions built using custom objects.
- [`step_4_calculating_gradients_chain_rule.ipynb`](building_autograd_step_by_step/step_4_calculating_gradients_chain_rule.ipynb)
    * Shows how to calculate gradients wrt variables in a mathematical expression using chain rule.
    * Shows how to visualize these gradients in the same expression graph.
- [`step_5_automating_gradient_calculation.ipynb`](building_autograd_step_by_step/step_5_automating_gradient_calculation.ipynb)
    * Implements the automation to calculate the gradients automatically similar to `backward` method in Pytorch.
- [`step_6_optional_mode_operations.ipynb`](building_autograd_step_by_step/step_6_optional_more_operations.ipynb)
    * Explores additional operations that can be used with object expressions and automates corresponding gradient calculations in the `backward` method.
- [`step_7_implement_neural_network.ipynb`](building_autograd_step_by_step/step_7_implement_neural_network.ipynb)
    * Implements a neural network using the object expressions and autograd developed in this repository.
- [`step_8_train_neural_network.ipynb`](building_autograd_step_by_step/step_8_train_neural_network.ipynb)
    * Trains a tiny neural network on different functions and explains the training process, and model behavior.

### `modularized_implementation`

This directory hosts the python scripts containing the same code from Jupyter notebooks but is modularized to be reused and training.

## Usage

### Model Overview

We train a neural network containing 2 hidden layers and 1 output layer. The hidden layers contains 4 neurons each and each of the neurons uses the same `tanh` activation function.

The dataset contains 4 data points and each data point contains 3 features. The true function to be learnt is a very simple *Even-Odd* function given by

$$ 
f(x, y, z) = 
\begin{cases} 
     1 & \text{if } x + y + z \text{ is even} \\
    -1 & \text{if } x + y + z \text{ is odd}
\end{cases}
$$

### Setup

Create a Virtual Environment for this project that will contain all the dependencies.

```python3 -m venv .autograd_venv```

Run the following command to install the necessary packages in the virtual environment.

```pip install -r requirements.txt```

The entry point to train a toy model is `modularized_implementation/train_model_main.py`.

Run the following command to train the model:

```python modularized_implementation/train_model_main.py```

It should take less than a second for the script to run. The value of this repository is going through step by step implementation of the code rather than actually running the script to train the toy model.