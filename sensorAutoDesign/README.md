# Sensor Design Optimization Shell
## Overview
<p>The Sensor Design Optimization Shell is an interactive command-line interface designed for optimizing sensor designs.<br>
It facilitates the configuration of sensors and optimizers and allows users to view the results of the optimization process.</p>

## Getting Started
### Prerequisites
<p>Ensure you have the following installed:</p>

- Python 3.10 or higher
- Necessary Python libraries: 
    - Tensorflow, NumPy
    - Matplotlib
    - Sympy
    - tqdm
    - dataclasses-json

### Installation
<p>Clone the repository or download the source code to your local machine.<br>
Navigate to the project directory in your command line interface.</p>

### Running the Shell
<p>To start the shell, execute:</p>

```bash
$ python main_script.py
```
<p>Replace main_script.py with the path to the Python file that launches the shell.</p>

## Shell Commands
<p>The shell supports the following commands:</p>

- `clear`: Clears the terminal.
- `exit`: Exits the shell.
- `readme`: Displays detailed instructions.
- `list` [ObjectType]: Lists objects of a specified type (Sensor, Optimizer, Solution).
- `delete` [ObjectType] [Name]: Deletes a specified object.
- `configure` [ObjectType] [-n [Name]]: Configures or creates an object. Optionally specify a name.
- `fit` [OptimizerName] [SensorName]: Runs optimization for a sensor design.
- `display` [SolutionName] [options]: Displays optimization results.

## Creating and Configuring Objects
<p>To create a sensor or optimizer:</p>

```bash
$ configure Sensor -n MySensor
```
<p>Follow the prompts to set properties.</p>

### JSON Config Format
<p>JSON configuration files are used to set up sensors and optimizers.<br> 
Here are the formatting guidelines:</p>

#### Sensors:
```json
{
  "trainable_variables": {
    "variable_1": [lower_bound, upper_bound],
    "variable_2": [lower_bound, upper_bound]
  },
  "bandwidth": [lower_input_bound, upper_input_bound],
  "input_symbol": "input_sym_string",
  "parameter_relationship": ["(equality/relationship_1)", "(equality/relatinoship_2)"],
  "footprint": "expression",
  "response": "expression_with_input_sym"
}
```
<p>variable_1, variable_2, ... are the names of the trainable variables.<br>
lower_bound and upper_bound are the numerical bounds for each variable.</p>

#### Sensors Example:
```json
{
  "trainable_variables": {
    "length": [0.5, 10.0],
    "width": [0.5, 5.0]
  },
  "bandwidth": [1.0, 100.0],
  "input_symbol": "frequency",
  "parameter_relationships": ["length > 2 * width"],
  "footprint": "length * width",
  "response": "length + width * frequency"
}
```

#### Optimizers Example:
```json
{
    "optimizer": "optimizer_name",
    "epochs": 10,
    "bandwidth_sampling_rate": 100.0,
    "learning_rate": 0.01,
    "initial_sensitivity_loss_weight": 1.0,
    "initial_mean_squared_error_loss_weight": 1.0,
    "initial_footprint_loss_weight": 1.0
}
```
<p>Current version (1.0) supports "ADAM" and "SGD" optimizers.</p>

## Running an Optimization
<p>Ensure you have a sensor and optimizer configured:</p>

```console
optimtool$ fit MyOptimizer MySensor
```

### Viewing Results
<p>To view optimization results:</p>

```console
optimtool$ display MySolution --all
```

## Additional Information
<p>The shell supports tab completion for commands.<br>
Use the <strong>help</strong> command for more details on specific commands.</p>