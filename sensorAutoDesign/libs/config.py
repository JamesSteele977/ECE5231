from dataclasses_json import dataclass_json
from dataclasses import dataclass
from typing import Tuple, Dict, List

@dataclass_json
@dataclass(frozen=True)
class SensorConfig():
    trainable_variables: Dict[str, Tuple[float, float]]
    bandwidth: Tuple[float, float]
    input_symbol: str
    parameter_relationships: Tuple[str, ...]
    footprint: str
    response: str

@dataclass_json
@dataclass(frozen=True)
class SolutionSave():
    variable_names: List[str]
    trainable_variables: List[List[float]]
    gradients: List[float]
    loss: List[float]
    sensitivity: List[float]
    mean_squared_error: List[float]
    footprint: List[float]
    sensitivity_loss_weight: List[float]
    mean_squared_error_loss_weight: List[float]
    footprint_loss_weight: List[float]
    constraint_penalty: List[float]
    response: List[List[float]]

solutionSaveVariableNamesKey: str = 'variable_names'

@dataclass_json
@dataclass(frozen=True)
class OptimConfig:
    optimizer: str
    epochs: int
    bandwidth_sampling_rate: float
    learning_rate: float
    initial_sensitivity_loss_weight: float
    initial_mean_squared_error_loss_weight: float
    initial_footprint_loss_weight: float

