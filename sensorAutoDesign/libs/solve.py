import numpy as np
from enum import Enum, auto
from typing import Tuple

class StateVariable(Enum):
    GRADIENTS = auto()
    TRAINABLE_VARIABLES = auto()
    LOSS = auto()
    SENSITIVITY = auto()
    MEAN_SQUARED_ERROR = auto()
    FOOTPRINT = auto()
    SENSITIVITY_LOSS_WEIGHT = auto()
    MEAN_SQIUARED_ERROR_LOSS_WEIGHT = auto()
    FOOTPRINT_LOSS_WEIGHT = auto()
    RESPONSE = auto()
    
class Solve():
    def __init__(
            self,
            n_trainable_variables: int,
            bandwidth: Tuple[float, float],
            bandwidth_sampling_rate: float,
            epochs: int
        ) -> None:
        self.n_trainable_variables: int = n_trainable_variables
        self.state_variables: np.ndarray = np.empty(
            (epochs, (n_trainable_variables * 2) + (bandwidth * bandwidth_sampling_rate) + 7),
            dtype=np.float32
        )
        pass

    def _get_state_variable(
            self, 
            epoch: int, 
            variable_type: StateVariable
            ) -> np.ndarray | np.float32:
        if (epoch < 0) or (epoch > self.state_variables.shape[0]):
            raise IndexError(f"Cannot get state variable from epoch {epoch}: out of range")
        if not isinstance(variable_type, StateVariable):
            raise TypeError(f"Argument 'variable_type' must be an instance of type StateVariable")
        loss_variables_starting_index = 2 * self.n_trainable_variables
        match variable_type:
            case StateVariable.GRADIENTS:
                return self.state_variables[epoch, 0:self.n_trainable_variables]
            case StateVariable.TRAINABLE_VARIABLES:
                return self.state_variables[epoch, self.n_trainable_variables:loss_variables_starting_index]
            case StateVariable.LOSS:
                return self.state_variables[epoch, loss_variables_starting_index + 1]
            case StateVariable.SENSITIVITY:
                return self.state_variables[epoch, loss_variables_starting_index + 2]
            case StateVariable.MEAN_SQUARED_ERROR:
                return self.state_variables[epoch, loss_variables_starting_index + 3]
            case StateVariable.FOOTPRINT:
                return self.state_variables[epoch, loss_variables_starting_index + 4]
            case StateVariable.SENSITIVITY_LOSS_WEIGHT:
                return self.state_variables[epoch, loss_variables_starting_index + 5]
            case StateVariable.MEAN_SQIUARED_ERROR_LOSS_WEIGHT:
                return self.state_variables[epoch, loss_variables_starting_index + 6]
            case StateVariable.FOOTPRINT_LOSS_WEIGHT:
                return self.state_variables[epoch, loss_variables_starting_index + 7]
            case StateVariable.RESPONSE:
                return self.state_variables[epoch, loss_variables_starting_index + 8:]