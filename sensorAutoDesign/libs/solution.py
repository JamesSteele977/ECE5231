import numpy as np
from enum import Enum, auto
from typing import Tuple

stateVarType: type = np.float32 | np.ndarray
class StateVariable(Enum):
    GRADIENTS = auto()
    TRAINABLE_VARIABLES = auto()
    LOSS = auto()
    SENSITIVITY = auto()
    MEAN_SQUARED_ERROR = auto()
    FOOTPRINT = auto()
    SENSITIVITY_LOSS_WEIGHT = auto()
    MEAN_SQUARED_ERROR_LOSS_WEIGHT = auto()
    FOOTPRINT_LOSS_WEIGHT = auto()
    CONSTRAINT_PENALTY = auto()
    RESPONSE = auto()
    
class Solution():
    def __init__(
            self,
            n_trainable_variables: int,
            bandwidth: Tuple[float, float],
            bandwidth_sampling_rate: float,
            epochs: int
        ) -> None:
        self.n_trainable_variables: int = n_trainable_variables
        self.state_variables: np.ndarray = np.empty(
            (epochs, (n_trainable_variables * 2) + int((bandwidth[-1]-bandwidth[0]) * bandwidth_sampling_rate) + 9),
            dtype=np.float32
        )
        self.epoch: int = 0
        pass

    def _state_variable_argument_check(self, variable_type: StateVariable) -> None:
        if (self.epoch < 0) or (self.epoch > self.state_variables.shape[0]):
            raise IndexError(f"Cannot get state variable from epoch {self.epoch}: out of range")
        if not isinstance(variable_type, StateVariable):
            raise TypeError(f"Argument 'variable_type' must be an instance of type StateVariable")
        pass

    def _get_state_variable_slice(self, variable_type: StateVariable, all_epochs: bool, epoch: int | None) -> slice:
        self._state_variable_argument_check(variable_type)
        loss_variables_starting_index: int = 2 * self.n_trainable_variables
        match variable_type:
            case StateVariable.GRADIENTS:
                query_slice: slice = np.s_[0:self.n_trainable_variables]
            case StateVariable.TRAINABLE_VARIABLES:
                query_slice: slice = np.s_[self.n_trainable_variables:loss_variables_starting_index]
            case StateVariable.LOSS:
                query_slice: slice = np.s_[loss_variables_starting_index + 1]
            case StateVariable.SENSITIVITY:
                query_slice: slice = np.s_[loss_variables_starting_index + 2]
            case StateVariable.MEAN_SQUARED_ERROR:
                query_slice: slice = np.s_[loss_variables_starting_index + 3]
            case StateVariable.FOOTPRINT:
                query_slice: slice = np.s_[loss_variables_starting_index + 4]
            case StateVariable.SENSITIVITY_LOSS_WEIGHT:
                query_slice: slice = np.s_[loss_variables_starting_index + 5]
            case StateVariable.MEAN_SQUARED_ERROR_LOSS_WEIGHT:
                query_slice: slice = np.s_[loss_variables_starting_index + 6]
            case StateVariable.FOOTPRINT_LOSS_WEIGHT:
                query_slice: slice = np.s_[loss_variables_starting_index + 7]
            case StateVariable.CONSTRAINT_PENALTY:
                query_slice: slice = np.s_[loss_variables_starting_index + 8]
            case StateVariable.RESPONSE:
                query_slice: slice = np.s_[loss_variables_starting_index + 9:]
        match all_epochs:
            case True:
                epoch_slice: slice = np.s_[:]
            case False:
                if epoch is None:
                    epoch_slice: slice = np.s_[self.epoch]
                else:
                    epoch_slice: slice = np.s_[epoch]
        return (epoch_slice, query_slice)

    def _get_state_variable(self, variable_type: StateVariable, all_epochs: bool = False, epoch: int | None = None) -> stateVarType:    
        return self.state_variables[self._get_state_variable_slice(variable_type, all_epochs, epoch)]
    
    def _set_state_variable(self, variable_type: StateVariable, value: stateVarType, epoch: int | None = None) -> None:
        self.state_variables[self._get_state_variable_slice(variable_type, False, epoch=epoch)]: stateVarType = value
        # print(f"{variable_type.name} set in epoch {self.epoch}:\t{value}")
        pass

    def _set_epoch(self, epoch: int):
        self.epoch: int = epoch
        pass
