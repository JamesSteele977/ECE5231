from numpy import ndarray, float32, empty, s_

from enum import Enum, auto
from typing import Tuple

stateVarType: type = float32 | ndarray

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
    """
    Solution object, for storage of and operation on optimization process data, such as
    gradients, loss, loss components, response function, etc.

    Attributes
    ----------
    - n_trainable_varialbes (int): Number of design parameters in specified design
    - state_variables (np.ndarray): Numpy array of size (n_epochs, 2*n_tvars + 
    response_function_size + 9). Stores full summary of optimizer state for each epoch.
    - epoch (int): Current epoch in optimization loop

    Methods
    -------
    __init__():
        parameters:
            n_trainable_varialbes (int): Number of design parameters in specified design
            bandwidth (Tuple[float, float]): Input range for sensor response
            bandwidth_sampling_rate (float): Sampling rate for calculation of response
            function
            epochs (int): Total number of training epochs
        returns:
            self (Solution)
    
    _get_state_variable_slice(): Returns proper index for target attribute in state_variables
    array
        parameters:
            variable_type (StateVariable): Target solution attribute
            all_epochs (bool): Set True for full slice of state_variables across all epochs
            epoch (int | None): Optional. For retrieval of data from specific epoch
        returns:
            state_variables_slice (slice): Index of target data
    
    _get_state_variable(): Retrieves target data from state_variables array
        parameters: 
            variable_type (StateVariable): Target solution attribute
            all_epochs (bool): Set True for full slice of state_variables across all epochs
                default: False
            epoch (int | None): Optional. For retrieval of data from specific epoch
                default: None
        returns:
            state_variable (np.ndarray | float): Target data from state_variables
    
    _set_state_variable(): Sets target data in state_variables array
        parameters:
            variable_type (StateVariable): Target solution attribute
            value (stateVarType): Data to store as state variable
            epoch (int | None): Optional. For assignation of data in specific epoch
                default: None
        returns:
            [none]
    
    _set_epoch(): Sets current epoch in optimization loop. For Use in get/set state variable
    functions
        parameters:
            epoch (int): Current epoch
        returns:
            [none]

    """
    def __init__(
            self,
            n_trainable_variables: int,
            bandwidth: Tuple[float, float],
            bandwidth_sampling_rate: float,
            epochs: int
        ) -> None:
        self.n_trainable_variables: int = n_trainable_variables
        self.state_variables: ndarray = empty(
            (epochs, (n_trainable_variables * 2) + int((bandwidth[-1]-bandwidth[0]) * bandwidth_sampling_rate) + 9),
            dtype=float32
        )
        self.epoch: int = 0
        pass

    def _get_state_variable_slice(self, variable_type: StateVariable, all_epochs: bool, epoch: int | None) -> slice:
        loss_variables_starting_index: int = 2 * self.n_trainable_variables
        match variable_type:
            case StateVariable.GRADIENTS:
                query_slice: slice = s_[0:self.n_trainable_variables]
            case StateVariable.TRAINABLE_VARIABLES:
                query_slice: slice = s_[self.n_trainable_variables:loss_variables_starting_index]
            case StateVariable.LOSS:
                query_slice: slice = s_[loss_variables_starting_index + 1]
            case StateVariable.SENSITIVITY:
                query_slice: slice = s_[loss_variables_starting_index + 2]
            case StateVariable.MEAN_SQUARED_ERROR:
                query_slice: slice = s_[loss_variables_starting_index + 3]
            case StateVariable.FOOTPRINT:
                query_slice: slice = s_[loss_variables_starting_index + 4]
            case StateVariable.SENSITIVITY_LOSS_WEIGHT:
                query_slice: slice = s_[loss_variables_starting_index + 5]
            case StateVariable.MEAN_SQUARED_ERROR_LOSS_WEIGHT:
                query_slice: slice = s_[loss_variables_starting_index + 6]
            case StateVariable.FOOTPRINT_LOSS_WEIGHT:
                query_slice: slice = s_[loss_variables_starting_index + 7]
            case StateVariable.CONSTRAINT_PENALTY:
                query_slice: slice = s_[loss_variables_starting_index + 8]
            case StateVariable.RESPONSE:
                query_slice: slice = s_[loss_variables_starting_index + 9:]
        match all_epochs:
            case True:
                epoch_slice: slice = s_[:]
            case False:
                if epoch is None:
                    epoch_slice: slice = s_[self.epoch]
                else:
                    epoch_slice: slice = s_[epoch]
        return (epoch_slice, query_slice)

    def _get_state_variable(self, variable_type: StateVariable, all_epochs: bool = False, epoch: int | None = None) -> stateVarType:    
        return self.state_variables[self._get_state_variable_slice(variable_type, all_epochs, epoch)]
    
    def _set_state_variable(self, variable_type: StateVariable, value: stateVarType, epoch: int | None = None) -> None:
        self.state_variables[self._get_state_variable_slice(variable_type, False, epoch=epoch)]: stateVarType = value
        pass

    def _set_epoch(self, epoch: int):
        self.epoch: int = epoch
        pass
