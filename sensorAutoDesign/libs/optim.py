import numpy as np
import tensorflow as tf
from tqdm import tqdm
from enum import Enum
from typing import Tuple
from dataclasses import dataclass

from .sensor import SensorProfile
from .solution import Solution, StateVariable

@dataclass(frozen=True)
class OptimConfig:
    optimizer: str
    epochs: int
    bandwidth_sampling_rate: float
    relationship_sampling_rate: float
    learning_rate: float

    initial_sensitivity_loss_weight: float
    initial_mean_squared_error_loss_weight: float
    initial_footprint_loss_weight: float

class TfOptimizer(Enum):
    ADAM: str = 'adam'
    STOCHASTIC_GRADIENT_DESCENT: str = 'sgd'

class Optim(tf.Module, Solution):
    def __init__(self, optim_config: OptimConfig, sensor_profile: SensorProfile) -> None:
        super().__init__()

        self.optim_config: OptimConfig = optim_config
        self.sensor_profile: SensorProfile = sensor_profile

        self.epoch: int = 0
        self._set_sensor_input()
        self._set_optimizer(self.optim_config.optimizer)  
        self._set_initial_loss_weights()

        for variable_name, bounds in self.sensor_profile.trainable_variables.items():
            self._set_trainable_variable(variable_name, bounds)

        Solution.__init__(self, len(self.trainable_variables), sensor_profile.bandwidth, self.optim_config.epochs)
        pass

    # -------------------------------------------------------------------------------------------
    """ INIT """
    def _set_sensor_input(self) -> None:
        self.sensor_input: tf.Tensor = tf.range(
            self.sensor_profile.bandwidth[0],
            self.sensor_profile.bandwidth[-1],
            1 / self.optim_config.bandwidth_sampling_rate,
            dtype=tf.float32
        )
        pass

    def _set_optimizer(self) -> None:
        match self.optim_config.optimizer:
            case TfOptimizer.ADAM:
                self.optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=self.optim_config.learning_rate)
            case TfOptimizer.STOCHASTIC_GRADIENT_DESCENT:
                self.optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.SGD(learning_rate=self.optim_config.learning_rate)                
            case _:
                raise ValueError(f"Unsupported optimizer: {self.optim_config.optimizer}")
        pass

    def _set_initial_loss_weights(self) -> None:
        loss_types: Tuple[str, str, str] = (
            StateVariable.SENSITIVITY_LOSS_WEIGHT, 
            StateVariable.MEAN_SQIUARED_ERROR_LOSS_WEIGHT, 
            StateVariable.FOOTPRINT_LOSS_WEIGHT
        )
        initial_values: Tuple[float, float, float] = (
            self.optim_config.initial_sensitivity_loss_weight,
            self.optim_config.initial_mean_squared_error_loss_weight,
            self.optim_config.initial_footprint_loss_weight
        )
        for loss_type, initial_value in zip(loss_types, initial_values):
            self._set_state_variable(loss_type, initial_value)
        pass

    def _set_trainable_variable(self, variable_name: str, bounds: Tuple[float, float]) -> None:
        variable: tf.Variable = tf.Variable(initial_value=(sum(bounds)/2), trainable=True)
        variable.bounds: Tuple[float, float] = bounds
        setattr(self, variable_name, variable)
        pass
    
    # -------------------------------------------------------------------------------------------
    """ CALL """
    def _get_mean_squared_error(self) -> tf.float32:
        response = self._get_state_variable(StateVariable.RESPONSE)
        mean_squared_error = tf.square(
            response - (self.input_range * self._get_state_variable(StateVariable.SENSITIVITY) + response[0])
        )/(self.sensor_input[-1] - self.sensor_input[0])
        self._set_state_variable(StateVariable.MEAN_SQUARED_ERROR, mean_squared_error)
        return mean_squared_error
    
    def _get_sensitivity(self) -> tf.float32:
        response = self._get_state_variable(StateVariable.RESPONSE)
        sensitivity = (response[-1]-response[0])/(self.sensor_input[-1]-self.sensor_input[0])
        self._set_state_variable(StateVariable.SENSITIVITY, sensitivity)
        return sensitivity
    
    def _get_loss(self):
        self._set_state_variable(
            StateVariable.RESPONSE, 
            self.sensor_profile._get_response(self.trainable_variables, self.sensor_input)
        )
        return -self._get_sensitivity() * self._get_state_variable(StateVariable.SENSITIVITY_LOSS_WEIGHT)\
        + self._get_mean_squared_error() * self._get_state_variable(StateVariable.MEAN_SQIUARED_ERROR_LOSS_WEIGHT)\
        + self.sensor_profile._get_footprint(self.trainable_variables) * self._get_state_variable(StateVariable.FOOTPRINT_LOSS_WEIGHT)

    def _enforce_parameter_relationships(self):
        for relationship in self.sensor_profile.parameter_relationships:
            if relationship.boolean_evaluation(self.trainable_variables):
                continue
            
            sampled_points: list = []
            for symbol in relationship.sympy_expression.free_symbols:
                bounds: Tuple[float, float] = self.sensor_profile.trainable_variables[str(symbol)]
                sampled_points.append(np.arange(bounds[0], bounds[-1], self.optim_config.relationship_sampling_rate))

        min_distance = np.inf
        closest_point = {}
        for values in np.nditer(np.meshgrid(*sampled_points)):
            current_point = {str(variables[i]): float(values[i]) for i in range(len(variables))}
            if relation.path(current_point):
                distance = sum((point[var] - current_point[var])**2 for var in point)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = current_point
        return closest_point

    def _clip_trainable_variables_to_boundaries(self):
        for var in sensor.trainable_variables:
            var.assign(tf.clip_by_value(
                var, 
                clip_value_min=var.bound[0], 
                clip_value_max=var.bound[-1]
            ))
        pass

    def _update_loss_component_weights(self):
        pass

    def _dereference_trainable_variables(self, trainable_variables: Tuple[tf.Variable, ...]) -> np.ndarray:
        return np.array([variable.numpy() for variable in trainable_variables], dtype=np.float32)

    def _train_step(self):
        self._set_state_variable(StateVariable.TRAINABLE_VARIABLES, self._dereference_trainable_variables(self.trainable_variables))

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
        
        self._set_state_variable(
            StateVariable.GRADIENTS, 
            tape.gradient(
                self._get_loss(), self.trainable_variables
            )
        )

        self.optimizer.apply_gradients(
            zip(
                self._get_state_variable(StateVariable.GRADIENTS), 
                self.trainable_variables
            )
        )

        self._enforce_parameter_relationships()
        self._clip_trainable_variables_to_boundaries()
        self._update_loss_component_weights()
        pass

    def __call__(self):
        for epoch in tqdm(range(self.epochs), desc="Fitting... "):
            self._set_epoch(epoch)
            self._train_step()
        pass
