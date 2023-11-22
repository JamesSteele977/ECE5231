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
        self.n_constraints_violated: int = 0

        for variable_name, bounds in self.sensor_profile.trainable_variables.items():
            self._set_trainable_variable(variable_name, bounds)

        Solution.__init__(
            self,
            n_trainable_variables=len(self.trainable_variables),
            bandwidth=self.sensor_profile.bandwidth,
            bandwidth_sampling_rate=self.optim_config.bandwidth_sampling_rate,
            epochs=self.optim_config.epochs
        )

        self._set_sensor_input()
        self._set_optimizer()  
        self._set_initial_loss_weights()
        pass

    # -------------------------------------------------------------------------------------------
    """ INIT """
    def _set_sensor_input(self) -> None:
        self.sensor_input: tf.Tensor = tf.range(
            self.sensor_profile.bandwidth[0],
            self.sensor_profile.bandwidth[-1],
            1 / self.optim_config.bandwidth_sampling_rate,
            dtype=tf.float32,
            name=self.sensor_profile.input_symbol
        )
        pass

    def _set_optimizer(self) -> None:
        match self.optim_config.optimizer:
            case TfOptimizer.ADAM.value:
                self.optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=self.optim_config.learning_rate)
            case TfOptimizer.STOCHASTIC_GRADIENT_DESCENT.value:
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
        variable: tf.Variable = tf.Variable(name=variable_name, initial_value=(sum(bounds)/2), trainable=True)
        variable.bounds: Tuple[float, float] = bounds
        setattr(self, variable_name, variable)
        pass
    
    # -------------------------------------------------------------------------------------------
    """ CALL """
    # Loss Function
    def _get_mean_squared_error(self) -> tf.float32:
        response: tf.Tensor = tf.convert_to_tensor(
            self._get_state_variable(StateVariable.RESPONSE),
            dtype=tf.float32
        )
        mean_squared_error: tf.float32 = tf.reduce_mean(tf.square(
            response - (self.sensor_input * self._get_state_variable(StateVariable.SENSITIVITY) + response[0])
        )/(self.sensor_input[-1] - self.sensor_input[0]))
        self._set_state_variable(StateVariable.MEAN_SQUARED_ERROR, mean_squared_error.numpy())
        return mean_squared_error
    
    def _get_sensitivity(self) -> tf.float32:
        response: tf.Tensor = tf.convert_to_tensor(
            self._get_state_variable(StateVariable.RESPONSE),
            dtype=tf.float32
        )
        sensitivity: tf.float32 = (response[-1]-response[0])/(self.sensor_input[-1]-self.sensor_input[0])
        self._set_state_variable(StateVariable.SENSITIVITY, sensitivity.numpy())
        return sensitivity
    
    def _get_loss(self) -> tf.float32:
        self._set_n_constraints_violated()
        self._set_state_variable(
            StateVariable.RESPONSE, 
            self.sensor_profile._get_response(self.trainable_variables, self.sensor_input).numpy()
        )

        unscaled_loss: tf.float32 = -self._get_sensitivity() * self._get_state_variable(StateVariable.SENSITIVITY_LOSS_WEIGHT)\
        + self._get_mean_squared_error() * self._get_state_variable(StateVariable.MEAN_SQIUARED_ERROR_LOSS_WEIGHT)\
        + self.sensor_profile._get_footprint(self.trainable_variables) * self._get_state_variable(StateVariable.FOOTPRINT_LOSS_WEIGHT)

        loss: tf.float32 = unscaled_loss + (self.n_constraints_violated * tf.abs(unscaled_loss))
        self._set_state_variable(StateVariable.LOSS, loss.numpy())
        return loss

    # Constrained Optimization
    def _set_n_constraints_violated(self) -> None:
        self.n_constraints_violated = 0
        for relationship in self.sensor_profile.parameter_relationships:
            if relationship.boolean_evaluation(self.trainable_variables):
                self.n_constraints_violated += 1
        pass
            
    def _clip_trainable_variables_to_boundaries(self) -> None:
        for var in self.trainable_variables:
            var.assign(tf.clip_by_value(
                var, 
                clip_value_min=var.bounds[0], 
                clip_value_max=var.bounds[-1]
            ))
        pass

    def _dereference_tf_tuple(self, trainable_variables: Tuple[tf.Variable, ...] | Tuple[tf.Tensor, ...]) -> np.ndarray:
        return np.array([variable.numpy() for variable in trainable_variables], dtype=np.float32)

    # Optimization Loop
    def _train_step(self) -> None:
        self._set_state_variable(StateVariable.TRAINABLE_VARIABLES, self._dereference_tf_tuple(self.trainable_variables))

        with tf.GradientTape() as tape:
            loss: tf.float32 = self._get_loss()
        
        gradient: Tuple[tf.Tensor, ...] = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))

        self._set_state_variable(StateVariable.GRADIENTS, self._dereference_tf_tuple(gradient))
        self._clip_trainable_variables_to_boundaries()
        pass

    def __call__(self) -> None:
        for epoch in tqdm(range(self.optim_config.epochs), desc="Fitting... "):
            self._set_epoch(epoch)
            self._train_step()
        pass
