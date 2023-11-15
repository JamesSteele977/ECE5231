import numpy as np
import tensorflow as tf
from tqdm import tqdm
from types import SimpleNamespace
from enum import Enum, auto
from typing import Tuple
from dataclasses import dataclass

from .sensor import Sensor
from .solve import Solve

@dataclass(frozen=True)
class OptimConfig:
    optimizer: str
    epochs: int
    bandwidth_sampling_rate: float
    learning_rate: float

class TfOptimizers(Enum):
    ADAM: str = 'adam'
    STOCHASTIC_GRADIENT_DESCENT: str = 'sgd'

class Optim(tf.Module):
    def __init__(self, optim_config: OptimConfig) -> None:
        super().__init__()
        Solve.__init__(self)
        self.reset = optim_config
        match optim_config.optimizer:
            case TfOptimizers.ADAM:
                self.optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=optim_config.learning_rate)
            case TfOptimizers.STOCHASTIC_GRADIENT_DESCENT:
                self.optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.SGD(learning_rate=optim_config.learning_rate)
        self.epochs: int = optim_config.epochs
        self.bandwidth_sampling_rate: float = optim_config.bandwidth_sampling_rate
        pass

    """ LOSS """
    def _get_mse(self) -> tf.float32:
        return tf.square(
            self.response[self._s_('response')]-(self.tracked[self._s_(*domain+response[0])])\
                /(self.input[-1]-self.input[0])
            )
    
    def _get_factor(self) -> tf.float32:
        return (self.response[self._s_('response')][-1]-self.response[self._s_('response')][0])\
            /(self.input[-1]-self.input[0])
    
    def _get_loss(self):
        self.response[self._s_('response')] = self.sensor._get_IO()
        return -self._get_factor()*self.tracked[self._s_('lw_factor')]\
        +self._get_mse()*self.tracked[self._s_('lw_mse')]\
        +sensor._get_footprint()*self.tracked[self._s_('lw_fprint')]

    def _rels_(self, relation: SimpleNamespace, gradient, vars):
        if relation.eval:
            return vars
        rel_params = list(relation.path.free_symbols)
        sampled_points = []
        for param in rel_params:
            bound = getattr(self, param).bound
            sampled_points.append(np.arange(bound[1], bound[2], self.rel_fs))
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

    def _clip_(self):
        for var in sensor.trainable_variables:
            var.assign(tf.clip_by_value(
                var, 
                clip_value_min=var.bound[0], 
                clip_value_max=var.bound[-1]
            ))
        pass

    def _train_step(self, sensor):
        with tf.GradientTape() as tape:
            tape.watch(sensor.trainable_variables)
        self.pargrad[self._s_('grads')] = tape.gradient(
            self._get_loss(), sensor.trainable_variables
        )
        self.optimizer.apply_gradients(
            zip(self.pargrad[self._s_('grads')], sensor.trainable_variables)
        )
        self._rels_()
        self._clip_()
        self.pargrad[self._s_('params')] = sensor._get_tvars()
        pass

    def fit(self, sensor: Sensor):
        self.__init__(self.reset)
        for i in tqdm(range(self.epochs), desc="Fitting... "):
            self.epoch = i
            self._train_step(sensor)
        return self._get_solution()

def 
