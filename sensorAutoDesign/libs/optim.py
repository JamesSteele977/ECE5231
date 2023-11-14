import numpy as np
import tensorflow as tf
from tqdm import tqdm
from types import SimpleNamespace

from .sensor import Sensor

optim_case = lambda x: {
    'adam': tf.keras.optimizers.Adam,
    'sgd': tf.keras.optimizers.SGD
}.get(x)

class Optim(tf.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.reset = config
        self.optimizer = optim_case(config['optimizer'].lower())(learning_rate=float(config['learning_rate']))
        self.epochs = int(config['epochs'])
        self.dI = float(config['dI'])
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
        +self.sensor._get_footprint()*self.tracked[self._s_('lw_fprint')]

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
        for var in self.sensor.trainable_variables:
            var.assign(tf.clip_by_value(
                var, 
                clip_value_min=var.bound[0], 
                clip_value_max=var.bound[-1]
            ))
        pass

    def _train_step(self):
        with tf.GradientTape() as tape:
            tape.watch(self.sensor.trainable_variables)
        self.pargrad[self._s_('grads')] = tape.gradient(
            self._get_loss(), self.sensor.trainable_variables
        )
        self.optimizer.apply_gradients(
            zip(self.pargrad[self._s_('grads')], self.sensor.trainable_variables)
        )
        self._rels_()
        self._clip_()
        self.pargrad[self._s_('params')] = self.sensor._get_tvars()
        pass

    def fit(self, sensor: Sensor):
        self.__init__(self.reset)
        setattr(self, 'sensor', sensor)
        Solve.__init__(self)
        for i in tqdm(range(self.epochs), desc="Fitting... "):
            self.epoch = i
            self._train_step()
        return self._get_solution()

class Solve():
    def __init__(self) -> None:
        # Array of tracked variables (loss, factor, mse, footprint, loss_weights(3))
        self.tracked = np.empty((self.epochs, 7), dtype=np.float32)
        # Input range based on bandwidth and sampling rate
        self.input = np.arange(
            self.sensor.bandwidth[0],
            self.sensor.bandwidth[1],
            self.dI,
            dtype=np.float32
        )
        # Array of IO curves
        self.response = np.empty(
            (self.epochs, self.input.shape[-1]),
            dtype=np.float32
        )
        # Array of parameters and associated gradients
        self.pargrad = np.empty(
            (self.epochs, len(self.sensor.trainable_variables), 2),
            dtype=np.float32
        )
        self.slice_case = lambda x: {
            'grads': np.s_[:,1],
            'params': np.s_[:,0],
            'loss': np.s_[0],
            'factor': np.s_[1],
            'mse': np.s_[2],
            'fprint': np.s_[3],
            'lw_factor': np.s_[4],
            'lw_mse': np.s_[5],
            'lw_fprint': np.s_[6],
            'response': np.s_[:]
        }.get(x)
        pass

    """ Slicing """
    def _s_(self, kw: str) -> slice:
        return np.s_[self.epoch][self.slice_case(kw)]
    
    """ Data Retrieval """
    def _get_solution(self) -> tuple:
        return (self.tracked, self.params, )

    def plot(self) -> None:
        pass

