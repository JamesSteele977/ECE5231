import numpy as np
import tensorflow as tf
from tqdm import tqdm
from types import SimpleNamespace

from libs.sensor import Sensor

optim_case = lambda x: {
    'adam': tf.keras.optimizers.Adam,
    'sgd': tf.keras.optimizers.SGD
}.get(x)

class Optim(tf.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.reset = config
        self.optimizer = optim_case(config['optimizer'])(learning_rate=float(config['learning_rate']))
        self.epochs = int(config['epochs'])
        pass

    """ LOSS """
    def _get_mse(
        self, 
        domain: tf.Tensor, 
        response: tf.Tensor, 
        factor: tf.float32
    ) -> tf.float32:
        return tf.square(response-(factor*domain+response[0]))/(domain[-1]-domain[0])
    
    def _get_factor(
        self,
        domain: tf.Tensor,
        response: tf.Tensor
    ) -> tf.float32:
        return (response[-1]-response[0])/(domain[-1]-domain[0])
    
    def _get_loss():
        pass

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
        self.pargrad[self._grd_()] = tape.gradient(self._get_loss(), self.sensor.trainable_variables)
        self.optimizer.apply_gradients(zip(self.pargrad[self._grd_()], self.sensor.trainable_variables))
        self._rels_()
        self._clip_()
        self.pargrad[self._par_()] = self.sensor._get_tvars()
        pass

    def fit(self, sensor):
        self = self.__init__(self.reset)
        self.sensor = sensor
        Solve.__init__(self)
        for i in tqdm(range(self.epochs), desc="Fitting... "):
            self.epoch = i
            self._train_step()
        return self._get_solution()

class Solve():
    def __init__(self) -> None:
        # Array of tracked variables (loss, factor, mse, footprint, loss_weights(3))
        self.tracked = np.empty((self.epochs, 7), dtype=np.float32)
        # Array of parameters and associated gradients
        self.pargrad = np.empty(
            (self.epochs, len(self.trainable_variables), 2),
            dtype=np.float32
        )
        pass

    """ Slicing """
    def _s_(self, s: slice) -> slice:
        return np.s_[self.epoch][s]
    def _grd_(self):
        return self._s_(np.s_[:,1])
    def _par_(self):
        return self._s_(np.s_[:,0])
    def _lss_(self):
        return self._s_(np.s_[0])
    def _fac_(self):
        return self._s_(np.s_[1])
    def _mse_(self):
        return self._s_(np.s_[2])
    def _ftp_(self):
        return self._s_(np.s_[3])
    def _lwf_(self):
        return self._s_(np.s_[4])
    def _lwm_(self):
        return self._s_(np.s_[5])
    def _lwp_(self):
        return self._s_(np.s_[6])
    
    """ Data Retrieval """
    def _get_solution(self) -> tuple:
        return (self.tracked, self.params, )

    def plot(self) -> None:
        pass