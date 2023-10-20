if __name__ != "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tqdm import tqdm

class Sensor(tf.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.optimizer = tf.optimizers.Adam()
        pass

    def summary(self):
        pass

    def _get_factor(
        self,
        domain: tf.Tensor,
        response: tf.Tensor
    ) -> tf.float32:
        return (response[-1]-response[0])/(domain[-1]-domain[0])

    def _get_mse(
        self, 
        domain: tf.Tensor, 
        response: tf.Tensor, 
        factor: tf.float32
    ) -> tf.float32:
        return tf.square(response-(factor*domain+response[0]))/(domain[-1]-domain[0])

    def _update_loss_weights(self, 
                             footprint,
                             nonlinearity,
                             avg_dO_dI):
        self.tracked["alpha"].append(self.optim_weights[0])
        self.tracked["beta"].append(self.optim_weights[1])
        self.tracked["gamma"].append(self.optim_weights[2])
        if footprint > self.inputs["specs"]["max_footprint"]:
            if self.footprint-footprint < 0:
                self.optim_weights[0] *= self.inputs["optim_config"]["secondary_lr"]
        if nonlinearity > self.inputs["specs"]["max_nonlinearity"]:
            if self.nonlinearity-nonlinearity < 0:
                self.optim_weights[1] *= self.inputs["optim_config"]["secondary_lr"]
        if avg_dO_dI < self.inputs["specs"]["min_sensitivity"]:
            if self.avg_dO_dI-avg_dO_dI > 0:
                self.optim_weights[2] *= self.inputs["optim_config"]["secondary_lr"]
        self.optim_weights = tf.nn.softmax(self.optim_weights).numpy()
        self.footprint = footprint
        self.nonlinearity = nonlinearity
        self.avg_dO_dI = avg_dO_dI
        pass

    def _bounds_enforcement(self) -> None:
        for i, bound in enumerate(self.param_bounds):
            self.trainable_variables[i].assign(tf.clip_by_value(self.trainable_variables[i],
                                                                bound[0],
                                                                bound[1]))
        pass

    def _get_loss(self) -> tuple:
        domain = tf.linspace
        response = self._get_output(domain, *self.trainable_variables)

        factor = self._get_factor(domain, response)
        mse = self._get_mse(domain, response, factor)
        footprint = self._get_footprint(*self.trainable_variables)
        
        loss = footprint+mse-factor
        return loss, factor, mse, footprint

    def _bounds_enforcement(self, 
                            gradients: tuple) -> tuple:
        new_grad = []
        for i, grad in enumerate(gradients):
            bound = self.param_bounds[i]
            param = self.trainable_variables[i]
            proj_val = (param+(grad*self.optimizer.learning_rate))
            new_grad.append(
                ((tf.minimum((
                    tf.relu(bound[1]-proj_val),
                    tf.relu(proj_val-bound[0])
                ))-param)/self.optimizer.learning_rate)
            )
        return tuple(new_grad)

    def _train_step(self):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            loss, footprint, nonlinearity, avg_dO_dI = self._get_loss()
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self._bounds_enforcement()
        self._update_loss_weights(footprint, nonlinearity, avg_dO_dI)
        return loss, footprint, nonlinearity, avg_dO_dI

    def fit(self):
        self.tracked["alpha"] = []
        self.tracked["beta"] = []
        self.tracked["gamma"] = []
        epochs = self.inputs["optim_config"]["epochs"]
        losses, footprints, nonlinearities, sensitivities =\
            tuple([np.empty((epochs), dtype=np.float32) for i in range(4)])
        params = np.empty((epochs, len(self.trainable_variables)))
        self.loss, self.footprint, self.nonlinearity, self.avg_dO_dI = self._get_loss()
        for epoch in tqdm(range(epochs), desc="Fitting... "):
            losses[epoch], footprints[epoch], nonlinearities[epoch], sensitivities[epoch] =\
                self._train_step()
            params[epoch, :] = self.trainable_variables
        return losses, params, footprints, nonlinearities, sensitivities