import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sympy import sympify, lambdify, symbols, solve
import inspect
from types import SimpleNamespace
from copy import deepcopy
from libs.optim import Optim

class Sensor(tf.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.desc = {
            'Parameters': kwargs['params'],
            'IO': kwargs['IO'],
            'Footprint': kwargs['footprint']
        }

        self.optimizer = tf.optimizers.Adam()
        self.relations = []
        self.syms = []

        for name, bound in kwargs['params'].items():
            self._set_param(name, bound)

        for name, expr in kwargs['expressions'].items():
            self._set_expr(name, expr)
        
        for rel in kwargs['relations']:
            self._set_relation(rel)

        self._IO = lambdify(self.syms, sympify(kwargs['IO']))
        self._footprint = lambdify(self.syms, sympify(kwargs['footprint']))
        pass

    def _set_param(self, name: str, bound: list) -> None:
        self.syms.append(symbols(name))
        setattr(
            self, name,
            tf.Variable(
                initial_value=np.mean(bound),
                trainable=True,
                name=name
            )
        )
        getattr(self, name).bound = tuple(bound)
        pass

    def _get_param(self, name):
        for var in self.trainable_variables:
            if var.name == name:
                return deepcopy(var)
        pass

    def _set_relation(self, rel):
        sym_expr = sympify(rel)
        lmd_expr = lambdify(self.syms, sym_expr)
        lmd_args = list(sym_expr.free_symbols)

        def rel_bool(self):
            input_args = [self._get_param(name) for name in lmd_args]
            if None in input_args:
                missing_args = [name for idx, name in enumerate(lmd_args) if input_args[idx] is None]
                raise ValueError(f"Missing keyword arguments: {', '.join(missing_args)}")
            return lmd_expr(*input_args)
        
        def rel_path(self):
            input_args = {name:self._get_param(name) for name in lmd_args}
            if None in input_args:
                missing_args = [name for idx, name in enumerate(lmd_args) if input_args[idx] is None]
                if len(missing_args) > 1:
                    raise ValueError(f"Missing keyword arguments: {', '.join(missing_args)}")
                x = missing_args[0]
            solution = solve(sym_expr.subs(input_args), x)
            return solution

        self.relations.append(
            SimpleNamespace(
                eval=rel_bool,
                path=rel_path
            )
        )
        pass

    def _set_expr(self, name, expr):
        self.syms.append(symbols(name))
        sym_expr = sympify(expr)
        lmd_expr = lambdify(self.syms, sym_expr)
        lmd_args = list(sym_expr.free_symbols)

        def expr_fn(self):
            input_args = [self._get_param(name) for name in lmd_args]
            if None in input_args:
                missing_args = [name for idx, name in enumerate(lmd_args) if input_args[idx] is None]
                raise ValueError(f"Missing keyword arguments: {', '.join(missing_args)}")
            return lmd_expr(*input_args)

        setattr(self, name, expr_fn)
        pass

    def _relation_enforcement(self, relation: SimpleNamespace, gradient, vars):
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
            self.trainable_variables[i].assign(
                tf.clip_by_value(self.trainable_variables[i],
                bound[0],
                bound[1])
            )
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