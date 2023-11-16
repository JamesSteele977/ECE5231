import numpy as np
import tensorflow as tf
from sympy import sympify, lambdify, symbols, solve, Expr
from types import SimpleNamespace
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Dict, Callable

@dataclass
class ParameterRelationship():
    boolean_evaluation: Callable[[Tuple[tf.Variable, ...]], bool]
    substitution_solve: Callable[[Tuple[tf.Variable, ...]], float]
    sympy_expression: Expr

@dataclass
class SensorProfile():
    trainable_variables: Dict[str, Tuple[float, float]]
    bandwidth: Tuple[float, float]

    parameter_relationships: Tuple[ParameterRelationship, ...]

    _get_footprint: Callable[[Tuple[tf.Variable, ...]], tf.float32]
    _get_response: Callable[[Tuple[tf.Variable, ...], tf.Tensor], tf.Tensor]

class Sensor(tf.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.desc = {
            'Parameters': config['params'],
            'IO': config['IO'],
            'Footprint': config['footprint'],
            'Bandwidth': config['bandwidth']
        }

        self.input_sym: str = config['input_sym']
        self.bandwidth: tuple = tuple([float(i) for i in config['bandwidth']])

        self.relations: list = []
        self.symbols: list = []
        self.parameters: list = []

        for name, bound in config['params'].items():
            self._set_param(name, bound)
        
        for rel in config['relations']:
            self._set_relation(rel)

        self._set_IO(config['IO'])
        self._set_expr('_get_footprint', config['footprint'])
        pass

    def _get_param(self, name):
        for var in self.trainable_variables:
            if var.name.split(':')[0] == name:
                return deepcopy(var)
        pass

    def _set_param(self, name: str, bound: list) -> None:
        self.syms.append(symbols(name))
        self.parameters.append()
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

        self.parameter_relationships.append(ParameterRelationship(rel_bool, rel_path, sym_expr))
        pass

    def _lmds(self, name, expr):
        self.syms.append(symbols(name))
        sym_expr = sympify(expr)
        lmd_expr = lambdify(self.syms, sym_expr)
        lmd_args = list(sym_expr.free_symbols)
        return lmd_expr, lmd_args

    def _set_expr(self, name, expr):
        lmd_expr, lmd_args = self._lmds(name, expr)

        def expr_fn():
            input_args = [self._get_param(str(name)) for name in lmd_args]
            return lmd_expr(*input_args)

        setattr(self, name, deepcopy(expr_fn))
        pass

    def _set_IO(self, expr):
        name = '_get_IO'
        lmd_expr, lmd_args = self._lmds(name, expr)
        get_args = lambda x: self._get_param(x) if (
            (x.lower() != self.input_sym) and (self._get_param(x) is not None)
        ) else getattr(self, x)

        def IO_fn():
            input_args = [get_args(str(name)) for name in lmd_args]
            return lmd_expr(*input_args, getattr(self.optim, self.input_sym))

        setattr(self, name, IO_fn)

    def _get_tvars(self):
        return np.array([var.value.numpy() for var in self.trainable_variables])