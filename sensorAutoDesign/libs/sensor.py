import numpy as np
import tensorflow as tf
from sympy import sympify, lambdify, symbols, solve, Expr
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Dict, Callable
from enum import Enum

def EvalType(returns: type) -> type:
    return Callable[[Tuple[tf.Variable, ...]], returns]

@dataclass
class SensorBasicInfo():
    trainable_variables: Dict[str, Tuple[float, float]]
    bandwidth: Tuple[float, float]
    input_symbol: str

@dataclass
class SensorConfig(SensorBasicInfo):
    parameter_relationships: Tuple[str, ...]
    footprint: str
    response: str

@dataclass
class ParameterRelationship():
    boolean_evaluation: EvalType(bool)
    substitution_solve: EvalType(float)
    sympy_expression: Expr

@dataclass
class SensorProfile(SensorBasicInfo):
    parameter_relationships: Tuple[ParameterRelationship, ...]
    _get_footprint: Callable[[Tuple[tf.Variable, ...]], tf.float32]
    _get_response: Callable[[Tuple[tf.Variable, ...], tf.Tensor], tf.Tensor]

class SensorDescription(Enum):
    TRAINABLE_VARIABLES: str = 'trainable_variables'
    BANDWIDTH: str = 'bandwidth'

    PARAMETER_RELATIONSHIPS: str = 'parameter_relationships'

    FOOTPRINT: str = 'footprint'
    INPUT_SYMBOL: str = 'input_symbol'
    RESPONSE: str = 'response'

class Sensor():
    def __init__(self, sensor_config: SensorConfig) -> None:
        self.shell_descritpion = {
            SensorDescription.TRAINABLE_VARIABLES: sensor_config.trainable_variables,
            SensorDescription.BANDWIDTH: sensor_config.bandwidth,

            SensorDescription.PARAMETER_RELATIONSHIPS: sensor_config.parameter_relationships,

            SensorDescription.FOOTPRINT: sensor_config.footprint,
            SensorDescription.INPUT_SYMBOL: sensor_config.input_symbol,
            SensorDescription.RESPONSE: sensor_config.response
        }

        self.symbols = symbols(list(sensor_config.trainable_variables.keys()))
        self.parameter_relationships = []
        for relationship in sensor_config.parameter_relationships:
            self._set_parameter_relationship(relationship)
        
        self._set_footprint(sensor_config.footprint)
        self._set_response(sensor_config.response)

        self.sensor_profile = SensorProfile(
            sensor_config.trainable_variables,
            sensor_config.bandwidth,
            sensor_config.input_symbol,

            
        )

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

    def _lambdify_parse_args(self, expression: str) -> Tuple[EvalType(float), Tuple[str, ...]]:
        sympy_expression: Expr = sympify(expression)
        lambda_function = lambdify(self.symbols, sympy_expression)
        lambda_arguments = tuple(sympy_expression.free_symbols)
        return lambda_function, lambda_arguments

    def _set_footprint(self, expression: str) -> None:
        lambda_funtion, arguments = self._lambdify_parse_args(expression)

        def expr_fn():
            input_args = [self._get_param(str(name)) for name in lmd_args]
            return lmd_expr(*input_args)

        setattr(self, name, deepcopy(expr_fn))
        pass

    def _set_response(self, expr):
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

    def _get_sensor_profile(self):
        return SensorProfile(self.trainable_variables)