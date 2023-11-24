import numpy as np
import tensorflow as tf
import sympy as sp
from dataclasses import dataclass, fields
from typing import Tuple, Dict, Callable, Union
from enum import Enum
import inspect

trainableVars: type = Tuple[tf.Variable, ...]
def EvalType(returns: type) -> type:
    return Callable[[trainableVars], returns]

@dataclass(frozen=True)
class SensorBasicInfo():
    trainable_variables: Dict[str, Tuple[float, float]]
    bandwidth: Tuple[float, float]
    input_symbol: str

@dataclass(frozen=True)
class SensorConfig(SensorBasicInfo):
    parameter_relationships: Tuple[str, ...]
    footprint: str
    response: str

@dataclass(frozen=True)
class ParameterRelationship():
    boolean_evaluation: EvalType(bool)
    conditional_loss_multiplier: EvalType(tf.float32)
    sympy_expression: sp.Expr

@dataclass(frozen=True)
class SensorProfile(SensorBasicInfo):
    parameter_relationships: Tuple[ParameterRelationship, ...]
    _get_footprint: EvalType(np.float32)
    _get_response: EvalType(np.float32)

class SensorDescription(Enum):
    TRAINABLE_VARIABLES: str = 'trainable_variables'
    BANDWIDTH: str = 'bandwidth'
    PARAMETER_RELATIONSHIPS: str = 'parameter_relationships'
    FOOTPRINT: str = 'footprint'
    INPUT_SYMBOL: str = 'input_symbol'
    RESPONSE: str = 'response'

class Sensor():
    def __init__(self, sensor_config: SensorConfig) -> None:
        self.shell_descritpion: Dict[SensorDescription, {f.type for f in fields(SensorConfig)}] = {
            SensorDescription.TRAINABLE_VARIABLES: sensor_config.trainable_variables,
            SensorDescription.BANDWIDTH: sensor_config.bandwidth,
            SensorDescription.PARAMETER_RELATIONSHIPS: sensor_config.parameter_relationships,
            SensorDescription.FOOTPRINT: sensor_config.footprint,
            SensorDescription.INPUT_SYMBOL: sensor_config.input_symbol,
            SensorDescription.RESPONSE: sensor_config.response
        }

        self.symbols: Tuple[sp.Symbol, ...] = sp.symbols(tuple(sensor_config.trainable_variables.keys()))
        self.parameter_relationships: list = []
        for relationship in sensor_config.parameter_relationships:
            self._set_parameter_relationship(relationship)
        self.parameter_relationships: Tuple[ParameterRelationship, ...] = tuple(self.parameter_relationships)
        
        self._get_footprint: EvalType(tf.float32) = self._get_footprint_function(sensor_config.footprint)
        self._get_response: EvalType(tf.float32) = self._get_response_funtion(sensor_config.response, sensor_config.input_symbol)

        self.sensor_profile: SensorProfile = SensorProfile(
            sensor_config.trainable_variables,
            sensor_config.bandwidth,
            sensor_config.input_symbol,

            self.parameter_relationships,
            self._get_footprint,
            self._get_response
        )

        pass

    def _lambdify_parse_expression(self, expression: str, argument_symbols: Tuple[sp.Symbol, ...]) -> Tuple[EvalType(tf.float32), Tuple[str, ...]]:
        sympy_expression: sp.Expr = sp.sympify(expression)
        lambda_function: EvalType(tf.float32) = sp.lambdify(argument_symbols, sympy_expression)
        return sympy_expression, lambda_function

    def _tf_index_to_name(self, tf_name: str) -> str:
        return ''.join(tf_name.split(':')[:-1])

    def _parse_trainable_variables(self, trainable_variables: trainableVars) -> dict:
        return {self._tf_index_to_name(variable.name):variable for variable in trainable_variables}
    
    def _get_symbolic_evaulation_function(self, lambda_function: EvalType(tf.float32)) -> EvalType(tf.float32) | EvalType(bool):

        def _evaluation(trainable_variables: trainableVars) -> Union[bool, tf.float32]:
            lambda_input: dict = self._parse_trainable_variables(trainable_variables)
            return lambda_function(**lambda_input)
        
        return _evaluation

    def _set_parameter_relationship(self, relationship_expression: str) -> None:
        sympy_expression, lambda_function = self._lambdify_parse_expression(relationship_expression, self.symbols)

        _boolean_evaluation: EvalType(bool) = self._get_symbolic_evaulation_function(lambda_function)

        difference_expression: sp.Expr = sympy_expression.lhs - sympy_expression.rhs

        def _conditional_loss_multiplier(trainable_variables: trainableVars) -> tf.float32:
            input_args: dict = self._parse_trainable_variables(trainable_variables)
            return tf.convert_to_tensor(float(difference_expression.subs(input_args))**2)

        self.parameter_relationships.append(ParameterRelationship(_boolean_evaluation, _conditional_loss_multiplier, sympy_expression))
        pass

    def _get_footprint_function(self, expression: str) -> EvalType(tf.float32):
        _, lambda_function = self._lambdify_parse_expression(expression, self.symbols)
        return self._get_symbolic_evaulation_function(lambda_function)

    def _get_response_funtion(self, expression: str, input_symbol: str) -> EvalType(tf.float32):
        _, lambda_function = self._lambdify_parse_expression(expression, self.symbols+(sp.symbols(input_symbol),))

        def _get_response(trainable_variables: trainableVars, sensor_input: tf.Tensor) -> tf.Tensor:
            lambda_input: dict = self._parse_trainable_variables(trainable_variables)
            lambda_input[self.sensor_profile.input_symbol] = sensor_input
            return lambda_function(**lambda_input)
        return _get_response
        