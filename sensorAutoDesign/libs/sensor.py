import numpy as np
import tensorflow as tf
from sympy import sympify, lambdify, symbols, solve, Expr, Symbol
from dataclasses import dataclass, fields
from typing import Tuple, Dict, Callable
from enum import Enum

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
    substitution_solve: EvalType(float)
    sympy_expression: Expr

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

        self.symbols: Tuple[Symbol, ...] = symbols(list(sensor_config.trainable_variables.keys())+list(sensor_config.input_symbol))
        self.parameter_relationships: list = []
        for relationship in sensor_config.parameter_relationships:
            self._set_parameter_relationship(relationship)
        self.parameter_relationships: Tuple[ParameterRelationship, ...] = tuple(self.parameter_relationships)
        
        self._get_footprint: EvalType(float) = self._get_expression_function(sensor_config.footprint)
        self._get_response: EvalType(float) = self._get_response_funtion(sensor_config.response)

        self.sensor_profile: SensorProfile = SensorProfile(
            sensor_config.trainable_variables,
            sensor_config.bandwidth,
            sensor_config.input_symbol,

            self.parameter_relationships,
            self._get_footprint,
            self._get_response
        )

        pass

    def _lambdify_parse_expression(self, expression: str) -> Tuple[EvalType(float), Tuple[str, ...]]:
        sympy_expression: Expr = sympify(expression)
        lambda_function: EvalType(float) = lambdify(self.symbols, sympy_expression)
        lambda_arguments: Tuple[str, ...] = tuple(str(sym) for sym in sympy_expression.free_symbols)
        return sympy_expression, lambda_function, lambda_arguments

    def _tf_index_to_name(self, tf_name: str) -> str:
        return ''.join(tf_name.split(':')[:-1])

    def _parse_trainable_variables(self, trainable_variables: trainableVars, arguments: Tuple[str, ...]) -> list:
        lambda_input: list = [variable for variable in trainable_variables if self._tf_index_to_name(variable.name) in arguments]
        lambda_input.sort(key=lambda variable: arguments.index(self._tf_index_to_name(variable.name)))
        return lambda_input
    
    def _get_symbolic_evaulation_function(self, lambda_function: EvalType(float), arguments: Tuple[str, ...]) -> EvalType(float) | EvalType(bool):

        def _evaluation(trainable_variables: trainableVars) -> bool:
            lambda_input: list = self._parse_trainable_variables(trainable_variables, arguments)
            return lambda_function(*lambda_input)
        return _evaluation

    def _set_parameter_relationship(self, relationship_expression: str) -> None:
        sympy_expression, lambda_function, arguments = self._lambdify_parse_expression(relationship_expression)

        _boolean_evaluation: EvalType(bool) = self._get_symbolic_evaulation_function(lambda_function, arguments)
        
        def _substitution_solve(substituted_variables: trainableVars) -> float:
            input_args: dict = {variable.name:variable for variable in substituted_variables}

            missing_args: list = [name for idx, name in enumerate(arguments) if input_args[idx] is None]
            degrees_of_freedom: int = len(missing_args)

            if degrees_of_freedom != 1:
                if degrees_of_freedom > 1:
                    raise ValueError(f"Missing keyword arguments: {', '.join(missing_args)}")
                if degrees_of_freedom < 1:
                    raise ValueError(f"All variables provided. Cannot perform substitution solve (at least one variable must be missing)")
            
            return solve(sympy_expression.subs(input_args), missing_args[0])

        self.parameter_relationships.append(ParameterRelationship(_boolean_evaluation, _substitution_solve, sympy_expression))
        pass

    def _get_expression_function(self, expression: str) -> EvalType(float):
        _, lambda_function, arguments = self._lambdify_parse_expression(expression)
        return self._get_symbolic_evaulation_function(lambda_function, arguments)

    def _get_response_funtion(self, expression: str) -> EvalType(float):
        _, lambda_function, arguments = self._lambdify_parse_expression(expression)

        def _get_response(trainable_variables: trainableVars, sensor_input: tf.Tensor) -> tf.Tensor:
            lambda_input: list = self._parse_trainable_variables(trainable_variables, arguments)
            lambda_input.insert(arguments.index(self.sensor_profile.input_symbol), sensor_input)

            if (None in lambda_input) or (len(lambda_input) == 0):
                raise ValueError(f"Missing variable args in trainable_variables")
            
            return lambda_function(*lambda_input)
        return _get_response
        