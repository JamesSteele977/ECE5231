import numpy as np
import tensorflow as tf
import sympy as sp
from dataclasses import dataclass, fields
from typing import Tuple, Dict, Callable, Union
from enum import Enum
import inspect

tfReturn: type = Union[tf.float32, tf.Tensor]
trainableVars: type = Tuple[tf.Variable, ...]
spSymbols: type = Tuple[sp.Symbol, ...]
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

        self.symbols: spSymbols = sp.symbols(tuple(sensor_config.trainable_variables.keys()))
        self.parameter_relationships: list = []
        for relationship in sensor_config.parameter_relationships:
            self._set_parameter_relationship(relationship)
        self.parameter_relationships: Tuple[ParameterRelationship, ...] = tuple(self.parameter_relationships)
        
        self._get_footprint: EvalType(tf.float32) = self._get_footprint_function(sensor_config.footprint)
        self._get_response: EvalType(tfReturn) = self._get_response_function(sensor_config.response)

        self.sensor_profile: SensorProfile = SensorProfile(
            sensor_config.trainable_variables,
            sensor_config.bandwidth,
            sensor_config.input_symbol,

            self.parameter_relationships,
            self._get_footprint,
            self._get_response
        )

        pass

    def _tf_index_to_name(self, tf_name: str) -> str:
        return ''.join(tf_name.split(':')[:-1])

    def _trainable_variables_to_dict(self, trainable_variables: trainableVars) -> dict:
        return {self._tf_index_to_name(variable.name):variable for variable in trainable_variables}
    
    def _get_tensorflow_function(self, expr: sp.Expr) -> EvalType(tfReturn):
        """ Convert a sympy expression to a tensorflow function. """
        def evaluate_sympy_expr(expr: sp.Expr, trainable_variables_dict: dict) -> tfReturn:
            """ Recursively evaluate a sympy expression with TensorFlow operations. """
            if isinstance(expr, sp.Symbol):
                # Substitute sympy symbol with tensorflow variable
                return trainable_variables_dict[str(expr)]
            elif isinstance(expr, sp.Number):
                # Convert sympy number to tensorflow constant
                return tf.constant(float(expr))
            elif isinstance(expr, sp.Add):
                # Recursively evaluate add operation
                return sum(evaluate_sympy_expr(arg, trainable_variables_dict) for arg in expr.args)
            elif isinstance(expr, sp.Mul):
                # Recursively evaluate multiply operation
                result = evaluate_sympy_expr(expr.args[0], trainable_variables_dict)
                for arg in expr.args[1:]:
                    result *= evaluate_sympy_expr(arg, trainable_variables_dict)
                return result
            elif isinstance(expr, sp.Pow):
                return tf.pow(
                        evaluate_sympy_expr(expr.as_base_exp()[0], trainable_variables_dict),
                        evaluate_sympy_expr(expr.as_base_exp()[1], trainable_variables_dict)
                    )
            elif isinstance(expr, sp.exp):
                return tf.exp(evaluate_sympy_expr(expr.args[0], trainable_variables_dict))
            elif isinstance(expr, sp.sin):
                # Evaluate sin operation
                return tf.sin(evaluate_sympy_expr(expr.args[0], trainable_variables_dict))
            elif isinstance(expr, sp.cos):
                # Evaluate cos operation
                return tf.cos(evaluate_sympy_expr(expr.args[0], trainable_variables_dict))
            # Add handling for more sympy operations as needed
            else:
                raise NotImplementedError(f"Operation {type(expr)} not implemented")

        def tensorflow_function(trainable_variables_dict: dict) -> tfReturn:
            """ Evaluate the expression using TensorFlow operations. """
            return evaluate_sympy_expr(expr, trainable_variables_dict)

        return tensorflow_function
    
    def _parse_str_to_expr_function(self, string_expression: str) -> Tuple[EvalType(tf.float32), Tuple[str, ...]]:
        sympy_expression: sp.Expr = sp.sympify(string_expression)
        tensorflow_function: EvalType(tfReturn) = self._get_tensorflow_function(sympy_expression)
        return sympy_expression, tensorflow_function

    """ Main Parsing Functions """
    def _set_parameter_relationship(self, string_expression: str) -> None:
        # Boolean
        boolean_sympy_expression, _ = self._parse_str_to_expr_function(string_expression)
        def _boolean_evaluation(trainable_variables: trainableVars) -> bool:
            trainable_variables_dict: dict = self._trainable_variables_to_dict(trainable_variables)
            return boolean_sympy_expression.subs(trainable_variables_dict)

        # Loss scalar
        difference_sympy_expression: sp.Expr = sp.Pow(boolean_sympy_expression.lhs - boolean_sympy_expression.rhs, 2, evaluate=False)
        tensorflow_difference_function: EvalType(tf.float32) = self._get_tensorflow_function(difference_sympy_expression)
        def _conditional_loss_multiplier(trainable_variables: trainableVars) -> tf.float32:
            trainable_variables_dict: dict = self._trainable_variables_to_dict(trainable_variables)
            return tensorflow_difference_function(trainable_variables_dict)

        self.parameter_relationships.append(ParameterRelationship(_boolean_evaluation, _conditional_loss_multiplier, boolean_sympy_expression))
        pass

    def _get_footprint_function(self, string_expression: str) -> EvalType(tf.float32):
        _, tensorflow_function = self._parse_str_to_expr_function(string_expression)

        def _get_footprint(trainable_variables: trainableVars) -> tf.float32:
            trainable_variables_dict: dict = self._trainable_variables_to_dict(trainable_variables)
            return tensorflow_function(trainable_variables_dict)
        
        return _get_footprint
    
    def _get_response_function(self, expression: str) -> EvalType(tfReturn):
        _, tensorflow_function = self._parse_str_to_expr_function(expression)

        def _get_response(trainable_variables: trainableVars, sensor_input: tfReturn) -> tfReturn:
            trainable_variable_dict: dict = self._trainable_variables_to_dict(trainable_variables)
            trainable_variable_dict[self.sensor_profile.input_symbol] = sensor_input
            return tensorflow_function(trainable_variable_dict)
        
        return _get_response
        