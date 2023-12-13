import numpy as np
import tensorflow as tf
import sympy as sp

from dataclasses import dataclass
from typing import Tuple, Dict, Callable, Union

from .config import SensorConfig

""" CUSTOM TYPES"""
tfReturn: type = Union[tf.float32, tf.Tensor]
trainableVars: type = Tuple[tf.Variable, ...]
spSymbols: type = Tuple[sp.Symbol, ...]
def EvalType(returns: type) -> type:
    return Callable[[trainableVars], returns]

""" DATACLASSES """
@dataclass(frozen=True)
class ParameterRelationship():
    boolean_evaluation: EvalType(bool)
    conditional_loss_multiplier: EvalType(tf.float32)
    sympy_expression: sp.Expr

@dataclass(frozen=True)
class SensorProfile():
    trainable_variables: Dict[str, Tuple[float, float]]
    bandwidth: Tuple[float, float]
    input_symbol: str
    parameter_relationships: Tuple[ParameterRelationship, ...]
    _get_footprint: EvalType(np.float32)
    _get_response: EvalType(np.float32)

class Sensor():
    """
    For converting string user input into functions/variables in sensor_profile, for
    implementation in Optim object

    Attributes
    ----------
    - shell_description (dict): Summary data for presentation in shell UI 
    - parameter_relationships (Tuple[ParameterRelationship, ...]): Tuple of
    ParameterRelationship objects, for use in constraint enforcement in Optim object
    - _get_footprint (EvalType(tf.float32)): Tensorflow-implemented footprint function
    - _get_response (EvalType(tfReturn)): Tensorflow-implemented response function
    - sensor_profile (SensorProfile): Configuration containing necessary data
    to be passed into Optim object
    
    Methods
    -------
    __init__():
        parameters:
            sensor_config (SensorConfig): Configuration object created in UI
        returns:
            self (Sensor)

    _tf_index_to_name(): Converts tensorflow trainable variable name format into
    standard format (removes ':index')
        parameters:
            tf_name (str): Unformatted variable name
        returns:
            name (str): Reformatted variable name
    
    _trainable_variables_to_dict(): Parses tuple of tensorflow trainable variables
    into dictionary for use by UI-defined tensorflow functions
        parameters:
            trainable_variables (trainableVars): Unformatted trainable variables
        returns:
            trainable_varialbes_dict (dict): Dictionary format of tvars
    
    _get_tensorflow_function(): Parses sympy Expr object into tensorflow operations,
    for gradient tracking in optimization loop
        parameters:
            expr (sp.Expr): Sympy expression form of function
        returns:
            tensorflow_function (EvalType(tfReturn)): Tensorflow form of function
    
    _set_parameter_relationship(): Parses string expression of parameter relationship
    into a ParameterRelationship object, consisting of a boolean evaluation of the
    relationship, a calculation of mean squared error from constraint satisfaction,
    and a sympy expression
        parameters:
            string_expression (str): String format of parameter relationship
        returns:
            [none]
    
    _get_footprint_function(): Parses string expression of footprint calculation
    into a tensorflow function
        parameters:
            string_expression (str): String format of footprint calculation
        returns:
            _get_footprint (EvalType(tf.float32)): Tensorflow function for footprint
            calculation
    
    _get_response_function(): Parses string expression of sensor input response 
    calculation into a tensorflow function
        parameters:
            string_expression (str): String format of response calculation
        returns:
            _get_response (EvalType(tf.float32)): Tensorflow function for response
            calculation

    """
    def __init__(self, sensor_config: SensorConfig) -> None:
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
                return tf.sin(evaluate_sympy_expr(expr.args[0], trainable_variables_dict))
            elif isinstance(expr, sp.cos):
                return tf.cos(evaluate_sympy_expr(expr.args[0], trainable_variables_dict))
            else:
                raise NotImplementedError(f"Operation {type(expr)} not implemented")

        def tensorflow_function(trainable_variables_dict: dict) -> tfReturn:
            return evaluate_sympy_expr(expr, trainable_variables_dict)

        return tensorflow_function
    
    """ Main Parsing Functions """
    def _set_parameter_relationship(self, string_expression: str) -> None:
        # Boolean
        boolean_sympy_expression: sp.Expr = sp.sympify(string_expression)
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
        tensorflow_function: EvalType(tf.float32) = self._get_tensorflow_function(sp.sympify(string_expression))

        def _get_footprint(trainable_variables: trainableVars) -> tf.float32:
            trainable_variables_dict: dict = self._trainable_variables_to_dict(trainable_variables)
            return tensorflow_function(trainable_variables_dict)
        
        return _get_footprint
    
    def _get_response_function(self, string_expression: str) -> EvalType(tfReturn):
        tensorflow_function: EvalType(tf.float32) = self._get_tensorflow_function(sp.sympify(string_expression))

        def _get_response(trainable_variables: trainableVars, sensor_input: tfReturn) -> tfReturn:
            trainable_variable_dict: dict = self._trainable_variables_to_dict(trainable_variables)
            trainable_variable_dict[self.sensor_profile.input_symbol] = sensor_input
            return tensorflow_function(trainable_variable_dict)
        
        return _get_response
        