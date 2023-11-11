import numpy as np
import tensorflow as tf
from sympy import sympify, lambdify, symbols, solve
from types import SimpleNamespace

class Sensor(tf.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.desc = {
            'Parameters': kwargs['params'],
            'IO': kwargs['IO'],
            'Footprint': kwargs['footprint']
        }

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

    def _get_tvars(self):
        return np.array([var.value.numpy() for var in self.trainable_variables])