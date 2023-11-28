import sympy as sp
import tensorflow as tf

def sympy_to_tensorflow(expr, symbols):
    """ Convert a sympy expression to a tensorflow function. """
    # Mapping from sympy to tensorflow functions
    replacements = {
        sp.sin: tf.sin,
        sp.cos: tf.cos,
        # Add other necessary mappings here
    }

    # Replace sympy functions with tensorflow functions in the expression
    for old, new in replacements.items():
        expr = expr.replace(old, lambda arg: new(arg))

    def tensorflow_eval(*args):
        """ Evaluate the expression using TensorFlow operations. """
        # Replace sympy symbols with tensorflow arguments
        subs = {sym: arg for sym, arg in zip(symbols, args)}
        return expr.subs(subs)

    return tensorflow_eval

# Example usage
if __name__ == '__main__':
    user_input = "sin(x) + cos(y)"
    x, y = sp.symbols('x y')
    expr = sp.sympify(user_input)

    # Convert to TensorFlow function
    tf_func = sympy_to_tensorflow(expr, (x, y))

    # TensorFlow variables or placeholders
    tf_x = tf.Variable(1.0, dtype=tf.float32)
    tf_y = tf.Variable(2.0, dtype=tf.float32)

    result = tf_func(tf_x, tf_y)

    # print(f"{tf_x}\n{tf_y}\n{result}")

    # Evaluate the TensorFlow function
    with tf.GradientTape() as tape:
        result = tf_func(tf_x, tf_y)

    print("Result:", result)
    # # Compute gradients if needed
    gradients = tape.gradient(result, [tf_x, tf_y])
    print("Gradients:", gradients)
