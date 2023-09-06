import tensorflow as tf
import numpy as np

w_ = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)
h_ = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)
l_ = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)

I_min = tf.constant(0.0)
I_max = tf.constant(100.0)
E = tf.constant(0.1)
N = 100

alpha = tf.constant(1.0)
beta = tf.constant(1.0)
gamma = tf.constant(1.0)

optimizer = tf.optimizers.Adam(learning_rate=0.01)

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        tape.watch([w_, h_, l_])
        
        # w_ = tf.maximum(w, 1.0)
        # h_ = tf.maximum(h, 1.0)
        # l_ = tf.maximum(l, 1.0)
        
        I_values = tf.linspace(I_min, I_max, N)

        O_values = (w_ * h_**l_) * I_values
        avg_dO_dI = tf.reduce_mean(tf.gradients(O_values, I_values))
        
        footprint = w_ * l_
        
        optimal_O = avg_dO_dI * I_values
        error = tf.math.abs(O_values - optimal_O)
        linearity = tf.reduce_sum(error)
        
        loss = -alpha * avg_dO_dI + beta * footprint - gamma * linearity
        # tf.print("LOSS ", loss,
        #          "\nLINEARITY", linearity,
        #          "\nERROR", tf.reduce_max(error),
        #          "\nSENSITIVITY", avg_dO_dI,
        #          "\nFOOTPRINT", footprint,
        #          "\n")

        loss = tf.reduce_mean(tf.where(error > E, tf.float32.max, loss))
        
    grads = tape.gradient(loss, [w_, h_, l_])
    optimizer.apply_gradients(zip(grads, [w_, h_, l_]))
    return loss, w_, h_, l_

epochs = 10000

# Training Loop
losses = np.empty((epochs))
ws = np.empty((epochs))
hs = np.empty((epochs))
ls = np.empty((epochs))
for epoch in range(epochs):
    loss, w, h, l = train_step()
    # print(f"Epoch {epoch}, Loss: {loss.numpy()}, Design: {w.numpy(), h.numpy(), l.numpy()}")
    losses[epoch] = loss
    ws[epoch] = w
    hs[epoch] = h
    ls[epoch] = l

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2)
plt.yscale('log')
ax[0].plot(losses)
plt.yscale('linear')
ax[0].legend(["loss"])
ax[1].plot(ws)
ax[1].plot(hs)
ax[1].plot(ls)
ax[1].legend(["width", "height", "length"])
plt.show()