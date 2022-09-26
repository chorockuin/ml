import tensorflow as tf
import matplotlib.pyplot as plt

class LinearRegression(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(1.0, trainable=True, dtype=tf.float32, name='weight')
        self.b = tf.Variable(0.0, trainable=True, dtype=tf.float32, name='bias')
    
    def __call__(self, x):
        return self.w * x + self.b

model = LinearRegression()

def loss_func(y, y_pred):
    return tf.losses.mean_squared_error(y, y_pred)
    # return tf.reduce_mean(tf.square(y - y_pred)) # mse

x = tf.random.normal([1000])
noise = tf.random.normal([1000])
w = 10.0
b = 3.0
y = w * x + b + noise

plt.scatter(x, y)
plt.plot(x, model(x), 'r-')
plt.show()

def train_step(model, x, y, learning_rate):
    with tf.GradientTape() as t:
        y_pred = model(x)
        loss = loss_func(y, y_pred)
    dw, db = t.gradient(loss, [model.w, model.b])
    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)
    return loss

w_hist = [model.w.numpy()]
b_hist = [model.b.numpy()]
for epoch in range(20):
    loss = train_step(model, x, y, 0.1)
    w_hist.append(model.w.numpy())
    b_hist.append(model.b.numpy())
    print(f'epoch: {epoch+1}, loss: {loss}')

plt.scatter(x, y)
plt.plot(x, model(x), 'r-')
plt.show()

print(model.w.numpy(), model.b.numpy())

plt.plot(w_hist, 'r-', label='weight')
plt.plot(b_hist, 'b-', label='bias')
plt.legend()
plt.plot([w] * len(w_hist), 'r--')
plt.plot([b] * len(b_hist), 'b--')
plt.show()
