import numpy as np
import pickle as pk

x = [1, 2, 3, 4]

print(np.asarray(x, dtype=np.float32))
print(np.asarray([[1, 2], [3, 4]], dtype=np.float32))
print(np.asarray(x, dtype=np.float32)[..., np.newaxis])
print(np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32))
print(np.asfortranarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32))

print(np.zeros(10))
print(np.zeros((10, 10)))
print(np.zeros((5, 5, 5)))
print(np.zeros((5, 10)))
print(np.zeros_like(x))

print(np.ones(10))
print(np.ones((10, 10)))
print(np.ones((5, 5, 5)))
print(np.ones_like(x))

print(np.eye(10))
print(np.identity(10))
print(np.eye(5, 10))
print(np.eye(10, 5))

print(np.full((5, 10), -1))
print(np.full_like([[1, 2], [3, 4]], 1.5, dtype=np.float32))
print(np.arange(5))
print(np.arange(5.0))
print(np.arange(2, 5))
print(np.arange(2, 10, 2))

print(np.linspace(0.0, 1.0, 21)) # include 1.0
print(np.linspace(0.0, 1.0, 21, endpoint=False)) # exclude 1.0

print(np.logspace(0.0, 1.0, 21))
print(np.logspace(0.0, 1.0, 20, endpoint=False))

print(np.random.rand(5))
print(np.random.rand(3, 4))

print(np.random.randn(5))
print(np.random.randn(3, 4))

print(np.random.randint(10))
print(np.random.randint(3, 10))

x = np.arange(10)
print(x)
print(x[1:7:2])
print(x[:7:2])
print(x[::2])
print(x[5:1:-1])
print(x[::-1])

x[:3] = 100
print(x)
x[::2] = [10, 20, 30, 40, 50]
print(x)

x = np.arange(25).reshape((5, -1))
print(x)
print(x[2])
print(x[2:5])
print(x[:, 2])
print(x[:,2:3])

x = np.arange(24).reshape((2, 3, 4))
print(x)
print(x[..., 2])
print(x[:, :, 2])
print(x[..., 2:3])
print(x[..., 1, 2])
print(x[:, 1, 2])
print(x[:, np.newaxis, :, :])

x = np.arange(10)
print(x)
x_ref = x[:4]
print(x_ref)
x_ref[2] = 100
print(x_ref)
print(x)

x = np.arange(10)
print(x)
x_copy = x[:4].copy()
print(x_copy)
x_copy[2] = 100
print(x_copy)
print(x)

x = np.array([1, 2, 3])
x_ref = np.asarray(x)
x_ref[2] = 100
print(x)

x = np.array([1, 2, 3])
x_copy = np.array(x)
x_copy[2] = 100
print(x)

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y)

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])
print(x * y)
print(np.matmul(x, y))

x = [1, 2, 3]
y = [4, 5, 6]
print([_ for _ in zip(x, y)])
print(list(map(lambda a: a[0] + a[1], zip(x, y))))

x = np.array([1, 2, 3])
print(x / 2)
print(x * 5)
print(10 + x)
print(x - 5)
print(list(map(lambda a: a + 5, x)))

x = np.random.randn(10)
y = np.random.randn(5)

with open('data/x.pk', 'wb') as f:
    pk.dump(x, f)

with open('data/x.pk', 'rb') as f:
    data = pk.load(f)
print(data)

with open('data/xy.pk', 'wb') as f:
    pk.dump({'x':x, 'y':y}, f)

with open('data/xy.pk', 'rb') as f:
    data = pk.load(f)
print(data)

x = np.random.randn(10)
y = np.random.randn(5)

np.save('x.npy', x)
data = np.load('x.npy')
print(data)

np.savez('xy.npz', x, y)
data = np.load('xy.npz')
print(data.files)
print(data['arr_0'])
print(data['arr_1'])

np.savez('xy.npz', x=x, y=y)
data = np.load('xy.npz')
print(data.files)
print(data['x'])
print(data['y'])

print(np.ones(20, dtype=np.int32))
print(np.arange(31))
print(np.arange(0.0, 32.0, 2.0))
print(np.array([[2 * i + 3 * j for j in range(10)] for i in range(10)]))
print(np.identity(10))
print(np.eye(10, 20))
print(np.concatenate([np.identity(10), np.zeros((10, 10))], axis=1))
print(np.random.randn(10, 10, 2) * 1.5 + 2.0)

x = np.arange(100).reshape(10, 10)
print(x)
print(x[0][0], x[-1][-1])
print(x[:5, :5])
print(x.flatten()[:25].reshape(5, 5))
print(x[::2, ::2])
print(np.repeat(x[..., np.newaxis], 5, axis=2))
print(x[..., np.newaxis] + np.zeros((10, 10, 5)))
print(x[np.array([0, -1])][:, np.array([0, -1])])
