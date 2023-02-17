import numpy as np

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def relu(a):
    return np.maximum(0, a)

def identity(a):
    return a

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # overflow 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def init_network():
    network = {}
    network['w1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # 2x3
    network['b1'] = np.array([0.1, 0.2, 0.3]) # 1x3
    network['w2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) # 3x2
    network['b2'] = np.array([0.1, 0.2]) # 1x2
    network['w3'] = np.array([[0.1, 0.3], [0.2, 0.4]]) # 2x2
    network['b3'] = np.array([0.1, 0.2]) # 1x2
    return network

def forward(network, x): # ?x2
    a1 = np.dot(x, network['w1']) + network['b1'] # ?x2 dot 2x3 + 1x3 = ?x3
    z1 = sigmoid(a1) # ?x3
    a2 = np.dot(z1, network['w2']) + network['b2'] # ?x3 dot 3x2 + 1x2 = ?x2
    z2 = sigmoid(a2) # ?x2
    a3 = np.dot(z2, network['w3']) + network['b3'] # ?x2 dot 2x2 + 1x2 = ?x2
    y = softmax(a3)
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)