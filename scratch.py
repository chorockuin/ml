import numpy as np
# import tensorflow as tf

# x: input
# w: weight
# b: bias
# h: hidden
# y: output
# t: target

# activation functions
def sigmoid(h):
    y = 1 / (1 + np.exp(-h))
    return y

def relu(h):
    y = np.maximum(0, h)
    return y

def softmax(h):
    if h.ndim == 2:
        h = h.T
        h = h - np.max(h, axis=0) # np.exp(x)가 너무 작으면 overflow가 발생함. np.exp(z-c)를 해도 결과는 같다는 것이 증명되었으므로, z-c를 써서 overflow를 방지함
        y = np.exp(h) / np.sum(np.exp(h), axis=0) # 가로 방향으로 훑으면서 세로 값들의 합을 구함
        return y.T
    h = h - np.max(h) 
    y = np.exp(h) / np.sum(np.exp(h))
    return y

# loss functions
def cross_entropy_error(y, t): # 1차원, one-hot encoding된 데이터
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size    

def CRE_batch(y, t): # 다차원, one-hot encoding된 데이터
    delta = 1e-7
    if y.ndim == 1:
        y = np.reshape(1, y.size) # 1차원 데이터일 경우, 아래 batch_size를 구하기 위해(.shape[0]을 사용하기 위해) reshape
        t = np.reshape(1, t.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size # 1개의 데이터당 평균 CRE구함

def CRE_batch_onehot_index(y, t): # 다차원, t 값 데이터가 one-hot encoding의 index
    delta = 1e-7
    if y.ndim == 1:
        y = np.reshape(1, y.size)
        t = np.reshape(1, t.size)
    batch_size = y.shape[0]
    # 정답의 index에 해당하는 신경망의 출력 값만 log 연산하면 된다는 논리
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size

# gradient
def _numerical_gradient_1d(loss_f, w): # 가중치에 따라 loss가 어떻게 변하는지 loss를 가중치에 따라 편미분
    min_c = 1e-4 # 0.0001
    grad = np.zeros_like(w)
    
    # loss_f 내부에는 이미 사용할 x값과 t값이 정해져있다. w만 변경(w+h, w-h)해서 넣으면 됨
    for i in range(w.size):
        tmp_val = w[i]
        w[i] = float(tmp_val) + min_c
        loss1 = loss_f() # loss_f(w + min_c) = 가중치에 min_c만큼 더한다음 x, t에 대해 loss값을 구함
        
        w[i] = tmp_val - min_c 
        loss2 = loss_f() # loss_f(x - min_c) = 가중치에서 min_c만큼 뺀다음 x, t에 대해 loss값을 구함
        grad[i] = (loss1 - loss2) / (min_c * 2) # 가중치 변화(min_c * 2)에 따라 loss 값이 얼만큼 변하는지 기울기 구함
        
        w[i] = tmp_val # 값 복원
    return grad

def numerical_gradient_2d(loss_f, W):
    if W.ndim == 1:
        return _numerical_gradient_1d(loss_f, W)
    else:
        grad = np.zeros_like(W)
        
        for idx, w in enumerate(W):
            grad[idx] = _numerical_gradient_1d(loss_f, w)
        return grad

def gradient_descent(loss_f, init_w, lr=0.01, step_num=100):
    w = init_w
    for i in range(step_num):
        grad = numerical_gradient_2d(loss_f, w)
        w -= lr * grad
    return w

# test_cross_entropy_error()
def test_cross_entropy_error():
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) 
    # 인덱스 2가 정답인데 신경망 출력도 인덱스 2의 값이 0.6으로 가장 높다. 정답에 가까우므로 CRE값은 0에 가깝겠지?
    y1 = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]) 
    # 인덱스 2가 정답인데 신경망 출력은 인덱스 3의 값, 0.6으로 가장 높다. 정답에 가깝지 않으니 CRE값은 0에서 멀겠지?
    y2 = np.array([0.1, 0.05, 0.0, 0.6, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]) 
    print(cross_entropy_error(y1, t)) # 0.5 정도 나온다
    print(cross_entropy_error(y2, t)) # 16 정도 나온다

    t = np.array([[0, 1, 0, 0], [1, 0, 0, 0]])
    y = np.array([[0.0, 0.6, 0.2, 0.2], [0.9, 0.0, 0.0, 0.1]])
    # CRE 값은 np.sum()으로 구해지므로, 2차원 이상의 t, y를 넣어도 결과 값은 행렬의 모든 원소의 CRE값을 다 더한 scalar 값임
    print(cross_entropy_error(y, t)) # CRE의 총합
    print(CRE_batch(y, t)) # 데이터 1개의 평균 CRE

    t = np.array([1, 0])
    y = np.array([[0.0, 0.6, 0.2, 0.2], [0.9, 0.0, 0.0, 0.1]])
    print(CRE_batch_onehot_index(y, t)) # 데이터 1개의 평균 CRE

# test_forward()
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
    h1 = np.dot(x, network['w1']) + network['b1'] # ?x2 dot 2x3 + 1x3 = ?x3
    h2 = sigmoid(h1) # ?x3
    h3 = np.dot(h2, network['w2']) + network['b2'] # ?x3 dot 3x2 + 1x2 = ?x2
    h4 = sigmoid(h3) # ?x2
    h5 = np.dot(h4, network['w3']) + network['b3'] # ?x2 dot 2x2 + 1x2 = ?x2
    y = softmax(h5)
    return y

def test_forward():
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)
    
# test_two_layer_net()
class TwoLayerNet:
    def __init__(self, x_size, h_size, y_size, w_init_std=0.01):
        self.params = {}
        self.params['w1'] = w_init_std * np.random.randn(x_size, h_size)
        self.params['b1'] = w_init_std * np.zeros(h_size)
        self.params['w2'] = w_init_std * np.random.randn(h_size, y_size)
        self.params['b2'] = w_init_std * np.zeros(y_size)
        
    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        h1 = np.dot(x, w1) + b1
        h2 = sigmoid(h1)
        h3 = np.dot(h2, w2) + b2
        y = softmax(h3)
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_w = lambda: self.loss(x, t)
        
        grads = {}
        grads['w1'] = numerical_gradient_2d(loss_w, self.params['w1'])
        grads['b1'] = numerical_gradient_2d(loss_w, self.params['b1'])
        grads['w2'] = numerical_gradient_2d(loss_w, self.params['w2'])
        grads['b2'] = numerical_gradient_2d(loss_w, self.params['b2'])
        return grads
        
def test_two_layer_net():
    net = TwoLayerNet(x_size=784, h_size=100, y_size=10)
    x = np.random.rand(100, 784)
    t = np.random.rand(100, 10)
    y = net.predict(x)
    grads = net.numerical_gradient(x, t)
        
# test_mini_batch()
import load_mnist
import matplotlib.pyplot as plt

def test_mini_batch():
    (x_train, t_train), (x_test, t_test) = load_mnist.load_mnist(normalize=True, one_hot_label=True)
    # x_train = (60000, 784)
    # t_train = (60000, 10)
    # x_test = (10000, 784)
    # t_train = (10000, 10)

    net = TwoLayerNet(x_size=784, h_size=50, y_size=10)

    iters_num = 10000
    train_size = x_train.shape[0] # 60000
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1) # 1 epoch = 모든 데이터를 1회 학습

    for i in range(iters_num):
        batch_indexes = np.random.choice(train_size, batch_size) # 100 indexes
        x_batch = x_train[batch_indexes] # 100
        t_batch = t_train[batch_indexes] # 100
        
        grad = net.numerical_gradient(x_batch, t_batch) # 손실함수에 대해 가중치 편미분해서 기울기 얻음
        
        for k in ('w1', 'b1', 'w2', 'b2'):
            net.params[k] -= learning_rate * grad[k] # 기울기(파라미터 편미분 값들 모아놓은 matrix) * 학습률 만큼 파라미터 업데이트

        loss = net.loss(x_batch, t_batch) # loss 구함
        train_loss_list.append(loss)
        print(f'train_loss:{loss}')

        if i % iter_per_epoch == 0:
            train_acc = net.accuracy(x_train, t_train)
            test_acc = net.accuracy(x_test, t_test)

            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            print(f'batch_size:{batch_size}, iter:{i}, train acc:{train_acc} test acc:{test_acc}')

    # 그래프 그리기
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()

# test_backward_apple()
class MultiplyLayer:
    def __init__(self):
        self.x1 = None
        self.x2 = None
        
    def forward(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        h = x1 * x2
        return h
    
    def backward(self, dh):
        dx1 = dh * self.x2
        dx2 = dh * self.x1
        return dx1, dx2
    
class AddLayer:
    def __init__(self):
        self.x1 = None
        self.x2 = None
        
    def forward(self, x1, x2):
        h = x1 + x2
        return h
    
    def backward(self, dh):
        dx1 = dh * 1
        dx2 = dh * 1
        return dx1, dx2
    
def test_backward_apple():
    apple = {'price': 100, 'quantity': 2, 'tax': 1.1}

    layer1 = MultiplyLayer()
    layer2 = MultiplyLayer()

    h1 = layer1.forward(apple['price'], apple['quantity'])
    print(h1)

    # price가 변할 때 h1은 얼마나 변하나? quantity가 변할 때 h1은 얼마나 변하나?
    # 왜 dh값을 1로 줄까? 얼마나 변하나?에 대한 기준 값이므로 1로 설정하면 1에 대한 비율을 알 수 있기 때문
    d_price, d_quantity = layer1.backward(1)
    # price가 변할 때 h1은 2씩 변하고, quantity가 변할 때 h1은 100씩 변한다
    print(d_price, d_quantity)

    h2 = layer2.forward(h1, apple['tax'])
    print(h2)

    # price * quantity가 변할 때 h2는 얼마나 변하나? tax가 변할 때 h2는 얼마나 변하나?
    # 왜 h2값을 1로 줄까? 얼마나 변하나?에 대한 기준 값이므로 1로 설정하면 1에 대한 비율을 알 수 있기 때문
    d_price_d_quantity, d_tax = layer2.backward(1)
    # price * quantity가 변할 때 y2는 1.1씩 변하고, tax가 변할 때 y2는 200씩 변한다
    print(d_price_d_quantity, d_tax)

    # 그렇다면 최종적으로 price가 변할 때 h2는 얼마나 변하나? quantity가 변할 때 h2는 얼마나 변하나?
    d_price, d_quantity = layer1.backward(d_price_d_quantity)
    # price가 변할 때 h2는 2.2씩 변하고 quantity가 변할 때 h2는 110씩 변한다
    print(d_price, d_quantity)

# test_backward_activation()
class ReLU:
    def __init__(self):
        self.zero_mask = None
    
    def forward(self, h):
        # y가 np.array일 경우에만 가능한 연산
        # 0으로 mask되어야할 index를 true로 설정해 줌
        self.zero_mask = (h <= 0)
        h[self.zero_mask] = 0
        return h
    
    def backward(self, dh): # 여기서 y는 네트워크의 뒤 부터 쭉 이어져온 값. 따라서 그냥 곱해준다고 생각하면 됨
        dh[self.zero_mask] = 0
        return dh
    
class Sigmoid:
    def __init__(self):
        self.y = None
    
    def forward(self, h):
        y = 1 / (1 + np.exp(-h))
        self.y = y
        return y
    
    def backward(self, dh): 
        dh = dh * (1.0 - self.y) * self.y # 여기서 y는 네트워크의 뒤 부터 쭉 이어져온 값. 따라서 그냥 곱해준다고 생각하면 됨
        return dh

def test_backward_activation():
    h = np.array([-3.2, 0.7, 0, -1.0, 4.3, 2.9])
    print(f'tensor: {h}')
    
    relu = ReLU()
    y = relu.forward(h)
    print(f'relu-foward: {y}')
    dy = relu.backward(y)
    print(f'relu-backward: {dy}')
    
    sigmoid = Sigmoid()
    y = sigmoid.forward(h)
    print(f'sigmoid-forward: {y}')
    dy = sigmoid.backward(y)
    print(f'sigmoid-backward: {dy}')
    
    # y = tf.math.sigmoid(h)
    # print(f'tf sigmoid-forward: {y}')

# test_backward_affine()
class AffineLayer:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        h = np.dot(x, self.w) + self.b
        return h
    
    def backward(self, dh):
        # w와 b에 대해서만 미분하면 되는데 왜 굳이 h에 대해 미분했을까?
        next_dh = np.dot(dh, self.w.T)
        # w와 b에 대해서 미분하려면 dh가 필요하기 때문이다. h에 대한 미분값이 다음 layer의 여기 dh에 들어가기 때문이다
        self.dw = np.dot(self.x.T, dh)
        print('dw:', self.dw.shape)
        print(self.dw)
        
        self.db = np.sum(dh, axis=0) # axis=0 은 column
        print('db:', self.db.shape)
        print(self.db)
        
        return next_dh
    
def test_backward_affine():
    # x = np.array([[2.2, 3.3], [-3.1, -2.2], [4.4, 9.1], [-1.1, 0.4]])
    # w = np.array([[3.2, 1.1, -0.4], [-4.3, 0.9, 1.1]])
    # b = np.array([1.0, -0.4, 0.4])
    x = np.array([[1,2],[3,4],[5,6],[7,8]])
    w = np.array([[0,0,0],[1,1,1]])
    b = np.array([2,2,2])
    print('x:', x.shape)
    print(x)
    print('w:', w.shape)
    print(w)
    print('b:', b.shape)
    print(b)
    
    affine = AffineLayer(w, b)
    h = affine.forward(x)
    print('h:', h.shape)
    print(h)
    
    dh = affine.backward(h)
    print('dh:', dh.shape)
    print(dh)

# test_backward()
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, y, t):
        self.t = t
        self.y = softmax(y)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dy=1):
        batch_size = self.t.shape[0]
        dh = (self.y - self.t) / batch_size
        return dh

from collections import OrderedDict

class LayerNet:
    def __init__(self, x_size, h_size, y_size, w_init_std=0.01):
        self.params = {}
        self.params['w1'] = w_init_std * np.random.randn(x_size, h_size)
        self.params['b1'] = np.zeros(h_size)
        self.params['w2'] = w_init_std * np.random.randn(h_size, y_size)
        self.params['b2'] = np.zeros(y_size)
        
        self.layers = OrderedDict()
        self.layers['affine1'] = AffineLayer(self.params['w1'], self.params['b1'])
        self.layers['relu1'] = ReLU()
        self.layers['affine2'] = AffineLayer(self.params['w2'], self.params['b2'])
        self.last_layer = SoftmaxWithLoss()
    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_w = lambda: self.loss(x, t)
        
        grads = {}
        grads['w1'] = numerical_gradient_2d(loss_w, self.params['w1'])
        grads['b1'] = numerical_gradient_2d(loss_w, self.params['b1'])
        grads['w2'] = numerical_gradient_2d(loss_w, self.params['w2'])
        grads['b2'] = numerical_gradient_2d(loss_w, self.params['b2'])
        return grads
    
    def gradient(self, x, t):
        self.loss(x, t)
        dh = self.last_layer.backward(1)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dh = layer.backward(dh)
        grads = {}
        grads['w1'] = self.layers['affine1'].dw
        grads['b1'] = self.layers['affine1'].db
        grads['w2'] = self.layers['affine2'].dw
        grads['b2'] = self.layers['affine2'].db
        return grads
    
def test_backward():
    (x_train, t_train), (x_test, t_test) = load_mnist.load_mnist(normalize=True, one_hot_label=True)
    net = LayerNet(x_size=784, h_size=50, y_size=10)
    
    x_batch = x_train[:3]
    t_batch = t_train[:3]
    
    grad_numerical = net.numerical_gradient(x_batch, t_batch)
    grad_backpropagation= net.gradient(x_batch, t_batch)
    
    for k in grad_numerical.keys():
        diff = np.average(np.abs(grad_backpropagation[k] - grad_numerical[k]))
        print(f'{k}: {diff}')

# test_layer_net()
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for k in params.keys():
            params[k] -= self.lr * grads[k] # 기울기 * 학습률 만큼 파라미터 업데이트
            
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for k, v in params.items():
                self.v[k] = np.zeros_like(v) # 파라미터 모양 그대로 0으로 초기화시킨 모멘텀 매트릭스 만들고
        for k in params.keys():
            # 모멘텀 매트릭스에 계속 업데이트할 가중치를 더한다
            # 만약 같은 방향의 기울기(grads)가 계속 구해졌다면, 파라미터에 업데이트할 값(self.v)은 점점 더 커질 것이다
            self.v[k] = self.momentum * self.v[k] - self.lr * grads[k]
            params[k] += self.v[k]
            
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for k, v in params.items():
                self.h[k] = np.zeros_like(v)
        for k in params.keys():
            self.h[k] += grads[k] * grads[k]
            params[k] -= self.lr * grads[k] / (np.sqrt(self.h[k]) + 1e-7)

def test_layer_net():
    (x_train, t_train), (x_test, t_test) = load_mnist.load_mnist(normalize=True, one_hot_label=True)
    # x_train = (60000, 784)
    # t_train = (60000, 10)
    # x_test = (10000, 784)
    # t_train = (10000, 10)

    net = LayerNet(x_size=784, h_size=50, y_size=10)
    # optimizer = SGD()
    # optimizer = Momentum()
    optimizer = AdaGrad()

    iters_num = 10000
    train_size = x_train.shape[0] # 60000
    batch_size = 100

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1) # 1 epoch = 모든 데이터를 1회 학습

    for i in range(iters_num):
        batch_indexes = np.random.choice(train_size, batch_size) # 100 indexes
        x_batch = x_train[batch_indexes] # 100
        t_batch = t_train[batch_indexes] # 100
        
        grads = net.gradient(x_batch, t_batch) # 손실함수에 대해 가중치 편미분해서 기울기 얻음
        params = net.params        
        optimizer.update(params, grads)
        
        loss = net.loss(x_batch, t_batch) # loss 구함
        train_loss_list.append(loss)
        print(f'train_loss:{loss}')

        if i % iter_per_epoch == 0:
            train_acc = net.accuracy(x_train, t_train)
            test_acc = net.accuracy(x_test, t_test)

            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            print(f'batch_size:{batch_size}, iter:{i}, train acc:{train_acc} test acc:{test_acc}')

    # 그래프 그리기
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()
    
# test_activation_value_distribution()
def test_activation_value_distribution():
    x = np.random.randn(1000, 100)
    node_num = 100
    h_layer_num = 5
    activations = {}
    
    for i in range(h_layer_num):
        if i != 0:
            x = activations[i-1]
        w = np.random.randn(node_num, node_num) * 1
        h = np.dot(x, w)
        y = sigmoid(h)
        activations[i] = y # 각 hidden layer마다 sigmoid 출력 값들의 분포를 한 번 봐보자
        
    # 그래프로 그려보면, 대부분의 값이 0 또는 1에 수렴하고 있음을 알게 됨
    # sigmoid의 출력 값이 0 또는 1에 수렴하면 그 미분은 0에 수렴하기 때문에 결국 학습이 잘 되지 않는다
    # 이러한 현상을 기울기 소실 현상이라고 함
    for i, a in activations.items():
        plt.subplot(1, len(activations), i+1)
        plt.title(str(i+1) + '-layer')
        plt.hist(a.flatten(), 30, range=(0, 1))
    plt.show()
    
# test
# print(softmax(np.array([9, 2, 1, 1, 4, 3, 2])))
# test_cross_entropy_error()
# test_forward()
# test_two_layer_net()
# test_mini_batch()
# test_backward_apple()
# test_backward_activation()
test_backward_affine()
# test_backward()
# test_layer_net()
# test_activation_value_distribution()
