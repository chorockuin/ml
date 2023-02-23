import numpy as np

# activation functions
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

def relu(x):
    y = np.maximum(0, x)
    return y

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0) # 가로 방향으로 훑으면서 세로 값들 중에 제일 큰 값을 구함
        y = np.exp(x) / np.sum(np.exp(x), axis=0) # 가로 방향으로 훑으면서 세로 값들의 합ㅇ르 구함
        return y.T
    x = x - np.max(x) # np.exp(x)가 너무 작으면 overflow가 발생함. np.exp(x-c)를 해도 결과는 같다는 것이 증명되었으므로, x-c를 써서 overflow를 방지함
    y = np.exp(x) / np.sum(np.exp(x))
    return y

# loss functions
def cross_entropy_error(y, t): # 1차원, one-hot encoding된 데이터
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) # y가 0이면 계산이 안됨. 따라서 아주 작은 값 delta를 더해줌

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
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        tmp = x[i]
        x[i] = tmp + h
        x_plus_h = f(x)
        
        x[i] = tmp - h
        x_minus_h = f(x)
        
        grad[i] = (x_plus_h - x_minus_h) / (2 * h)
        x[i] = tmp
    return grad
    
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

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
    a1 = np.dot(x, network['w1']) + network['b1'] # ?x2 dot 2x3 + 1x3 = ?x3
    z1 = sigmoid(a1) # ?x3
    a2 = np.dot(z1, network['w2']) + network['b2'] # ?x3 dot 3x2 + 1x2 = ?x2
    z2 = sigmoid(a2) # ?x2
    a3 = np.dot(z2, network['w3']) + network['b3'] # ?x2 dot 2x2 + 1x2 = ?x2
    y = softmax(a3)
    return y

def test_forward():
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)
        
# test_two_layer_net()
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = weight_init_std * np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = weight_init_std * np.zeros(output_size)
        
    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)
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
        loss_w = lambda w: self.loss(x, t)
        
        grads = {}
        grads['w1'] = numerical_gradient(loss_w, self.params['w1'])
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grads['w2'] = numerical_gradient(loss_w, self.params['w2'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])
        return grads
        
def test_two_layer_net():
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    x = np.random.rand(100, 784)
    t = np.random.rand(100, 10)

    y = net.predict(x)
    grads = net.numerical_gradient(x, t)
    
# # test_mini_batch()
# import load_mnist

# def test_mini_batch():
#     (x_train, t_train), (x_test, t_test) = load_mnist.load_mnist(normalize=True, one_hot_label=True)
#     # x_train = (60000, 784)
#     # t_train = (60000, 10)
#     # x_test = (10000, 784)
#     # t_train = (10000, 10)

#     train_loss_list = []
#     train_acc_list = []
#     test_acc_list = []

#     train_size = x_train.shape[0] # 60000
#     batch_size = 100
#     learning_rate = 0.1
#     epoch = 10
#     # iter = max(train_size / batch_size, 1) # 60000 / 100 = 600
#     iter = 10

#     net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
#     for e in range(epoch): # 4번
#         for i in range(iter): # np.random.choice 때문에 모든 train 데이터를 사용하는 것은 아니지만, 그래도 60000개 개수는 맞추자
#             batch_indexes = np.random.choice(train_size, batch_size) # 100 indexes
#             x_batch = x_train[batch_indexes] # 100
#             t_batch = t_train[batch_indexes] # 100
            
#             grad = net.numerical_gradient(x_batch, t_batch) # 손실함수에 대해 가중치 편미분해서 기울기 얻음
            
#             for k in ('w1', 'b1', 'w2', 'b2'):
#                 net.params[k] -= learning_rate * grad[k] # 기울기 * 학습률 만큼 파라미터 업데이트

#             loss = net.loss(x_batch, t_batch) # loss 구함
#             train_loss_list.append(loss)
#             print(f'train_loss:{loss}')
#         train_acc = net.accuracy(x_train, t_train)
#         test_acc = net.accuracy(x_test, t_test)
#         train_acc_list.append(train_acc)
#         test_acc_list.append(test_acc)
#         print(f'batch_size:{batch_size}, epoch:{e}, train acc:{train_acc} test acc:{test_acc}')

# test
print(softmax(np.array([9, 2, 1, 1, 4, 3, 2])))
test_cross_entropy_error()
test_forward()
test_two_layer_net()
# test_mini_batch()