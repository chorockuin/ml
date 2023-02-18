import numpy as np

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def relu(a):
    return np.maximum(0, a)

def identity(a):
    return a

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # np.exp(a)가 너무 작으면 overflow가 발생함. np.exp(a-c)를 해도 결과는 같다는 것이 증명되었으므로, a-c를 써서 overflow를 방지함
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def cross_entropy_error(out, target): # 1차원, one-hot encoding된 데이터
    delta = 1e-7
    return -np.sum(target * np.log(out + delta)) # y가 0이면 계산이 안됨. 따라서 아주 작은 값 delta를 더해줌

def CRE_batch(out, target): # 다차원, one-hot encoding된 데이터
    delta = 1e-7
    if out.ndim == 1:
        out = np.reshape(1, out.size) # 1차원 데이터일 경우, 아래 batch_size를 구하기 위해(.shape[0]을 사용하기 위해) reshape
        target = np.reshape(1, target.size)
    batch_size = out.shape[0]
    return -np.sum(target * np.log(out + delta)) / batch_size # 1개의 데이터당 평균 CRE구함

def CRE_batch_onehot_index(out, target): # 다차원, target 값 데이터가 one-hot encoding의 index
    delta = 1e-7
    if out.ndim == 1:
        out = np.reshape(1, out.size)
        target = np.reshape(1, target.size)
    batch_size = out.shape[0]
    # 정답의 index에 해당하는 신경망의 출력 값만 log 연산하면 된다는 논리
    return -np.sum(np.log(out[np.arange(batch_size), target] + delta)) / batch_size

def test_cross_entropy_error():
    target = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) 
    # 인덱스 2가 정답인데 신경망 출력도 인덱스 2의 값이 0.6으로 가장 높다. 정답에 가까우므로 CRE값은 0에 가깝겠지?
    out1 = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]) 
    # 인덱스 2가 정답인데 신경망 출력은 인덱스 3의 값, 0.6으로 가장 높다. 정답에 가깝지 않으니 CRE값은 0에서 멀겠지?
    out2 = np.array([0.1, 0.05, 0.0, 0.6, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]) 
    print(cross_entropy_error(out1, target)) # 0.5 정도 나온다
    print(cross_entropy_error(out2, target)) # 16 정도 나온다

    target = np.array([[0, 1, 0, 0], [1, 0, 0, 0]])
    out = np.array([[0.0, 0.6, 0.2, 0.2], [0.9, 0.0, 0.0, 0.1]])
    # CRE 값은 np.sum()으로 구해지므로, 2차원 이상의 t, y를 넣어도 결과 값은 행렬의 모든 원소의 CRE값을 다 더한 scalar 값임
    print(cross_entropy_error(out, target)) # CRE의 총합
    print(CRE_batch(out, target)) # 데이터 1개의 평균 CRE

    target = np.array([1, 0])
    out = np.array([[0.0, 0.6, 0.2, 0.2], [0.9, 0.0, 0.0, 0.1]])
    print(CRE_batch_onehot_index(out, target)) # 데이터 1개의 평균 CRE

test_cross_entropy_error()

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