import numpy as np

from sklearn import datasets
iris = datasets.load_iris()

x, y = iris.data, iris.target

def to_one_hot(y):
    return np.squeeze(np.eye(len(np.unique(y)))[y])

y = to_one_hot(y)

def sigm(x):
    return 1 / (1 + np.exp(-x))

def d_sigm(x):
    return sigm(x) *(1-sigm(x))

class NN():
    
    def __init__(self):
        self.w1 = np.random.uniform(-1,1,size=(4, 20))
        self.w2 = np.random.uniform(-1,1,size=(20, 3))
        self.lr = 0.001
        
    def forward(self, x):
        
        h1 = np.dot(x, self.w1)
        a_h1 = sigm(h1)
        y_hat = np.dot(a_h1, self.w2)

        return y_hat, h1, a_h1

    def backward(self,x):
        
        y_hat, h1, a_h1   = self.forward(x)
        loss = y - y_hat

        d_3 = loss                       # no derivation because of linear output
        grads_w2 = np.dot(a_h1.T, d_3)    #grads = a_i * delta_k 

        d_2 = self.w2.dot(d_3.T).T * d_sigm(h1)   # d = f'(s)
        grads_w1 = np.dot(x.T,d_2)

        self.w1 += grads_w1 * self.lr
        self.w2 += grads_w2 * self.lr

        return y_hat, np.mean(loss)

nn = NN()
for i in range(10000):
    y_hat, loss = nn.backward(x)
        
    if np.count_nonzero((np.argmax(y_hat,axis=1) - iris.target)) < 4:
        print("Iteration", i)
        print("Accuracy > 97%")
        print("Label differences:")
        print(np.argmax(y_hat,axis=1) - iris.target)
        break