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
        self.w1 = np.random.uniform(-1,1,size=(4, 10))
        self.w2 = np.random.uniform(-1,1,size=(10, 3))
        self.lr = 0.001
        
    def forward(self, x):
        
        s_j = np.dot(x, self.w1)
        a_j = sigm(s_j)
        s_k = np.dot(a_j, self.w2)
        y_hat = s_k

        return y_hat, s_j, a_j

    def backward(self,x):
        
        y_hat, s_j, a_j   = self.forward(x)
        loss = y - y_hat

        d_k = loss                       
        grads_w2 = np.dot(a_j.T, d_k)    

        d_j = self.w2.dot(d_k.T).T * d_sigm(s_j)   
        grads_w1 = np.dot(x.T,d_j)

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






