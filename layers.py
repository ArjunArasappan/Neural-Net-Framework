import numpy as np
import math

class Layer:
    
    def __init__(self, num, inputSize):
        self.size = num
        self.inputSize = inputSize
        
        self.a = None # activations: nodes x 1
        self.z = None # pre-activ: nodes x 1
        self.lastx = None # last input: inputSize x batchSize
        
        self.b = np.random.randn(num, 1) # nodes x 1
        self.one_vector = None # 1 x batchSize
    
        # nodes x input: xavier initialization
        self.weights = self.sigmoid_init(num, inputSize)
        
        
        self.activ = self.sigmoid
        self.deriv = self.sigmoid_deriv
        
    def sigmoid_init(self, i, j):
        return (2 * np.random.rand(i, j) - 1) / math.sqrt(self.inputSize) 
    
    def relu_init(self, i, j):
        return np.random.randn(i, j) / math.sqrt(math.sqrt((2 / self.size)))
        
        
    def ReLu(self, x):
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print(x)
            raise ValueError("Relu Input")
        out = np.maximum(x, 0)
        
        if np.any(np.isnan(out)) or np.any(np.isinf(out)):
            print(out)
            raise ValueError("Relu Output")
        return out
    
    def ReLu_deriv(self, x):
        return x > 0
    

    def sigmoid(self, x):
        pos_mask = (x >= 0)
        neg_mask = (x < 0)
        z = np.zeros_like(x)
        
        z[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        z[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))
        
        return z

    
    def sigmoid_deriv(self, x):
        sig = self.sigmoid(x)
        return sig * (1- sig)
    
    def softmax(self, x):
            
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)
    
    def setActivation(self, str):
        dic = {'ReLu' : self.ReLu, 'Sigm' : self.sigmoid, 'Soft' : self.softmax}
        deriv_dic = {'ReLu' : self.ReLu_deriv, 'Sigm' : self.sigmoid_deriv}
        
        self.activ = dic.get(str)
        self.deriv = deriv_dic.get(str)
        
    def forwardProp(self, x):
        
        self.lastx = x
        
        self.one_vector = np.ones(shape=(1, len(x[0]))) # 1 x input size
        
        # nodes x batchSize
        self.z = np.dot(self.weights, x) + self.b.dot(self.one_vector)
        
        if np.any(np.isnan(self.z)) or np.any(np.isinf(self.z)):
            print(self.z)
            raise ValueError("Z computation")
        
        self.a = self.activ(self.z)
        
    
        
    def backProp(self, dz, alpha): # nodes x batch
 
        dW = np.dot(dz, self.lastx.transpose()) / self.size        
        db = np.sum(dz, 1).reshape(self.size, 1) / self.size
        
        dx = np.dot(self.weights.transpose(), dz) #input x batch
        
        self.weights = self.weights - alpha * dW
        self.b = self.b - alpha * db
        
        return dx
        
        
        
        
        
        