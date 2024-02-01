from Layers import Layer
import random
import numpy as np

class NeuralNetwork:
    
    def __init__(self, numLayers, layerSizes, inputSize):
        
        if numLayers != len(layerSizes):
            raise Exception("Layer Sizes List Invalid")
        
        self.alpha = 0.05
        self.varRate = None # variable learning rate, exponential decay
        self.updateRate = 100 # update varRate every _ epochs
        
        
        self.layerCount = numLayers
        
        self.layers = []
        
        self.inputSize = inputSize
        
        for i, v in enumerate(layerSizes):
            layer = None
            
            if i == len(layerSizes) - 1:
                layer = Layer(v, layerSizes[i-1])
                layer.setActivation('Soft')
            elif i != 0:
                layer = Layer(v, layerSizes[i-1])
            else:
                layer = Layer(v, self.inputSize)
                
            self.layers.append(layer)
                        
    def setActivation(self, layerNum, act):
        self.layers[layerNum].setActivation(act)
            
    def forwardProp(self, input): # b x 784
        activations = input.transpose()
        for layer in self.layers:
            layer.forwardProp(activations)
            activations = layer.a
            
       
        return activations
    
    def updateVarRate(self, epochs):
        initial_a = 0.1
        benchRate_b = 0.05
        benchEpoch_c = 20000

        asymp_rate_d = 0.01
        
        j = np.log(benchRate_b - asymp_rate_d) - np.log(initial_a)
        j = j / -benchEpoch_c
        
        self.varRate = initial_a * np.exp(-j * epochs) + asymp_rate_d
        
    def cross_entropy_grad(self, ypred, y):
        return ypred - y # 10 x b
    
    def backProp(self, ypred, y_one_hot):
        
        error = self.cross_entropy_grad(ypred, y_one_hot) # 10 x b
        
  
        for i, l in enumerate(self.layers[::-1]):
            if i != 0:
                error = error * l.deriv(l.z)
            error = l.backProp(error, self.alpha)

            
    def one_hot(self, x):
        one_hot = np.zeros((x.size, 10))
        one_hot[np.arange(x.size), x] = 1
        return x
    

    
    def from_one_hot(self, y):
        return np.argmax(y, axis=0)
    

    def getAccuracy(self, data):
        
        y_test = data[:, 0]
        x_test = data[:, 1:]
        y_pred = self.forwardProp(x_test)
        y_pred = self.from_one_hot(y_pred)
        
        arr = (y_test - y_pred)
        c = np.count_nonzero(arr)
        return (len(arr) - c) / len(arr)
        

    def stochasticGD(self, train_data, max_iter = 100, batch_size = 3):
        
        train_arr = train_data.to_numpy()
        
        
        for epoch in range(max_iter):
            
            # if epoch % self.updateRate == 0:
            #     self.updateVarRate(epoch)
                
            batch_indx = []
            
            for i in range(batch_size):
                batch_indx.append(random.randint(0, len(train_data) - 1))
                
            x_train = []
            y_train = []
            
            for i in batch_indx:
                datum = train_arr[i]
                y_val = datum[0]
                datum = np.delete(datum, 0)
                
                                
                y_one_hot = np.zeros(10)
                y_one_hot[y_val] = 1
                
                y_train.append(y_one_hot)
                x_train.append(datum)
                
            x_train = np.array(x_train) # 784 x b
            y_train = np.array(y_train) # b x 10
            
            y_train = y_train.transpose() # 10 x b

                                    
            output = self.forwardProp(x_train)
            self.backProp(output, y_train)
            

                
            
            if epoch % 10000 == 0:
                print('Epoch:', epoch)
                print("   Accuracy:", 100 * self.getAccuracy(train_arr))
                print("   Rate:", self.varRate)
            #print(np.sum(output, axis=0))
    
    

        
    