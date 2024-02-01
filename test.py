from NeuralNet import NeuralNetwork
import pandas as pd
from sklearn.model_selection import train_test_split



data = pd.read_csv('neural-net-framework/data/mnist.csv')

network = NeuralNetwork(2, [10, 10], 784)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

network.stochasticGD(train_data, batch_size=1, max_iter=1000000)

test = test_data.to_numpy()[:, :300]
print("Accuracy:", 100 * network.getAccuracy(test_data.to_numpy()))

